"""Stage 11: Monitoring - drift detection against an explicit baseline.

Numeric columns: two-sample Kolmogorov-Smirnov statistic.
Categorical columns: PSI (Population Stability Index).
Schema: explicit schema comparison.

Small samples below MIN_SAMPLE_SIZE are skipped and reported as
insufficient rather than producing misleading drift alerts.
"""

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sqlalchemy.orm import Session

from app.models import DatasetVersion, StoredProfile
from app.services.rule_execution import _load_table
from app.services import storage as storage_module

MIN_SAMPLE_SIZE = 30
PSI_THRESHOLD = 0.2
KS_THRESHOLD = 0.15

MAX_BINS = 10
MAX_COLUMNS = 40


def _psi(expected: np.ndarray, actual: np.ndarray) -> float:
    """Population Stability Index between two categorical distributions."""
    expected = expected[expected > 0]
    actual = actual[actual > 0]

    if len(expected) == 0 or len(actual) == 0:
        return 0.0

    expected_ratio = expected / expected.sum()
    actual_ratio = actual / actual.sum()

    psi = np.sum(
        (actual_ratio - expected_ratio)
        * np.log(actual_ratio / expected_ratio)
    )

    return float(psi)


def _numeric_ks(
    baseline_values: pd.Series,
    comparison_values: pd.Series,
) -> float:
    statistic, _ = scipy_stats.ks_2samp(
        baseline_values.dropna().to_numpy(),
        comparison_values.dropna().to_numpy(),
    )
    return float(statistic)


def _categorical_psi(
    baseline_values: pd.Series,
    comparison_values: pd.Series,
) -> float:
    baseline_counts = baseline_values.value_counts(normalize=True)
    comparison_counts = comparison_values.value_counts(normalize=True)

    all_categories = set(baseline_counts.index) | set(comparison_counts.index)

    expected = np.array(
        [baseline_counts.get(category, 0.0) for category in all_categories]
    )
    actual = np.array(
        [comparison_counts.get(category, 0.0) for category in all_categories]
    )

    # Smooth to avoid log(0).
    epsilon = 1e-6
    expected = expected + epsilon
    actual = actual + epsilon

    return _psi(expected, actual)


def _stored_file(
    db: Session,
    version_id: int,
) -> str | None:
    from app.models import FileMetadata

    metadata = (
        db.query(FileMetadata)
        .filter(FileMetadata.version_id == version_id)
        .first()
    )

    return metadata.stored_filename if metadata else None


def compare_versions(
    db: Session,
    dataset_id: int,
    baseline_version_id: int,
    comparison_version_id: int,
) -> dict:
    """Compare a version against an explicit baseline version."""
    baseline = db.get(DatasetVersion, baseline_version_id)
    comparison = db.get(DatasetVersion, comparison_version_id)

    if baseline is None or comparison is None:
        raise ValueError("Baseline or comparison version not found.")

    results = {
        "baseline_version_id": baseline_version_id,
        "baseline_version_number": baseline.version_number,
        "comparison_version_id": comparison_version_id,
        "comparison_version_number": comparison.version_number,
        "schema_changes": [],
        "column_drift": [],
        "insufficient_sample": [],
        "methods": {
            "numeric": "Kolmogorov-Smirnov two-sample statistic",
            "categorical": "Population Stability Index (PSI)",
            "schema": "explicit schema fingerprint comparison",
        },
    }

    # Schema comparison.
    if baseline.schema_fingerprint != comparison.schema_fingerprint:
        results["schema_changes"].append(
            {
                "type": "schema_fingerprint_change",
                "baseline": baseline.schema_fingerprint,
                "comparison": comparison.schema_fingerprint,
            }
        )

    baseline_file = _stored_file(db, baseline_version_id)
    comparison_file = _stored_file(db, comparison_version_id)

    if baseline_file is None or comparison_file is None:
        results["error"] = "Raw files not available for drift comparison."
        return results

    try:
        baseline_df = _load_table(
            dataset_id, baseline.version_number, baseline_file
        )
        comparison_df = _load_table(
            dataset_id, comparison.version_number, comparison_file
        )
    except Exception as exc:
        results["error"] = f"Could not load raw files: {exc}"
        return results

    common_columns = [
        column
        for column in baseline_df.columns
        if column in comparison_df.columns
    ][:MAX_COLUMNS]

    for column in common_columns:
        base_series = baseline_df[column].dropna()
        comp_series = comparison_df[column].dropna()

        if len(base_series) < MIN_SAMPLE_SIZE or len(comp_series) < MIN_SAMPLE_SIZE:
            results["insufficient_sample"].append(
                {
                    "column": column,
                    "baseline_count": int(len(base_series)),
                    "comparison_count": int(len(comp_series)),
                    "minimum_required": MIN_SAMPLE_SIZE,
                }
            )
            continue

        if pd.api.types.is_numeric_dtype(base_series) and pd.api.types.is_numeric_dtype(comp_series):
            statistic = _numeric_ks(base_series, comp_series)
            threshold = KS_THRESHOLD

            results["column_drift"].append(
                {
                    "column": column,
                    "method": "ks",
                    "statistic": round(statistic, 4),
                    "threshold": threshold,
                    "drift_detected": statistic > threshold,
                }
            )

        else:
            base_text = base_series.astype(str)
            comp_text = comp_series.astype(str)

            statistic = _categorical_psi(base_text, comp_text)
            threshold = PSI_THRESHOLD

            results["column_drift"].append(
                {
                    "column": column,
                    "method": "psi",
                    "statistic": round(statistic, 4),
                    "threshold": threshold,
                    "drift_detected": statistic > threshold,
                }
            )

    drift_count = sum(
        1 for item in results["column_drift"] if item["drift_detected"]
    )

    results["summary"] = {
        "columns_compared": len(results["column_drift"]),
        "drift_columns": drift_count,
        "schema_changed": bool(results["schema_changes"]),
        "insufficient_sample_count": len(results["insufficient_sample"]),
    }

    return results


def run_monitoring(
    db: Session,
    dataset_id: int,
    baseline_version_id: int,
    comparison_version_id: int,
) -> dict:
    """Run drift monitoring and persist results."""
    from app.models import DriftResult
    from datetime import datetime, timezone

    result = compare_versions(
        db,
        dataset_id,
        baseline_version_id,
        comparison_version_id,
    )

    now = datetime.now(timezone.utc)

    for item in result.get("column_drift", []):
        db.add(
            DriftResult(
                dataset_id=dataset_id,
                baseline_version_id=baseline_version_id,
                comparison_version_id=comparison_version_id,
                method=item["method"],
                column_name=item["column"],
                table_name="",
                statistic=item["statistic"],
                threshold=item["threshold"],
                drift_detected=1 if item["drift_detected"] else 0,
                details_json={},
                computed_at=now,
            )
        )

    db.flush()

    return result

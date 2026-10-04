"""Stage 10: Remediation.

Controlled remediation: only safe, deterministic, formatting-level
corrections are automated (whitespace trimming). Everything else requires
human approval and is never guessed semantically. Remediation never
overwrites a version - it creates a new immutable version.
"""

import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sqlalchemy.orm import Session

from app.models import (
    FileMetadata,
    RemediationApproval,
    RemediationRecord,
    Rule,
    RuleExecution,
    TableMetadata,
)
from app.services.rule_execution import (
    RuleExecutionError,
    _load_table,
    execute_rule_on_dataframe,
)
from app.services import storage as storage_module
from app.services.versioning import create_dataset_version
from app.services.dataset_fingerprinting import (
    calculate_dataset_content_fingerprint,
    calculate_schema_fingerprint,
)
from app.services.ingestion import discover_tables

SAFE_REMEDIATION_TYPES = {"whitespace_normalization"}


def _is_text_column(series: pd.Series) -> bool:
    """True for object or pandas string dtype columns."""
    return (
        series.dtype == object
        or pd.api.types.is_string_dtype(series)
    )


def _normalize_whitespace_value(value):
    """Trim leading/trailing whitespace from strings only."""
    if isinstance(value, str):
        return value.strip()
    return value


def propose_remediation(
    db: Session,
    dataset_id: int,
    version_id: int,
    version_number: int,
) -> dict:
    """Scan for safe, deterministic corrections and build a proposal."""
    rules = (
        db.query(Rule)
        .filter(
            Rule.version_id == version_id,
            Rule.status == "approved",
        )
        .all()
    )

    tables = (
        db.query(TableMetadata)
        .filter(TableMetadata.version_id == version_id)
        .all()
    )

    corrections: list[dict] = []
    affected_rows_total = set()

    for table in tables:
        stored_filename = _stored_file_for_table(db, version_id, table.table_name)

        if stored_filename is None:
            continue

        try:
            dataframe = _load_table(dataset_id, version_number, stored_filename)
        except RuleExecutionError:
            continue

        changed_mask = pd.Series(False, index=dataframe.index)

        for column in dataframe.columns:
            if not _is_text_column(dataframe[column]):
                continue

            original = dataframe[column]
            normalized = original.map(_normalize_whitespace_value)

            differs = original.map(_value_repr) != normalized.map(_value_repr)

            if differs.any():
                changed_mask = changed_mask | differs

                sample_diffs = []
                for index in dataframe.index[differs][:5]:
                    sample_diffs.append(
                        {
                            "row_index": int(index),
                            "column": column,
                            "before": _value_repr(original.loc[index]),
                            "after": _value_repr(normalized.loc[index]),
                        }
                    )

                corrections.append(
                    {
                        "table": table.table_name,
                        "file": stored_filename,
                        "column": column,
                        "affected_rows": int(differs.sum()),
                        "remediation_type": "whitespace_normalization",
                        "samples": sample_diffs,
                    }
                )

        for index in dataframe.index[changed_mask]:
            affected_rows_total.add((table.table_name, int(index)))

    total_corrections = sum(
        correction["affected_rows"] for correction in corrections
    )

    return {
        "dataset_id": dataset_id,
        "version_id": version_id,
        "remediation_type": "whitespace_normalization",
        "corrections": corrections,
        "total_corrections": total_corrections,
        "affected_rows": len(affected_rows_total),
        "risk": "low",
        "requires_approval": total_corrections > 0,
        "safety_note": (
            "Only leading/trailing whitespace trimming is proposed. Values "
            "are never replaced semantically; malformed identifiers, category "
            "values or financial amounts always require source-system fixes."
        ),
    }


def _value_repr(value) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value)


def _stored_file_for_table(
    db: Session,
    version_id: int,
    table_key,
) -> str | None:
    """Resolve the stored filename for a table by id or name."""
    if isinstance(table_key, int):
        table = db.get(TableMetadata, table_key)
    else:
        table = (
            db.query(TableMetadata)
            .filter(
                TableMetadata.version_id == version_id,
                TableMetadata.table_name == table_key,
            )
            .first()
        )

    if table is None:
        return None

    files = (
        db.query(FileMetadata)
        .filter(FileMetadata.version_id == version_id)
        .all()
    )

    for metadata in files:
        if metadata.original_filename == table.source_file:
            return metadata.stored_filename

    return None


def apply_remediation(
    db: Session,
    dataset_id: int,
    version_id: int,
    version_number: int,
    proposal: dict,
) -> dict:
    """Apply approved whitespace normalization and create a new version."""
    tables = (
        db.query(TableMetadata)
        .filter(TableMetadata.version_id == version_id)
        .all()
    )

    new_version = create_dataset_version(
        db=db,
        dataset_id=dataset_id,
        parent_version_id=version_id,
        schema_fingerprint="pending",
        content_fingerprint="pending",
    )

    new_version_number = new_version.version_number

    output_dir = (
        storage_module.RAW_DATA_DIR
        / str(dataset_id)
        / f"v{new_version_number}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    file_fingerprints: list[tuple[str, str]] = []
    all_tables: list[dict] = []

    total_corrections = 0
    affected_rows = 0

    from app.services.fingerprinting import calculate_file_fingerprint
    from app.services.version_snapshot import register_version_snapshot

    snapshot_files: list[dict] = []

    for table in tables:
        stored_filename = _stored_file_for_table(db, version_id, table.table_name)

        if stored_filename is None:
            continue

        source_path = (
            storage_module.RAW_DATA_DIR
            / str(dataset_id)
            / f"v{version_number}"
            / stored_filename
        )

        if not source_path.exists():
            continue

        dataframe = _load_table(dataset_id, version_number, stored_filename)

        correction_count = 0

        for column in dataframe.columns:
            if not _is_text_column(dataframe[column]):
                continue

            original = dataframe[column]
            normalized = original.map(_normalize_whitespace_value)

            differs = original.map(_value_repr) != normalized.map(_value_repr)
            correction_count += int(differs.sum())

            if differs.any():
                dataframe[column] = normalized

        total_corrections += correction_count
        affected_rows += (
            int((dataframe.apply(lambda row: True, axis=1)).sum())
            if correction_count
            else 0
        )

        # Write the corrected file under the SAME stored filename as the
        # parent version so file metadata and table lookups stay consistent.
        final_path = output_dir / stored_filename

        if stored_filename.lower().endswith(".csv"):
            dataframe.to_csv(final_path, index=False)
        elif stored_filename.lower().endswith((".xls", ".xlsx")):
            dataframe.to_excel(final_path, index=False)
        elif stored_filename.lower().endswith(".parquet"):
            dataframe.to_parquet(final_path, index=False)
        else:
            shutil.copy2(source_path, final_path)

        fingerprint = calculate_file_fingerprint(final_path)
        file_fingerprints.append((stored_filename, fingerprint))

        discovered = discover_tables(final_path)
        all_tables.extend(discovered)

        snapshot_files.append(
            {
                "filename": stored_filename,
                "source_path": final_path,
                "fingerprint": fingerprint,
                "tables": discovered,
            }
        )

    schema_fingerprint = calculate_schema_fingerprint(all_tables)
    content_fingerprint = calculate_dataset_content_fingerprint(file_fingerprints)

    new_version.schema_fingerprint = schema_fingerprint
    new_version.content_fingerprint = content_fingerprint

    register_version_snapshot(
        db=db,
        dataset_id=dataset_id,
        version_id=new_version.version_id,
        version_number=new_version_number,
        files=snapshot_files,
    )

    db.flush()

    # Carry approved semantic decisions into the new version so the
    # pipeline (recommendations, rules, scores) continues seamlessly.
    from app.models import SemanticPrediction

    approved_predictions = (
        db.query(SemanticPrediction)
        .filter(
            SemanticPrediction.version_id == version_id,
            SemanticPrediction.status.in_(["approved", "edited"]),
        )
        .all()
    )

    for prediction in approved_predictions:
        db.add(
            SemanticPrediction(
                version_id=new_version.version_id,
                table_name=prediction.table_name,
                column_name=prediction.column_name,
                predicted_concept_id=prediction.user_confirmed_concept_id,
                predicted_concept=prediction.predicted_concept,
                confidence_score=prediction.confidence_score,
                confidence_level=prediction.confidence_level,
                alternatives_json=prediction.alternatives_json,
                evidence_json=prediction.evidence_json,
                status="approved",
                user_confirmed_concept_id=prediction.user_confirmed_concept_id,
                user_defined_type=prediction.user_defined_type,
                decision_source=prediction.decision_source,
                decided_at=prediction.decided_at,
            )
        )

    db.flush()

    return {
        "new_version_id": new_version.version_id,
        "new_version_number": new_version_number,
        "correction_count": total_corrections,
        "affected_rows": affected_rows,
    }

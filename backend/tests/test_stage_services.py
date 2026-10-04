"""Tests for semantic stage, offline learning, monitoring and remediation."""

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.models import (
    Dataset,
    DatasetVersion,
    SemanticFeedback,
    SemanticPrediction,
    StoredProfile,
)


def make_db():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(bind=engine)
    return sessionmaker(bind=engine)()


# ---------------------------------------------------------------------------
# Semantic confidence mapping
# ---------------------------------------------------------------------------


def test_confidence_requires_evidence_coverage():
    from app.services.semantic_stage import confidence_level

    # High score but no evidence coverage must not read as Strong.
    assert confidence_level(0.95, 0.0) == "Ambiguous"
    assert confidence_level(0.95, 0.8) == "Strong"
    assert confidence_level(0.7, 0.8) == "Ambiguous"
    assert confidence_level(0.3, 0.9) == "Unknown"


# ---------------------------------------------------------------------------
# Offline learning
# ---------------------------------------------------------------------------


def _seed_feedback(db, count, label_split):
    """Seed feedback rows with features; label_split = (positives, negatives)."""
    dataset = Dataset(dataset_name="D")
    db.add(dataset)
    db.flush()

    version = DatasetVersion(
        dataset_id=dataset.dataset_id,
        version_number=1,
        schema_fingerprint="s",
        content_fingerprint="c",
    )
    db.add(version)
    db.flush()

    features = {
        "embedding_similarity": 0.8,
        "name_similarity": 0.9,
        "description_similarity": 0.1,
        "datatype_compatibility": 1.0,
        "profile_compatibility": 1.0,
        "context_similarity": 0.0,
    }

    positives, negatives = label_split

    for index in range(positives + negatives):
        decision = "approved" if index < positives else "rejected"

        prediction = SemanticPrediction(
            version_id=version.version_id,
            table_name="customers",
            column_name=f"col_{index}",
            predicted_concept_id=1,
            predicted_concept="Customer ID",
            confidence_score=0.8,
            confidence_level="Strong",
            status=decision,
        )
        db.add(prediction)
        db.flush()

        db.add(
            SemanticFeedback(
                dataset_id=dataset.dataset_id,
                version_id=version.version_id,
                prediction_id=prediction.prediction_id,
                table_name="customers",
                column_name=f"col_{index}",
                original_concept_id=None,
                corrected_concept_id=1 if decision == "approved" else None,
                decision=decision,
                features_json=features,
                model_name="all-MiniLM-L6-v2",
            )
        )

    db.commit()
    return dataset, version


def test_training_refuses_insufficient_feedback():
    from app.services.offline_learning import train_model

    db = make_db()
    _seed_feedback(db, 5, (3, 2))

    result = train_model(db)

    assert result["status"] == "insufficient_data"
    assert result["example_count"] == 5


def test_training_refuses_single_class():
    from app.services.offline_learning import train_model

    db = make_db()
    _seed_feedback(db, 25, (25, 0))

    result = train_model(db)

    assert result["status"] == "insufficient_data"
    assert "one class" in result["message"]


def test_training_creates_candidate_model_version():
    from app.services.offline_learning import train_model, list_models

    db = make_db()
    _seed_feedback(db, 30, (15, 15))

    result = train_model(db)

    assert result["status"] == "trained"

    models = list_models(db)

    assert models[0]["status"] == "candidate"
    assert models[0]["metrics"]["example_count"] == 30


def test_promotion_requires_human_action_and_archives_previous():
    from app.services.offline_learning import (
        train_model,
        promote_model,
        list_models,
    )

    db = make_db()
    _seed_feedback(db, 30, (15, 15))

    first = train_model(db)
    second = train_model(db)

    promote_model(db, first["model_version_id"])

    models = list_models(db)
    by_version = {m["version"]: m for m in models}

    assert by_version[first["version"]]["status"] == "promoted"
    assert by_version[second["version"]]["status"] == "candidate"

    # Promoting the second demotes the first.
    promote_model(db, second["model_version_id"])

    models = list_models(db)
    by_version = {m["version"]: m for m in models}

    assert by_version[first["version"]]["status"] == "archived"
    assert by_version[second["version"]]["status"] == "promoted"


# ---------------------------------------------------------------------------
# Monitoring / drift
# ---------------------------------------------------------------------------


def test_psi_detects_category_shift():
    from app.services.monitoring import _categorical_psi

    baseline = pd.Series(["A"] * 50 + ["B"] * 50)
    shifted = pd.Series(["A"] * 10 + ["B"] * 90)

    statistic = _categorical_psi(baseline.astype(str), shifted.astype(str))

    assert statistic > 0.2


def test_psi_stable_distribution_is_low():
    from app.services.monitoring import _categorical_psi

    baseline = pd.Series(["A"] * 50 + ["B"] * 50)
    same = pd.Series(["A"] * 50 + ["B"] * 50)

    statistic = _categorical_psi(baseline.astype(str), same.astype(str))

    assert statistic < 0.05


def test_ks_numeric_detection():
    from app.services.monitoring import _numeric_ks

    baseline = pd.Series(np.random.default_rng(1).normal(0, 1, 500))
    shifted = pd.Series(np.random.default_rng(2).normal(5, 1, 500))

    statistic = _numeric_ks(baseline, shifted)

    assert statistic > 0.9


def test_compare_versions_flags_small_samples():
    from app.services.monitoring import compare_versions

    db = make_db()

    dataset = Dataset(dataset_name="D")
    db.add(dataset)
    db.flush()

    v1 = DatasetVersion(
        dataset_id=dataset.dataset_id,
        version_number=1,
        schema_fingerprint="s1",
        content_fingerprint="c1",
    )
    v2 = DatasetVersion(
        dataset_id=dataset.dataset_id,
        version_number=2,
        parent_version_id=v1.version_id,
        schema_fingerprint="s1",
        content_fingerprint="c2",
    )
    db.add_all([v1, v2])
    db.commit()

    result = compare_versions(db, dataset.dataset_id, v1.version_id, v2.version_id)

    # No raw files: comparison reports the error honestly.
    assert "error" in result or result["summary"]["columns_compared"] == 0


# ---------------------------------------------------------------------------
# Remediation
# ---------------------------------------------------------------------------


def test_whitespace_normalization_detects_only_safe_changes():
    from app.services.remediation import (
        _is_text_column,
        _normalize_whitespace_value,
    )

    dataframe = pd.DataFrame(
        {
            "name": pd.Series([" Alice ", "Bob", None], dtype="object"),
            "amount": pd.Series([10, 20, 30]),
        }
    )

    assert _is_text_column(dataframe["name"])
    assert not _is_text_column(dataframe["amount"])

    normalized = dataframe["name"].map(_normalize_whitespace_value)

    assert normalized.iloc[0] == "Alice"
    assert normalized.iloc[1] == "Bob"
    assert pd.isna(normalized.iloc[2])


def test_propose_remediation_requires_existing_version():
    from app.services.remediation import propose_remediation

    db = make_db()
    dataset = Dataset(dataset_name="D")
    db.add(dataset)
    db.flush()

    version = DatasetVersion(
        dataset_id=dataset.dataset_id,
        version_number=1,
        schema_fingerprint="s",
        content_fingerprint="c",
    )
    db.add(version)
    db.commit()

    # No tables/files: proposal is empty, not fabricated.
    proposal = propose_remediation(
        db, dataset.dataset_id, version.version_id, 1
    )

    assert proposal["total_corrections"] == 0
    assert proposal["requires_approval"] is False


# ---------------------------------------------------------------------------
# Profile persistence + stage state
# ---------------------------------------------------------------------------


def test_stage_progress_tracks_completion():
    from app.services.stage_state import (
        get_stage_progress,
        mark_stage_complete,
    )

    db = make_db()
    dataset = Dataset(dataset_name="D")
    db.add(dataset)
    db.commit()

    progress = get_stage_progress(db, dataset.dataset_id)
    assert progress["first_incomplete_stage"] == "ingestion"

    mark_stage_complete(db, dataset.dataset_id, "ingestion")
    mark_stage_complete(db, dataset.dataset_id, "context")

    progress = get_stage_progress(db, dataset.dataset_id)

    assert progress["first_incomplete_stage"] == "version"
    assert progress["completed_count"] == 2


def test_mark_stage_complete_is_idempotent():
    from app.services.stage_state import mark_stage_complete, get_stage_progress

    db = make_db()
    dataset = Dataset(dataset_name="D")
    db.add(dataset)
    db.commit()

    mark_stage_complete(db, dataset.dataset_id, "profiling")
    mark_stage_complete(db, dataset.dataset_id, "profiling")

    progress = get_stage_progress(db, dataset.dataset_id)
    assert progress["completed_count"] == 1

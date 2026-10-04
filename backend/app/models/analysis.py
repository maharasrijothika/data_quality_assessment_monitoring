from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, ForeignKey, Integer, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class StoredProfile(Base):
    """Persisted profiling result for one dataset version."""

    __tablename__ = "stored_profiles"

    profile_id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        index=True,
    )

    version_id: Mapped[int] = mapped_column(
        ForeignKey("dataset_versions.version_id"),
        nullable=False,
        index=True,
    )

    profile_json: Mapped[dict] = mapped_column(
        JSON,
        nullable=False,
    )

    row_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_utcnow,
    )


class SemanticPrediction(Base):
    """Persisted semantic analysis for one column of one version.

    user_confirmed_concept_id stores the human decision, which may differ
    from the model prediction. status tracks the review workflow:
    pending / approved / rejected / edited.
    """

    __tablename__ = "semantic_predictions"

    prediction_id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        index=True,
    )

    version_id: Mapped[int] = mapped_column(
        ForeignKey("dataset_versions.version_id"),
        nullable=False,
        index=True,
    )

    table_name: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    column_name: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    predicted_concept_id: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    predicted_concept: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    confidence_score: Mapped[float] = mapped_column(
        nullable=False,
        default=0.0,
    )

    confidence_level: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        default="Unknown",
    )

    alternatives_json: Mapped[dict] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )

    evidence_json: Mapped[dict] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )

    status: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        default="pending",
    )

    user_confirmed_concept_id: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    # User-defined semantic type (free text) recorded when the human decision
    # does not map to any existing KB concept. The original recommendation
    # (predicted_concept) is never overwritten by this value.
    user_defined_type: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    # How the final type was reached: kb_approval | user_defined | rejection.
    decision_source: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    # Timestamp of the human decision (None while pending).
    decided_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_utcnow,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_utcnow,
    )


class SemanticRun(Base):
    """Append-only semantic run record (audit + S1-vs-S2 comparison).

    Every semantic analysis run for a dataset version is RECORDED, never
    overwritten: run_number increments, and the full per-column input
    snapshot (including the exact embedding text) and prediction outcome
    are kept so any logic/KB/model change produces a comparable new run.
    """

    __tablename__ = "semantic_runs"

    run_id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        index=True,
    )

    dataset_id: Mapped[int] = mapped_column(
        ForeignKey("datasets.dataset_id"),
        nullable=False,
        index=True,
    )

    version_id: Mapped[int] = mapped_column(
        ForeignKey("dataset_versions.version_id"),
        nullable=False,
        index=True,
    )

    run_number: Mapped[int] = mapped_column(Integer, nullable=False)

    model_version: Mapped[str] = mapped_column(Text, nullable=False)

    embedding_representation_version: Mapped[int] = mapped_column(
        Integer, nullable=False
    )

    evidence_version: Mapped[int] = mapped_column(Integer, nullable=False)

    kb_fingerprint: Mapped[str] = mapped_column(Text, nullable=False)

    configuration_json: Mapped[dict] = mapped_column(JSON, nullable=False)

    semantic_input_json: Mapped[dict] = mapped_column(JSON, nullable=False)

    predictions_json: Mapped[dict] = mapped_column(JSON, nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=_utcnow,
    )


class SemanticCandidateRecord(Base):
    """Persisted alternative candidates for a semantic prediction."""

    __tablename__ = "semantic_candidate_records"

    candidate_id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        index=True,
    )

    prediction_id: Mapped[int] = mapped_column(
        ForeignKey("semantic_predictions.prediction_id"),
        nullable=False,
        index=True,
    )

    concept_id: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    concept_name: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    confidence_score: Mapped[float] = mapped_column(
        nullable=False,
        default=0.0,
    )

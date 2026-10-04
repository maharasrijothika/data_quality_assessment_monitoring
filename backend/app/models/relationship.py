"""Relationship discovery models.

Relationship discovery produces CANDIDATES with deterministic evidence.
Only human-approved relationships may become authoritative referential
integrity checks. Discovery is NOT RI validation: a candidate with low
containment may still be a real relationship whose child table contains
orphan records.
"""

from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class RelationshipCandidate(Base):
    """A discovered (parent -> child) relationship candidate with evidence."""

    __tablename__ = "relationship_candidates"

    relationship_id: Mapped[int] = mapped_column(
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

    # Parent (candidate primary-key side).
    parent_table: Mapped[str] = mapped_column(Text, nullable=False)
    parent_column: Mapped[str] = mapped_column(Text, nullable=False)

    # Child (candidate foreign-key side).
    child_table: Mapped[str] = mapped_column(Text, nullable=False)
    child_column: Mapped[str] = mapped_column(Text, nullable=False)

    # "pk_candidate" evidence for the parent side; always present for now.
    parent_role: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        default="pk_candidate",
    )

    # relationship (default) = cross-table FK candidate; pk = single-column
    # key candidate; composite_pk = multi-column key candidate.
    # Human-declared entries use the same vocabulary as discovery output.
    candidate_kind: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        default="relationship",
    )

    # Deterministic weighted evidence score (NOT a probability).
    score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)

    # Per-feature evidence: {feature: {"value": float, "applicable": bool}}.
    evidence_json: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)

    # Additional deterministic observations (containment counts etc.).
    stats_json: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)

    # pending | approved | rejected | missed | manual
    status: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        default="pending",
    )

    # Which side the human marked: e.g. "orphan_heavy" notes.
    human_note: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_utcnow,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_utcnow,
        onupdate=_utcnow,
    )


class RelationshipFeedback(Base):
    """Human decision feedback for offline relationship model evaluation.

    Feedback goes to the feedback store only - it never directly modifies
    discovery behaviour or promotes candidates on its own.
    """

    __tablename__ = "relationship_feedback"

    feedback_id: Mapped[int] = mapped_column(
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

    relationship_id: Mapped[int | None] = mapped_column(
        ForeignKey("relationship_candidates.relationship_id"),
        nullable=True,
    )

    parent_table: Mapped[str] = mapped_column(Text, nullable=False)
    parent_column: Mapped[str] = mapped_column(Text, nullable=False)
    child_table: Mapped[str] = mapped_column(Text, nullable=False)
    child_column: Mapped[str] = mapped_column(Text, nullable=False)

    # approved | rejected | edited | missed | manual_add
    decision: Mapped[str] = mapped_column(Text, nullable=False)

    evidence_json: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_utcnow,
    )

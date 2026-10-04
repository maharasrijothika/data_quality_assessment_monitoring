from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, ForeignKey, Integer, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class KnowledgeBaseVersion(Base):
    """Versioned knowledge base admission records.

    A candidate alias/concept becomes permanent only after explicit
    approval with repeated evidence. Every change is auditable.
    """

    __tablename__ = "knowledge_base_versions"

    kb_entry_id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        index=True,
    )

    concept_id: Mapped[int | None] = mapped_column(
        ForeignKey("semantic_concepts.concept_id"),
        nullable=True,
        index=True,
    )

    kb_version: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    action: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    alias_text: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    evidence_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    status: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        default="pending",
    )

    proposed_by: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        default="system",
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_utcnow,
    )


class ConceptAlias(Base):
    """Approved aliases attached to concepts, with admission evidence."""

    __tablename__ = "concept_aliases"

    alias_id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        index=True,
    )

    concept_id: Mapped[int] = mapped_column(
        ForeignKey("semantic_concepts.concept_id"),
        nullable=False,
        index=True,
    )

    alias_text: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    evidence_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    source: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        default="seed",
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_utcnow,
    )

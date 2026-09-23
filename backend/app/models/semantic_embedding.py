from datetime import datetime, timezone

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class SemanticConceptEmbedding(Base):
    __tablename__ = "semantic_concept_embeddings"

    embedding_id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        index=True,
    )

    concept_id: Mapped[int] = mapped_column(
        ForeignKey("semantic_concepts.concept_id"),
        nullable=False,
        unique=True,
        index=True,
    )

    model_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    embedding: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
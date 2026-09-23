from datetime import datetime, timezone

from sqlalchemy import DateTime, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class SemanticConcept(Base):
    __tablename__ = "semantic_concepts"

    concept_id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        index=True,
    )

    concept_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        unique=True,
        index=True,
    )

    category: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    description: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    aliases: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    expected_data_types: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    profile_expectations: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
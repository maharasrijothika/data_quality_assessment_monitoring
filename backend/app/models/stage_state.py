from datetime import datetime, timezone

from sqlalchemy import DateTime, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class StageState(Base):
    """Backend-persisted stage completion per dataset.

    stage_key is the canonical stage identifier
    (e.g. "ingestion", "context", "profiling", "semantic", ...).
    """

    __tablename__ = "stage_states"
    __table_args__ = (
        UniqueConstraint(
            "dataset_id",
            "stage_key",
            name="uq_stage_states_dataset_stage",
        ),
    )

    stage_state_id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        index=True,
    )

    dataset_id: Mapped[int] = mapped_column(
        ForeignKey("datasets.dataset_id"),
        nullable=False,
        index=True,
    )

    stage_key: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
    )

    completed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utcnow,
        nullable=False,
    )

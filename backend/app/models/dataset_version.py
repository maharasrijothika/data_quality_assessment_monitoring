from datetime import datetime, timezone

from sqlalchemy import DateTime, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class DatasetVersion(Base):
    __tablename__ = "dataset_versions"

    version_id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        index=True,
    )

    dataset_id: Mapped[int] = mapped_column(
        ForeignKey("datasets.dataset_id"),
        nullable=False,
        index=True,
    )

    parent_version_id: Mapped[int | None] = mapped_column(
        ForeignKey("dataset_versions.version_id"),
        nullable=True,
    )

    version_number: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    schema_fingerprint: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
    )

    content_fingerprint: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
    )

    uploaded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
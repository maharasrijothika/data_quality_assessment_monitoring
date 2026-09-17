from datetime import datetime, timezone

from sqlalchemy import DateTime, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class FileMetadata(Base):
    __tablename__ = "file_metadata"

    file_id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        index=True,
    )

    version_id: Mapped[int] = mapped_column(
        ForeignKey("dataset_versions.version_id"),
        nullable=False,
        index=True,
    )

    original_filename: Mapped[str] = mapped_column(
        String(500),
        nullable=False,
    )

    stored_filename: Mapped[str] = mapped_column(
        String(500),
        nullable=False,
    )

    content_fingerprint: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        index=True,
    )

    file_extension: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
    )

    uploaded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
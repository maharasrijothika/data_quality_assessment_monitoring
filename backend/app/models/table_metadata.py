from sqlalchemy import ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class TableMetadata(Base):
    __tablename__ = "table_metadata"

    table_id: Mapped[int] = mapped_column(
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
        String(255),
        nullable=False,
    )

    source_file: Mapped[str] = mapped_column(
        String(500),
        nullable=False,
    )

    sheet_name: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )

    row_count: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )
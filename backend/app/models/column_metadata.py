from sqlalchemy import ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class ColumnMetadata(Base):
    __tablename__ = "column_metadata"

    column_id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        index=True,
    )

    table_id: Mapped[int] = mapped_column(
        ForeignKey("table_metadata.table_id"),
        nullable=False,
        index=True,
    )

    column_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    data_type: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    description: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )
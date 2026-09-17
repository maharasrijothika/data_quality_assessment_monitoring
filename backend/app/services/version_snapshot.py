
from pathlib import Path

from sqlalchemy.orm import Session

from app.models import (
    ColumnMetadata,
    FileMetadata,
    TableMetadata,
)
from app.services.storage import store_raw_file


def register_version_snapshot(
    db: Session,
    dataset_id: int,
    version_id: int,
    version_number: int,
    files: list[dict],
) -> list[dict]:
    """
    Register the complete metadata snapshot for a dataset version
    and store its immutable raw files under the corresponding
    version directory.

    Example:

        dataset_id=1
        version_number=2

    Raw files are stored under:

        data/raw/datasets/1/v2/

    The caller controls the database transaction.

    If storing or registering any file fails, all physical files
    created by this snapshot operation are removed.

    Database rollback remains the responsibility of the caller.
    """

    registered_files: list[dict] = []

    stored_paths: list[Path] = []

    try:
        for file_data in files:

            filename = file_data["filename"]
            fingerprint = file_data["fingerprint"]
            tables = file_data["tables"]

            source_path = file_data.get(
                "source_path"
            )

            if source_path is None:
                raise ValueError(
                    f"source_path is required for {filename}."
                )

            # -----------------------------------------------------
            # Store immutable raw file
            # -----------------------------------------------------

            stored_path = store_raw_file(
                source_path=source_path,
                dataset_id=dataset_id,
                version_number=version_number,
                filename=filename,
            )

            stored_paths.append(
                stored_path
            )

            # -----------------------------------------------------
            # File metadata
            # -----------------------------------------------------

            file_metadata = FileMetadata(
                version_id=version_id,
                original_filename=filename,
                stored_filename=stored_path.name,
                content_fingerprint=fingerprint,
                file_extension=Path(
                    filename
                ).suffix.lower(),
            )

            db.add(file_metadata)

            db.flush()

            registered_tables: list[str] = []

            # -----------------------------------------------------
            # Table metadata
            # -----------------------------------------------------

            for table in tables:

                table_metadata = TableMetadata(
                    version_id=version_id,
                    table_name=table["table_name"],
                    source_file=table["source_file"],
                    sheet_name=table["sheet_name"],
                    row_count=table["row_count"],
                )

                db.add(
                    table_metadata
                )

                db.flush()

                # -------------------------------------------------
                # Column metadata
                # -------------------------------------------------

                for column in table["columns"]:

                    column_metadata = ColumnMetadata(
                        table_id=table_metadata.table_id,
                        column_name=column[
                            "column_name"
                        ],
                        data_type=column[
                            "data_type"
                        ],
                        description=column.get(
                            "description"
                        ),
                    )

                    db.add(
                        column_metadata
                    )

                registered_tables.append(
                    table["table_name"]
                )

            registered_files.append(
                {
                    "file_id": file_metadata.file_id,
                    "filename": filename,
                    "stored_filename": stored_path.name,
                    "tables": registered_tables,
                    "stored_path": str(
                        stored_path
                    ),
                }
            )

        return registered_files

    except Exception:

        # ---------------------------------------------------------
        # Filesystem cleanup
        # ---------------------------------------------------------
        #
        # Database rollback is deliberately NOT performed here.
        # The caller owns the database transaction.
        #

        for stored_path in stored_paths:
            stored_path.unlink(
                missing_ok=True
            )

        raise


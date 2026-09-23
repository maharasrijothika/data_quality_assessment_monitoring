from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import (
    ColumnMetadata,
    Dataset,
    DatasetVersion,
    FileMetadata,
    TableMetadata,
)


router = APIRouter(
    prefix="/datasets",
    tags=["dataset-context"],
)


# ---------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------


class ColumnContextUpdate(BaseModel):
    column_id: int
    description: str | None = None


class TableContextUpdate(BaseModel):
    table_id: int
    description: str | None = None
    columns: list[ColumnContextUpdate] = Field(default_factory=list)


class DatasetContextUpdate(BaseModel):
    description: str | None = None
    domain: str | None = None
    source_system: str | None = None
    update_cadence: str | None = None
    tables: list[TableContextUpdate] = Field(default_factory=list)


# ---------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------


def get_latest_version(
    db: Session,
    dataset_id: int,
) -> DatasetVersion | None:
    return (
        db.query(DatasetVersion)
        .filter(
            DatasetVersion.dataset_id == dataset_id
        )
        .order_by(
            DatasetVersion.version_number.desc()
        )
        .first()
    )


# ---------------------------------------------------------------------
# GET context
# ---------------------------------------------------------------------


@router.get("/{dataset_id}/context")
def get_dataset_context(
    dataset_id: int,
    db: Session = Depends(get_db),
):
    dataset = (
        db.query(Dataset)
        .filter(
            Dataset.dataset_id == dataset_id
        )
        .first()
    )

    if dataset is None:
        raise HTTPException(
            status_code=404,
            detail=f"Dataset {dataset_id} not found.",
        )

    latest_version = get_latest_version(
        db,
        dataset_id,
    )

    if latest_version is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Dataset {dataset_id} has no registered version."
            ),
        )

    tables = (
        db.query(TableMetadata)
        .filter(
            TableMetadata.version_id
            == latest_version.version_id
        )
        .order_by(
            TableMetadata.table_id
        )
        .all()
    )
    files = (
        db.query(FileMetadata)
        .filter(
            FileMetadata.version_id
            == latest_version.version_id
        )
        .order_by(
            FileMetadata.file_id
        )
        .all()
    )


    table_response = []

    for table in tables:
        columns = (
            db.query(ColumnMetadata)
            .filter(
                ColumnMetadata.table_id
                == table.table_id
            )
            .order_by(
                ColumnMetadata.column_id
            )
            .all()
        )

        table_response.append(
            {
                "table_id": table.table_id,
                "table_name": table.table_name,
                "source_file": table.source_file,
                "sheet_name": table.sheet_name,
                "row_count": table.row_count,
                "description": table.description,
                "columns": [
                    {
                        "column_id": column.column_id,
                        "column_name": column.column_name,
                        "data_type": column.data_type,
                        "description": column.description,
                    }
                    for column in columns
                ],
            }
        )

    return {
        "dataset_id": dataset.dataset_id,
        "dataset_name": dataset.dataset_name,
        "description": dataset.description,
        "domain": dataset.domain,
        "source_system": dataset.source_system,
        "update_cadence": dataset.update_cadence,
         "current_version": {
            "version_id": latest_version.version_id,
            "version_number": latest_version.version_number,
            "parent_version_id": latest_version.parent_version_id,
            "schema_fingerprint": latest_version.schema_fingerprint,
            "content_fingerprint": latest_version.content_fingerprint,
            "files": [
                {
                    "file_id": file.file_id,
                    "filename": file.original_filename,
                    "stored_filename": file.stored_filename,
                    "content_fingerprint": file.content_fingerprint,
                }
                for file in files
            ],
        },
        "tables": table_response,
    }
# ---------------------------------------------------------------------
# GET registered datasets
# ---------------------------------------------------------------------


@router.get("")
def get_registered_datasets(
    db: Session = Depends(get_db),
):
    datasets = (
        db.query(Dataset)
        .order_by(Dataset.dataset_id.desc())
        .all()
    )

    response = []

    for dataset in datasets:
        latest_version = get_latest_version(
            db,
            dataset.dataset_id,
        )

        if latest_version is None:
            response.append(
                {
                    "dataset_id": dataset.dataset_id,
                    "dataset_name": dataset.dataset_name,
                    "source_system": dataset.source_system,
                    "domain": dataset.domain,
                    "update_cadence": dataset.update_cadence,
                    "description": dataset.description,
                    "version_number": None,
                    "file_count": 0,
                }
            )
            continue

        file_count = (
            db.query(FileMetadata)
            .filter(
                FileMetadata.version_id
                == latest_version.version_id
            )
            .count()
        )

        table_count = (
            db.query(TableMetadata)
            .filter(
                TableMetadata.version_id
                == latest_version.version_id
            )
            .count()
        )

        response.append(
            {
                "dataset_id": dataset.dataset_id,
                "dataset_name": dataset.dataset_name,
                "source_system": dataset.source_system,
                "domain": dataset.domain,
                "update_cadence": dataset.update_cadence,
                "description": dataset.description,
                "version_number": latest_version.version_number,
                "file_count": file_count,
                "table_count": table_count,
            }
        )

    return {
        "datasets": response,
        "count": len(response),
    }


# ---------------------------------------------------------------------
# UPDATE context
# ---------------------------------------------------------------------


@router.put("/{dataset_id}/context")
def update_dataset_context(
    dataset_id: int,
    context: DatasetContextUpdate,
    db: Session = Depends(get_db),
):
    dataset = (
        db.query(Dataset)
        .filter(
            Dataset.dataset_id == dataset_id
        )
        .first()
    )

    if dataset is None:
        raise HTTPException(
            status_code=404,
            detail=f"Dataset {dataset_id} not found.",
        )

    latest_version = get_latest_version(
        db,
        dataset_id,
    )

    if latest_version is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Dataset {dataset_id} has no registered version."
            ),
        )

    # -------------------------------------------------------------
    # Dataset-level context
    # -------------------------------------------------------------

    dataset.description = context.description
    dataset.domain = context.domain
    dataset.source_system = context.source_system
    dataset.update_cadence = context.update_cadence

    # -------------------------------------------------------------
    # Current-version table/column context
    # -------------------------------------------------------------

    valid_tables = {
        table.table_id: table
        for table in (
            db.query(TableMetadata)
            .filter(
                TableMetadata.version_id
                == latest_version.version_id
            )
            .all()
        )
    }

    for table_update in context.tables:

        table = valid_tables.get(
            table_update.table_id
        )

        if table is None:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Table {table_update.table_id} does not "
                    f"belong to the current dataset version."
                ),
            )

        table.description = table_update.description

        valid_columns = {
            column.column_id: column
            for column in (
                db.query(ColumnMetadata)
                .filter(
                    ColumnMetadata.table_id
                    == table.table_id
                )
                .all()
            )
        }

        for column_update in table_update.columns:

            column = valid_columns.get(
                column_update.column_id
            )

            if column is None:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Column {column_update.column_id} does not "
                        f"belong to table {table.table_id}."
                    ),
                )

            column.description = (
                column_update.description
            )

    db.commit()
    db.refresh(dataset)

    return {
        "message": "Dataset context updated successfully.",
        "dataset_id": dataset.dataset_id,
        "current_version": {
            "version_id": latest_version.version_id,
            "version_number": latest_version.version_number,
        },
    }
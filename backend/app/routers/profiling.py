from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Dataset, DatasetVersion, FileMetadata
from app.services.ingestion import read_table
from app.services.profiling import profile_dataframe
from app.services.storage import RAW_DATA_DIR


router = APIRouter(
    prefix="/datasets",
    tags=["profiling"],
)


@router.get("/{dataset_id}/profiling")
def get_dataset_profiling(
    dataset_id: int,
    db: Session = Depends(get_db),
):
    """
    Generate a profile for the latest immutable dataset version.

    Profiling is read-only. It does not modify the dataset,
    version, files, or metadata.
    """

    dataset = (
        db.query(Dataset)
        .filter(Dataset.dataset_id == dataset_id)
        .first()
    )

    if dataset is None:
        raise HTTPException(
            status_code=404,
            detail=f"Dataset {dataset_id} not found.",
        )

    version = (
        db.query(DatasetVersion)
        .filter(DatasetVersion.dataset_id == dataset_id)
        .order_by(DatasetVersion.version_number.desc())
        .first()
    )

    if version is None:
        raise HTTPException(
            status_code=404,
            detail=f"No version found for dataset {dataset_id}.",
        )

    files = (
        db.query(FileMetadata)
        .filter(FileMetadata.version_id == version.version_id)
        .order_by(FileMetadata.file_id)
        .all()
    )

    if not files:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No files found for dataset version "
                f"{version.version_id}."
            ),
        )

    tables = []

    version_dir = (
        RAW_DATA_DIR
        / str(dataset_id)
        / f"v{version.version_number}"
    )

    for file_metadata in files:
        file_path = version_dir / file_metadata.stored_filename

        if not file_path.exists():
            raise HTTPException(
                status_code=500,
                detail=(
                    "Immutable raw file is missing for "
                    f"file {file_metadata.file_id}."
                ),
            )

        try:
            if file_metadata.file_extension.lower() in {
                ".xls",
                ".xlsx",
            }:
                import pandas as pd

                workbook = pd.ExcelFile(file_path)

                for sheet_name in workbook.sheet_names:
                    dataframe = read_table(
                        file_path,
                        sheet_name=sheet_name,
                    )

                    profile = profile_dataframe(
                        dataframe,
                        table_name=str(sheet_name),
                    )

                    profile["source_file"] = (
                        file_metadata.original_filename
                    )
                    profile["sheet_name"] = str(sheet_name)

                    tables.append(profile)

            else:
                dataframe = read_table(file_path)

                profile = profile_dataframe(
                    dataframe,
                    table_name=file_path.stem,
                )

                profile["source_file"] = (
                    file_metadata.original_filename
                )
                profile["sheet_name"] = None

                tables.append(profile)

        except Exception as exc:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Unable to profile "
                    f"{file_metadata.original_filename}: {exc}"
                ),
            ) from exc

    total_rows = sum(
        table["row_count"]
        for table in tables
    )

    total_columns = sum(
        table["column_count"]
        for table in tables
    )

    return {
        "dataset_id": dataset.dataset_id,
        "dataset_name": dataset.dataset_name,
        "version_id": version.version_id,
        "version_number": version.version_number,
        "parent_version_id": version.parent_version_id,
        "table_count": len(tables),
        "total_rows": total_rows,
        "total_columns": total_columns,
        "tables": tables,
    }
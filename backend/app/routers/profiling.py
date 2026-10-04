from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Dataset, DatasetVersion, FileMetadata, StoredProfile
from app.services.ingestion import read_table
from app.services.profiling import profile_dataframe
from app.services.stage_state import get_latest_version, mark_stage_complete
from app.services import storage as storage_module


router = APIRouter(
    prefix="/datasets",
    tags=["profiling"],
)


def get_dataset_profiling_result(
    db: Session,
    dataset_id: int,
    version: DatasetVersion | None = None,
) -> dict:
    """Profile the latest (or given) immutable dataset version."""

    if version is None:
        version = (
            db.query(DatasetVersion)
            .filter(DatasetVersion.dataset_id == dataset_id)
            .order_by(DatasetVersion.version_number.desc())
            .first()
        )

    if version is None:
        raise ValueError(f"No version found for dataset {dataset_id}.")

    files = (
        db.query(FileMetadata)
        .filter(FileMetadata.version_id == version.version_id)
        .order_by(FileMetadata.file_id)
        .all()
    )

    if not files:
        raise ValueError(
            f"No files found for dataset version {version.version_id}."
        )

    tables = []

    version_dir = (
        storage_module.RAW_DATA_DIR
        / str(dataset_id)
        / f"v{version.version_number}"
    )

    for file_metadata in files:
        file_path = version_dir / file_metadata.stored_filename

        if not file_path.exists():
            raise ValueError(
                "Immutable raw file is missing for "
                f"file {file_metadata.file_id}."
            )

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

    total_rows = sum(
        table["row_count"]
        for table in tables
    )

    total_columns = sum(
        table["column_count"]
        for table in tables
    )

    return {
        "dataset_id": dataset_id,
        "version_id": version.version_id,
        "version_number": version.version_number,
        "parent_version_id": version.parent_version_id,
        "table_count": len(tables),
        "total_rows": total_rows,
        "total_columns": total_columns,
        "tables": tables,
    }


def persist_profile(
    db: Session,
    version_id: int,
    result: dict,
) -> StoredProfile:
    """Persist the profile result for a version (upsert)."""
    stored = (
        db.query(StoredProfile)
        .filter(StoredProfile.version_id == version_id)
        .first()
    )

    if stored is None:
        stored = StoredProfile(
            version_id=version_id,
            profile_json=result,
            row_count=result.get("total_rows", 0),
        )
        db.add(stored)
    else:
        stored.profile_json = result
        stored.row_count = result.get("total_rows", 0)

    db.flush()
    return stored


@router.get("/{dataset_id}/profiling")
def get_dataset_profiling(
    dataset_id: int,
    db: Session = Depends(get_db),
):
    """
    Generate (and persist) a profile for the latest immutable
    dataset version.

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

    version = get_latest_version(db, dataset_id)

    if version is None:
        raise HTTPException(
            status_code=404,
            detail=f"No version found for dataset {dataset_id}.",
        )

    try:
        result = get_dataset_profiling_result(db, dataset_id, version)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    # Persist the profile so later stages reuse the same computed evidence.
    persist_profile(db, version.version_id, result)
    mark_stage_complete(db, dataset_id, "profiling")
    db.commit()

    # Audit artifact: export the EXACT stored profiling result (no
    # recalculation) for reproducibility.
    try:
        from app.services import audit_artifacts

        audit_artifacts.export_profile_artifact(
            dataset_id,
            version.version_id,
            result,
        )
    except OSError:  # pragma: no cover - artifact export never breaks a run
        pass

    return result

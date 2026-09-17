
from pathlib import Path
import shutil
import tempfile

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Dataset
from app.services.dataset_fingerprinting import (
    calculate_dataset_content_fingerprint,
    calculate_schema_fingerprint,
)
from app.services.file_comparison import find_duplicate_files
from app.services.fingerprinting import calculate_file_fingerprint
from app.services.ingestion import discover_tables
from app.services.validation import validate_uploaded_file
from app.services.version_snapshot import register_version_snapshot
from app.services.versioning import create_dataset_version


router = APIRouter(
    prefix="/datasets",
    tags=["datasets"],
)


@router.post("/upload")
async def upload_dataset(
    files: list[UploadFile] = File(...),
    dataset_name: str = Form(...),
    source_system: str | None = Form(None),
    description: str | None = Form(None),
    db: Session = Depends(get_db),
):
    """
    Upload and register one or more data files as a new dataset.

    Processing flow:

        1. Validate uploaded files
        2. Store files temporarily
        3. Calculate file fingerprints
        4. Discover tables
        5. Detect duplicate files
        6. Calculate schema/content fingerprints
        7. Create Dataset
        8. Create DatasetVersion V1
        9. Register complete metadata snapshot
       10. Store immutable raw files under V1
       11. Commit transaction

    The first upload for a dataset always creates version 1.

    Raw files are stored under:

        data/raw/datasets/{dataset_id}/v{version_number}/

    The database transaction is rolled back if registration fails.
    Physical files created during a failed transaction are explicitly
    removed because the filesystem itself is not transactional.
    """

    if not files:
        raise HTTPException(
            status_code=400,
            detail="At least one file is required.",
        )

    temp_dir: Path | None = None


    try:
        # ---------------------------------------------------------
        # 1. Create temporary upload directory
        # ---------------------------------------------------------

        processed_dir = (
            Path(__file__).resolve().parents[3]
            / "data"
            / "processed"
        )

        processed_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        temp_dir = Path(
            tempfile.mkdtemp(
                prefix="dq_upload_",
                dir=processed_dir,
            )
        )

        prepared_files: list[dict] = []

        # ---------------------------------------------------------
        # 2. Validate and temporarily store uploaded files
        # ---------------------------------------------------------

        for uploaded_file in files:

            if not uploaded_file.filename:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Every uploaded file must have "
                        "a filename."
                    ),
                )

            content = await uploaded_file.read()

            try:
                validate_uploaded_file(
                    uploaded_file.filename,
                    len(content),
                )
            except ValueError as exc:
                raise HTTPException(
                    status_code=400,
                    detail=str(exc),
                ) from exc

            safe_filename = Path(
                uploaded_file.filename
            ).name

            temp_path = (
                temp_dir
                / safe_filename
            )

            temp_path.write_bytes(content)

            prepared_files.append(
                {
                    "filename": uploaded_file.filename,
                    "temp_path": temp_path,
                }
            )

        # ---------------------------------------------------------
        # 3. Fingerprint and discover tables
        # ---------------------------------------------------------

        for prepared_file in prepared_files:

            temp_path = prepared_file["temp_path"]

            fingerprint = calculate_file_fingerprint(
                temp_path
            )

            try:
                tables = discover_tables(
                    temp_path
                )
            except Exception as exc:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Could not read "
                        f"{prepared_file['filename']}: {exc}"
                    ),
                ) from exc

            if not tables:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"No readable tables found in "
                        f"{prepared_file['filename']}."
                    ),
                )

            prepared_file["fingerprint"] = fingerprint
            prepared_file["tables"] = tables

        # ---------------------------------------------------------
        # 4. Detect exact duplicate files
        # ---------------------------------------------------------

        duplicate_files: list[dict] = []

        for prepared_file in prepared_files:

            matches = find_duplicate_files(
                db,
                prepared_file["fingerprint"],
            )

            if matches:

                duplicate_files.append(
                    {
                        "filename": prepared_file[
                            "filename"
                        ],
                        "content_fingerprint": prepared_file[
                            "fingerprint"
                        ],
                        "existing_files": [
                            {
                                "file_id": match.file_id,
                                "original_filename": (
                                    match.original_filename
                                ),
                                "stored_filename": (
                                    match.stored_filename
                                ),
                                "version_id": (
                                    match.version_id
                                ),
                            }
                            for match in matches
                        ],
                    }
                )

        if duplicate_files:

            raise HTTPException(
                status_code=409,
                detail={
                    "message": (
                        "One or more uploaded files already "
                        "exist in the registered file history."
                    ),
                    "duplicate_files": duplicate_files,
                },
            )

        # ---------------------------------------------------------
        # 5. Build dataset-level fingerprints
        # ---------------------------------------------------------

        dataset_tables: list[dict] = []

        file_fingerprints: list[
            tuple[str, str]
        ] = []

        for prepared_file in prepared_files:

            filename = prepared_file[
                "filename"
            ]

            fingerprint = prepared_file[
                "fingerprint"
            ]

            tables = prepared_file[
                "tables"
            ]

            file_fingerprints.append(
                (
                    filename,
                    fingerprint,
                )
            )

            dataset_tables.extend(
                tables
            )

        schema_fingerprint = (
            calculate_schema_fingerprint(
                dataset_tables
            )
        )

        content_fingerprint = (
            calculate_dataset_content_fingerprint(
                file_fingerprints
            )
        )

        # ---------------------------------------------------------
        # 6. Create dataset
        # ---------------------------------------------------------

        dataset = Dataset(
            dataset_name=dataset_name,
            source_system=source_system,
            description=description,
        )

        db.add(dataset)

        db.flush()

        # ---------------------------------------------------------
        # 7. Create initial dataset version
        # ---------------------------------------------------------

        version = create_dataset_version(
            db=db,
            dataset_id=dataset.dataset_id,
            parent_version_id=None,
            schema_fingerprint=schema_fingerprint,
            content_fingerprint=content_fingerprint,
        )

        # ---------------------------------------------------------
        # 8. Prepare snapshot input
        # ---------------------------------------------------------

        snapshot_files: list[dict] = []

        for prepared_file in prepared_files:

            snapshot_files.append(
                {
                    "filename": prepared_file[
                        "filename"
                    ],
                    "source_path": prepared_file[
                        "temp_path"
                    ],
                    "fingerprint": prepared_file[
                        "fingerprint"
                    ],
                    "tables": prepared_file[
                        "tables"
                    ],
                }
            )

        # ---------------------------------------------------------
        # 9. Register metadata + store immutable raw files
        # ---------------------------------------------------------

        registered_files = (
            register_version_snapshot(
                db=db,
                dataset_id=dataset.dataset_id,
                version_id=version.version_id,
                version_number=version.version_number,
                files=snapshot_files,
            )
        )

        # ---------------------------------------------------------
        # 10. Commit database transaction
        # ---------------------------------------------------------

        db.commit()

        # ---------------------------------------------------------
        # 11. Return response
        # ---------------------------------------------------------

        response_files = []

        for registered_file in registered_files:

            prepared_file = next(
                item
                for item in prepared_files
                if item["filename"]
                == registered_file["filename"]
            )

            response_files.append(
                {
                    "filename": registered_file[
                        "filename"
                    ],
                    "stored_filename": registered_file[
                        "stored_filename"
                    ],
                    "content_fingerprint": prepared_file[
                        "fingerprint"
                    ],
                    "tables": registered_file[
                        "tables"
                    ],
                }
            )

            # Track the permanent file so it can be
            # removed if an unexpected error occurs
            # after snapshot registration but before
            # the transaction is committed.
            

        return {
            "message": (
                "Dataset uploaded successfully."
            ),
            "dataset_id": dataset.dataset_id,
            "dataset_name": dataset.dataset_name,
            "version_id": version.version_id,
            "version_number": version.version_number,
            "schema_fingerprint": schema_fingerprint,
            "content_fingerprint": content_fingerprint,
            "files": response_files,
        }

    except HTTPException:
        # ---------------------------------------------------------
        # Roll back database changes
        # ---------------------------------------------------------

        db.rollback()

        # ---------------------------------------------------------
        # Remove permanent files created before failure
        # ---------------------------------------------------------

        

        raise

    except Exception as exc:
        db.rollback()

        raise HTTPException(
           status_code=500,
           detail=(
              f"Dataset upload failed: {exc}"
           ),
        ) from exc

    finally:
        # ---------------------------------------------------------
        # Always remove temporary upload directory
        # ---------------------------------------------------------

        if temp_dir is not None:
            shutil.rmtree(
                temp_dir,
                ignore_errors=True,
            )


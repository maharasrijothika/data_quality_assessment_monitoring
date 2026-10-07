from sqlalchemy.orm import Session

from app.models import Dataset
from app.models.dataset_version import DatasetVersion
from app.models.file_metadata import FileMetadata


def find_duplicate_files(
    db: Session,
    content_fingerprint: str,
) -> list[FileMetadata]:
    """
    Find files with the same content fingerprint that belong
    to an active dataset.

    Files belonging to logically deleted datasets are retained
    for historical lineage but do not block re-registration.
    """
    return (
        db.query(FileMetadata)
        .join(
            DatasetVersion,
            DatasetVersion.version_id == FileMetadata.version_id,
        )
        .join(
            Dataset,
            Dataset.dataset_id == DatasetVersion.dataset_id,
        )
        .filter(
            FileMetadata.content_fingerprint == content_fingerprint,
            Dataset.deleted_at.is_(None),
        )
        .all()
    )
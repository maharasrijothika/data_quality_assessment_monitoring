from sqlalchemy.orm import Session

from app.models.file_metadata import FileMetadata


def find_duplicate_files(
    db: Session,
    content_fingerprint: str,
) -> list[FileMetadata]:
    """
    Find all previously registered files with the same
    content fingerprint.

    An exact content fingerprint match means the incoming
    file is an exact duplicate of an already registered file.
    """
    return (
        db.query(FileMetadata)
        .filter(
            FileMetadata.content_fingerprint
            == content_fingerprint
        )
        .all()
    )
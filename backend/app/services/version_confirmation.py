from sqlalchemy.orm import Session

from app.services.version_snapshot import register_version_snapshot
from app.services.versioning import create_dataset_version


def confirm_dataset_version(
    db: Session,
    dataset_id: int,
    parent_version_id: int,
    schema_fingerprint: str,
    content_fingerprint: str,
    files: list[dict],
):
    """
    Create a new dataset version and register its complete
    immutable snapshot.

    The caller controls the transaction.

    Version creation and snapshot registration are intentionally
    performed within the same database transaction.

    Physical raw files are stored under:

        data/raw/datasets/{dataset_id}/v{version_number}/
    """

    version = create_dataset_version(
        db=db,
        dataset_id=dataset_id,
        parent_version_id=parent_version_id,
        schema_fingerprint=schema_fingerprint,
        content_fingerprint=content_fingerprint,
    )

    register_version_snapshot(
        db=db,
        dataset_id=dataset_id,
        version_id=version.version_id,
        version_number=version.version_number,
        files=files,
    )

    return version
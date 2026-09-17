from sqlalchemy.orm import Session

from app.models import Dataset, DatasetVersion


def get_next_version_number(
    db: Session,
    dataset_id: int,
) -> int:
    """
    Return the next version number for a dataset.
    """

    latest_version = (
        db.query(DatasetVersion)
        .filter(
            DatasetVersion.dataset_id == dataset_id
        )
        .order_by(
            DatasetVersion.version_number.desc()
        )
        .first()
    )

    if latest_version is None:
        return 1

    return latest_version.version_number + 1


def create_dataset_version(
    db: Session,
    dataset_id: int,
    parent_version_id: int | None,
    schema_fingerprint: str,
    content_fingerprint: str,
) -> DatasetVersion:
    """
    Create a new immutable dataset version.

    The existing version is never modified.
    """

    dataset = (
        db.query(Dataset)
        .filter(
            Dataset.dataset_id == dataset_id
        )
        .first()
    )

    if dataset is None:
        raise ValueError(
            f"Dataset {dataset_id} does not exist."
        )

    if parent_version_id is not None:

        parent_version = (
            db.query(DatasetVersion)
            .filter(
                DatasetVersion.version_id
                == parent_version_id
            )
            .first()
        )

        if parent_version is None:
            raise ValueError(
                f"Parent version "
                f"{parent_version_id} does not exist."
            )

        if parent_version.dataset_id != dataset_id:
            raise ValueError(
                "Parent version belongs to a different dataset."
            )

    version_number = get_next_version_number(
        db,
        dataset_id,
    )

    version = DatasetVersion(
        dataset_id=dataset_id,
        parent_version_id=parent_version_id,
        version_number=version_number,
        schema_fingerprint=schema_fingerprint,
        content_fingerprint=content_fingerprint,
    )

    db.add(version)
    db.flush()

    return version
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from app.models import Dataset


def logically_delete_dataset(
    db: Session,
    dataset_id: int,
) -> Dataset:
    """
    Mark a dataset as logically deleted.

    The dataset, its versions, lineage, profiles, and immutable
    raw files are preserved. The dataset ID is never reused.
    """

    dataset = (
        db.query(Dataset)
        .filter(
            Dataset.dataset_id == dataset_id,
            Dataset.deleted_at.is_(None),
        )
        .first()
    )

    if dataset is None:
        raise ValueError(
            f"Active dataset {dataset_id} does not exist."
        )

    dataset.deleted_at = datetime.now(timezone.utc)

    db.flush()

    return dataset
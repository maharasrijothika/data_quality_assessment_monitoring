from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.services.dataset_deletion import logically_delete_dataset


router = APIRouter(
    prefix="/datasets",
    tags=["datasets"],
)


@router.delete("/{dataset_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_dataset(
    dataset_id: int,
    db: Session = Depends(get_db),
) -> None:
    """
    Logically delete a dataset.

    Dataset versions, lineage, profiles, and immutable raw files
    are preserved.
    """

    try:
        logically_delete_dataset(
            db=db,
            dataset_id=dataset_id,
        )

        db.commit()

    except ValueError as exc:
        db.rollback()

        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc

    except Exception:
        db.rollback()
        raise
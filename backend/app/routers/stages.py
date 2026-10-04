from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.services.stage_state import (
    get_dataset_or_none,
    get_stage_progress,
)

router = APIRouter(prefix="/datasets", tags=["stages"])


@router.get("/{dataset_id}/stages")
def get_dataset_stage_progress(
    dataset_id: int,
    db: Session = Depends(get_db),
):
    dataset = get_dataset_or_none(db, dataset_id)

    if dataset is None:
        raise HTTPException(
            status_code=404,
            detail=f"Dataset {dataset_id} not found.",
        )

    progress = get_stage_progress(db, dataset_id)

    return {
        "dataset_id": dataset.dataset_id,
        "dataset_name": dataset.dataset_name,
        **progress,
    }

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import DatasetVersion
from app.services.stage_state import (
    get_dataset_or_none,
    get_latest_version,
    mark_stage_complete,
)
from app.services.monitoring import run_monitoring

router = APIRouter(prefix="/datasets", tags=["monitoring"])


class MonitoringRequest(BaseModel):
    baseline_version_id: int | None = None


@router.post("/{dataset_id}/monitoring")
def run_monitoring_endpoint(
    dataset_id: int,
    request: MonitoringRequest,
    db: Session = Depends(get_db),
):
    """Run drift monitoring comparing latest version against a baseline."""
    dataset = get_dataset_or_none(db, dataset_id)

    if dataset is None:
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found.")

    comparison = get_latest_version(db, dataset_id)

    if comparison is None:
        raise HTTPException(
            status_code=404,
            detail=f"No version found for dataset {dataset_id}.",
        )

    baseline_version_id = request.baseline_version_id

    if baseline_version_id is None:
        # Default baseline: the previous version before the latest.
        previous = (
            db.query(DatasetVersion)
            .filter(
                DatasetVersion.dataset_id == dataset_id,
                DatasetVersion.version_number < comparison.version_number,
            )
            .order_by(DatasetVersion.version_number.desc())
            .first()
        )

        if previous is None:
            raise HTTPException(
                status_code=400,
                detail=(
                    "No baseline version available. Upload a new version or "
                    "specify baseline_version_id."
                ),
            )

        baseline_version_id = previous.version_id

    baseline = db.get(DatasetVersion, baseline_version_id)

    if baseline is None or baseline.dataset_id != dataset_id:
        raise HTTPException(
            status_code=400,
            detail="Baseline version does not belong to this dataset.",
        )

    if baseline_version_id == comparison.version_id:
        raise HTTPException(
            status_code=400,
            detail="Baseline and comparison versions must differ.",
        )

    try:
        result = run_monitoring(
            db=db,
            dataset_id=dataset_id,
            baseline_version_id=baseline_version_id,
            comparison_version_id=comparison.version_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    mark_stage_complete(db, dataset_id, "monitoring")
    db.commit()

    return result

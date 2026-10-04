"""Canonical DQ pipeline stages and backend-persisted stage progress."""

from sqlalchemy.orm import Session

from app.models import Dataset, DatasetVersion, StageState

PIPELINE_STAGES = [
    "ingestion",
    "context",
    "version",
    "profiling",
    "semantic",
    "relationships",
    "recommendations",
    "validation",
    "execution",
    "scoring",
    "rca",
    "remediation",
    "monitoring",
    "feedback",
]

STAGE_LABELS = {
    "ingestion": "Ingestion",
    "context": "Dataset Context",
    "version": "Version & Fingerprint",
    "profiling": "Data Profiling",
    "semantic": "Semantic Understanding",
    "relationships": "Relationship Discovery",
    "recommendations": "Metric & Rule Recommendation",
    "validation": "Rule Validation",
    "execution": "Human Approval & Rule Execution",
    "scoring": "DQ Scoring",
    "rca": "Root Cause Analysis",
    "remediation": "Remediation & Reassessment",
    "monitoring": "Monitoring",
    "feedback": "Feedback & Offline Learning",
}


def get_dataset_or_none(db: Session, dataset_id: int) -> Dataset | None:
    return (
        db.query(Dataset)
        .filter(Dataset.dataset_id == dataset_id)
        .first()
    )


def get_latest_version(
    db: Session,
    dataset_id: int,
) -> DatasetVersion | None:
    return (
        db.query(DatasetVersion)
        .filter(DatasetVersion.dataset_id == dataset_id)
        .order_by(DatasetVersion.version_number.desc())
        .first()
    )


def get_version(
    db: Session,
    dataset_id: int,
    version_id: int,
) -> DatasetVersion | None:
    return (
        db.query(DatasetVersion)
        .filter(
            DatasetVersion.dataset_id == dataset_id,
            DatasetVersion.version_id == version_id,
        )
        .first()
    )


def mark_stage_complete(
    db: Session,
    dataset_id: int,
    stage_key: str,
) -> None:
    """Persist stage completion. Idempotent."""
    if stage_key not in PIPELINE_STAGES:
        raise ValueError(f"Unknown stage: {stage_key}")

    existing = (
        db.query(StageState)
        .filter(
            StageState.dataset_id == dataset_id,
            StageState.stage_key == stage_key,
        )
        .first()
    )

    if existing is None:
        db.add(
            StageState(
                dataset_id=dataset_id,
                stage_key=stage_key,
            )
        )
        db.flush()


def is_stage_complete(
    db: Session,
    dataset_id: int,
    stage_key: str,
) -> bool:
    return (
        db.query(StageState)
        .filter(
            StageState.dataset_id == dataset_id,
            StageState.stage_key == stage_key,
        )
        .first()
        is not None
    )


def get_stage_progress(db: Session, dataset_id: int) -> dict:
    completed = {
        stage.stage_key
        for stage in db.query(StageState)
        .filter(StageState.dataset_id == dataset_id)
        .all()
    }

    ordered = [s for s in PIPELINE_STAGES]

    first_incomplete = None
    for stage in ordered:
        if stage not in completed:
            first_incomplete = stage
            break

    return {
        "stages": [
            {
                "stage_key": stage,
                "label": STAGE_LABELS[stage],
                "completed": stage in completed,
            }
            for stage in ordered
        ],
        "first_incomplete_stage": first_incomplete,
        "completed_count": len(completed & set(ordered)),
        "total_stages": len(ordered),
    }

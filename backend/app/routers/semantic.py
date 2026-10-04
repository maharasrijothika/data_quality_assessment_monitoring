from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Dataset
from app.services.stage_state import (
    get_dataset_or_none,
    get_latest_version,
    mark_stage_complete,
)
from app.models import SemanticRun
from app.services.semantic_stage import (
    apply_semantic_decision,
    get_or_create_profile,
    get_semantic_predictions,
    run_semantic_analysis,
    semantic_feedback_summary,
)

router = APIRouter(prefix="/datasets", tags=["semantic"])


class SemanticDecisionRequest(BaseModel):
    decision: str
    concept_id: int | None = None
    user_defined_type: str | None = None


def _get_dataset_or_404(db: Session, dataset_id: int) -> Dataset:
    dataset = get_dataset_or_none(db, dataset_id)
    if dataset is None:
        raise HTTPException(
            status_code=404,
            detail=f"Dataset {dataset_id} not found.",
        )
    return dataset


@router.post("/{dataset_id}/semantic/analyze")
def analyze_semantics(
    dataset_id: int,
    db: Session = Depends(get_db),
):
    """Run semantic understanding for the latest version and persist predictions."""
    dataset = _get_dataset_or_404(db, dataset_id)

    version = get_latest_version(db, dataset_id)
    if version is None:
        raise HTTPException(
            status_code=404,
            detail=f"No version found for dataset {dataset_id}.",
        )

    try:
        get_or_create_profile(db, dataset_id, version.version_id)
        result = run_semantic_analysis(db, dataset, version.version_id)
    except ValueError as exc:
        db.rollback()
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    mark_stage_complete(db, dataset_id, "semantic")
    db.commit()

    return result


@router.get("/{dataset_id}/semantic")
def get_semantic(
    dataset_id: int,
    db: Session = Depends(get_db),
):
    """Return persisted semantic predictions for the latest version."""
    dataset = _get_dataset_or_404(db, dataset_id)

    version = get_latest_version(db, dataset_id)
    if version is None:
        raise HTTPException(
            status_code=404,
            detail=f"No version found for dataset {dataset_id}.",
        )

    predictions = get_semantic_predictions(db, version.version_id)

    return {
        "dataset_id": dataset.dataset_id,
        "version_id": version.version_id,
        "column_count": len(predictions),
        "columns": [
            {
                "prediction_id": p.prediction_id,
                "table_name": p.table_name,
                "column_name": p.column_name,
                "predicted_concept": p.predicted_concept,
                "concept_id": p.user_confirmed_concept_id or p.predicted_concept_id,
                "confidence_score": p.confidence_score,
                "confidence_level": p.confidence_level,
                "evidence": p.evidence_json.get("features", p.evidence_json),
                "evidence_coverage": p.evidence_json.get("evidence_coverage", 0.0),
                "source": p.evidence_json.get("source", "KB"),
                "match_status": p.evidence_json.get("match_status", "MATCHED" if p.predicted_concept else "UNKNOWN"),
                "normalization": p.evidence_json.get("normalization"),
                "run_number": p.evidence_json.get("run_number"),
                "semantic_input": p.evidence_json.get("semantic_input"),
                # v2 additive fields (absent in v1 payloads -> .get defaults)
                "decision_kind": p.evidence_json.get("decision_kind"),
                "semantic_family": p.evidence_json.get("semantic_family"),
                "proposal": p.evidence_json.get("proposal"),
                "candidates": p.evidence_json.get("candidates", []),
                "decision_rule": p.evidence_json.get("decision_rule"),
                "margin": p.evidence_json.get("margin"),
                "alternatives": p.alternatives_json.get("alternatives", []),
                "status": p.status,
                "final_semantic_type": (
                    p.user_defined_type
                    if p.user_defined_type
                    else (p.predicted_concept if p.user_confirmed_concept_id else None)
                ),
                "user_defined_type": p.user_defined_type,
                "decision_source": p.decision_source,
                "decided_at": p.decided_at.isoformat() if p.decided_at else None,
            }
            for p in predictions
        ],
    }


@router.get("/{dataset_id}/semantic/runs")
def list_semantic_runs(
    dataset_id: int,
    db: Session = Depends(get_db),
):
    """List every semantic run for the latest version (append-only history)."""
    dataset = _get_dataset_or_404(db, dataset_id)

    version = get_latest_version(db, dataset_id)
    if version is None:
        raise HTTPException(
            status_code=404,
            detail=f"No version found for dataset {dataset_id}.",
        )

    runs = (
        db.query(SemanticRun)
        .filter(SemanticRun.version_id == version.version_id)
        .order_by(SemanticRun.run_number.asc())
        .all()
    )

    return {
        "dataset_id": dataset.dataset_id,
        "version_id": version.version_id,
        "runs": [
            {
                "run_id": run.run_id,
                "run_number": run.run_number,
                "created_at": run.created_at.isoformat(),
                "model_version": run.model_version,
                "embedding_representation_version": (
                    run.embedding_representation_version
                ),
                "evidence_version": run.evidence_version,
                "kb_fingerprint": run.kb_fingerprint,
                "configuration": run.configuration_json,
                "column_count": len(
                    (run.predictions_json or {}).get("columns", [])
                ),
            }
            for run in runs
        ],
    }


@router.get("/{dataset_id}/semantic/runs/compare")
def compare_semantic_runs(
    dataset_id: int,
    db: Session = Depends(get_db),
):
    """Compare the two most recent semantic runs (S(n-1) vs S(n)).

    Comparison is computed from the append-only run records; nothing is
    overwritten, so runs of different semantic logic stay comparable.
    """
    dataset = _get_dataset_or_404(db, dataset_id)

    version = get_latest_version(db, dataset_id)
    if version is None:
        raise HTTPException(
            status_code=404,
            detail=f"No version found for dataset {dataset_id}.",
        )

    runs = (
        db.query(SemanticRun)
        .filter(SemanticRun.version_id == version.version_id)
        .order_by(SemanticRun.run_number.desc())
        .limit(2)
        .all()
    )
    runs.reverse()

    if len(runs) < 2:
        return {
            "dataset_id": dataset.dataset_id,
            "comparison_available": False,
            "reason": "Fewer than two semantic runs recorded.",
            "runs": [
                {"run_number": run.run_number, "created_at": run.created_at.isoformat()}
                for run in runs
            ],
        }

    old_run, new_run = runs

    def _predictions(run: SemanticRun) -> dict[tuple[str, str], dict]:
        return {
            (column["table_name"], column["column_name"]): column
            for column in (run.predictions_json or {}).get("columns", [])
        }

    old_columns = _predictions(old_run)
    new_columns = _predictions(new_run)

    entries = []
    for key in sorted(set(old_columns) | set(new_columns)):
        old_column = old_columns.get(key)
        new_column = new_columns.get(key)
        table_name, column_name = key
        entries.append(
            {
                "table_name": table_name,
                "column_name": column_name,
                "old_prediction": (
                    old_column.get("predicted_concept") if old_column else None
                ),
                "old_confidence_level": (
                    old_column.get("confidence_level") if old_column else None
                ),
                "new_prediction": (
                    new_column.get("predicted_concept") if new_column else None
                ),
                "new_confidence_level": (
                    new_column.get("confidence_level") if new_column else None
                ),
                "changed": (
                    (old_column or {}).get("predicted_concept")
                    != (new_column or {}).get("predicted_concept")
                ),
            }
        )

    return {
        "dataset_id": dataset.dataset_id,
        "comparison_available": True,
        "old_run": {
            "run_number": old_run.run_number,
            "created_at": old_run.created_at.isoformat(),
            "model_version": old_run.model_version,
            "kb_fingerprint": old_run.kb_fingerprint,
        },
        "new_run": {
            "run_number": new_run.run_number,
            "created_at": new_run.created_at.isoformat(),
            "model_version": new_run.model_version,
            "kb_fingerprint": new_run.kb_fingerprint,
        },
        "columns": entries,
    }


@router.post("/{dataset_id}/semantic/{prediction_id}/decision")
def decide_semantic(
    dataset_id: int,
    prediction_id: int,
    request: SemanticDecisionRequest,
    db: Session = Depends(get_db),
):
    """Human decision: approve / reject / edit a semantic prediction."""
    dataset = _get_dataset_or_404(db, dataset_id)

    version = get_latest_version(db, dataset_id)
    if version is None:
        raise HTTPException(
            status_code=404,
            detail=f"No version found for dataset {dataset_id}.",
        )

    try:
        prediction = apply_semantic_decision(
            db=db,
            dataset=dataset,
            version_id=version.version_id,
            prediction_id=prediction_id,
            decision=request.decision,
            concept_id=request.concept_id,
            user_defined_type=request.user_defined_type,
        )
    except ValueError as exc:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    db.commit()

    return {
        "message": f"Semantic prediction {request.decision}.",
        "prediction_id": prediction.prediction_id,
        "status": prediction.status,
        "concept_id": prediction.user_confirmed_concept_id,
        "predicted_concept": prediction.predicted_concept,
        "final_semantic_type": prediction.user_defined_type or prediction.predicted_concept,
        "decision_source": prediction.decision_source,
    }


@router.get("/{dataset_id}/semantic/feedback")
def get_semantic_feedback(
    dataset_id: int,
    db: Session = Depends(get_db),
):
    dataset = _get_dataset_or_404(db, dataset_id)
    return semantic_feedback_summary(db, dataset_id)

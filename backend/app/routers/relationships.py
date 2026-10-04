"""Relationship discovery stage APIs.

Discovery produces RANKED CANDIDATES with evidence. Only HUMAN-APPROVED
candidates may later become authoritative referential integrity rules.
This router never executes RI checks itself.
"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Dataset
from app.services.stage_state import (
    get_dataset_or_none,
    get_latest_version,
    mark_stage_complete,
)
from app.services.relationship_discovery import (
    _candidate_response,
    add_manual_key_candidate,
    add_manual_relationship,
    apply_relationship_decision,
    discover_relationships,
    edit_relationship_candidate,
    list_relationship_candidates,
)

router = APIRouter(prefix="/datasets", tags=["relationships"])


class RelationshipDecisionRequest(BaseModel):
    decision: str
    note: str | None = None


class ManualRelationshipRequest(BaseModel):
    parent_table: str
    parent_column: str
    child_table: str
    child_column: str
    note: str | None = None


class ManualKeyCandidateRequest(BaseModel):
    """User-declared single-column or composite primary-key candidate."""

    table_name: str
    key_columns: list[str]
    note: str | None = None


class EditRelationshipRequest(BaseModel):
    """User correction of a recommended relationship."""

    parent_table: str
    parent_column: str
    child_table: str
    child_column: str
    note: str | None = None


def _dataset_or_404(db: Session, dataset_id: int) -> Dataset:
    dataset = get_dataset_or_none(db, dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found.")
    return dataset


def _version_or_404(db: Session, dataset_id: int):
    version = get_latest_version(db, dataset_id)
    if version is None:
        raise HTTPException(
            status_code=404,
            detail=f"No version found for dataset {dataset_id}.",
        )
    return version


@router.post("/{dataset_id}/relationships/discover")
def run_relationship_discovery(
    dataset_id: int,
    db: Session = Depends(get_db),
):
    """Discover PK/FK relationship candidates for the latest version."""
    dataset = _dataset_or_404(db, dataset_id)
    version = _version_or_404(db, dataset_id)

    try:
        result = discover_relationships(db, dataset, version.version_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    db.commit()

    # Audit artifact: the EXACT discovery result (all evidence signals and
    # orphan statistics) exported without recalculation.
    try:
        from app.services import audit_artifacts

        audit_artifacts.export_relationships_artifact(
            dataset_id,
            version.version_id,
            result,
        )
    except OSError:  # pragma: no cover - artifact export never breaks a run
        pass

    return result


@router.get("/{dataset_id}/relationships")
def list_relationships(
    dataset_id: int,
    db: Session = Depends(get_db),
):
    dataset = _dataset_or_404(db, dataset_id)
    version = _version_or_404(db, dataset_id)

    candidates = list_relationship_candidates(db, version.version_id)

    return {
        "dataset_id": dataset_id,
        "version_id": version.version_id,
        "count": len(candidates),
        "candidates": [_candidate_response(candidate) for candidate in candidates],
    }


@router.post("/{dataset_id}/relationships/{relationship_id}/decision")
def decide_relationship(
    dataset_id: int,
    relationship_id: int,
    request: RelationshipDecisionRequest,
    db: Session = Depends(get_db),
):
    """Human decision: approve / reject / mark missed."""
    dataset = _dataset_or_404(db, dataset_id)
    version = _version_or_404(db, dataset_id)

    try:
        candidate = apply_relationship_decision(
            db,
            dataset,
            version.version_id,
            relationship_id,
            request.decision,
            request.note,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    db.commit()

    return {
        "message": f"Relationship {request.decision}.",
        "relationship_id": candidate.relationship_id,
        "status": candidate.status,
    }


@router.post("/{dataset_id}/relationships/{relationship_id}/edit")
def edit_relationship(
    dataset_id: int,
    relationship_id: int,
    request: EditRelationshipRequest,
    db: Session = Depends(get_db),
):
    """Correct a recommended relationship; the correction becomes the user's
    final decision and the original recommendation stays in history."""
    dataset = _dataset_or_404(db, dataset_id)
    version = _version_or_404(db, dataset_id)

    try:
        candidate = edit_relationship_candidate(
            db,
            dataset,
            version.version_id,
            relationship_id,
            request.parent_table,
            request.parent_column,
            request.child_table,
            request.child_column,
            request.note,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    db.commit()

    return {
        "message": "Relationship corrected by user.",
        "relationship_id": candidate.relationship_id,
        "status": candidate.status,
        "candidate": _candidate_response(candidate),
    }


@router.post("/{dataset_id}/relationships/manual-key")
def add_manual_key(
    dataset_id: int,
    request: ManualKeyCandidateRequest,
    db: Session = Depends(get_db),
):
    """Declare a PK / composite-PK candidate manually (single-file datasets
    included). N entries are allowed; existing entries are kept."""
    dataset = _dataset_or_404(db, dataset_id)
    version = _version_or_404(db, dataset_id)

    try:
        candidate = add_manual_key_candidate(
            db,
            dataset,
            version.version_id,
            request.table_name,
            request.key_columns,
            request.note,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    db.commit()

    return {
        "message": f"Key candidate for {request.table_name} added manually.",
        "relationship_id": candidate.relationship_id,
        "status": candidate.status,
    }


@router.post("/{dataset_id}/relationships/manual")
def add_relationship_manually(
    dataset_id: int,
    request: ManualRelationshipRequest,
    db: Session = Depends(get_db),
):
    """Add a human-specified relationship."""
    dataset = _dataset_or_404(db, dataset_id)
    version = _version_or_404(db, dataset_id)

    candidate = add_manual_relationship(
        db,
        dataset,
        version.version_id,
        request.parent_table.strip(),
        request.parent_column.strip(),
        request.child_table.strip(),
        request.child_column.strip(),
        request.note,
    )

    db.commit()

    return {
        "message": "Manual relationship added.",
        "relationship_id": candidate.relationship_id,
        "status": candidate.status,
    }


@router.post("/{dataset_id}/relationships/complete")
def complete_relationship_stage(
    dataset_id: int,
    db: Session = Depends(get_db),
):
    """Mark the relationship stage as reviewed."""
    _dataset_or_404(db, dataset_id)

    mark_stage_complete(db, dataset_id, "relationships")
    db.commit()

    return {"message": "Relationship discovery stage completed."}

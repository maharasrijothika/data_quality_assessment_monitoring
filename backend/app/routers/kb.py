from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import ConceptAlias, KnowledgeBaseVersion, SemanticConcept, SemanticFeedback
from app.services.semantic_kb import seed_semantic_concepts

router = APIRouter(prefix="/kb", tags=["knowledge-base"])


class AliasProposalRequest(BaseModel):
    concept_id: int
    alias: str
    evidence_count: int = 1


class AliasAdmissionRequest(BaseModel):
    approve: bool
    evidence_count: int = 1


@router.get("/concepts")
def list_concepts(db: Session = Depends(get_db)):
    seed_semantic_concepts(db)

    concepts = db.query(SemanticConcept).order_by(SemanticConcept.concept_name).all()

    return {
        "count": len(concepts),
        "concepts": [
            {
                "concept_id": c.concept_id,
                "concept_name": c.concept_name,
                "category": c.category,
                "description": c.description,
                "aliases": c.aliases,
                "expected_data_types": c.expected_data_types,
            }
            for c in concepts
        ],
    }


@router.post("/aliases/propose")
def propose_alias(request: AliasProposalRequest, db: Session = Depends(get_db)):
    """Propose a new alias for KB admission. Requires human approval."""
    concept = (
        db.query(SemanticConcept)
        .filter(SemanticConcept.concept_id == request.concept_id)
        .first()
    )

    if concept is None:
        raise HTTPException(status_code=404, detail="Concept not found.")

    existing = (
        db.query(KnowledgeBaseVersion)
        .filter(
            KnowledgeBaseVersion.concept_id == request.concept_id,
            KnowledgeBaseVersion.alias_text == request.alias.strip().lower(),
            KnowledgeBaseVersion.status == "pending",
        )
        .first()
    )

    if existing is not None:
        existing.evidence_count += request.evidence_count
        db.commit()
        return {
            "message": "Alias proposal evidence increased.",
            "kb_entry_id": existing.kb_entry_id,
            "evidence_count": existing.evidence_count,
        }

    entry = KnowledgeBaseVersion(
        concept_id=request.concept_id,
        kb_version=1,
        action="add_alias",
        alias_text=request.alias.strip().lower(),
        evidence_count=request.evidence_count,
        status="pending",
    )

    db.add(entry)
    db.commit()

    return {
        "message": "Alias proposed for admission.",
        "kb_entry_id": entry.kb_entry_id,
        "evidence_count": entry.evidence_count,
    }


@router.post("/aliases/{kb_entry_id}/admission")
def decide_alias_admission(
    kb_entry_id: int,
    request: AliasAdmissionRequest,
    db: Session = Depends(get_db),
):
    """Human admission decision for a proposed alias."""
    entry = (
        db.query(KnowledgeBaseVersion)
        .filter(KnowledgeBaseVersion.kb_entry_id == kb_entry_id)
        .first()
    )

    if entry is None:
        raise HTTPException(status_code=404, detail="KB entry not found.")

    if entry.status != "pending":
        raise HTTPException(status_code=400, detail="KB entry already decided.")

    if request.approve:
        entry.status = "approved"
        entry.evidence_count = max(entry.evidence_count, request.evidence_count)

        db.add(
            ConceptAlias(
                concept_id=entry.concept_id,
                alias_text=entry.alias_text,
                evidence_count=entry.evidence_count,
                source="admitted",
            )
        )

        concept = (
            db.query(SemanticConcept)
            .filter(SemanticConcept.concept_id == entry.concept_id)
            .first()
        )

        if concept is not None:
            import json

            try:
                aliases = json.loads(concept.aliases) if concept.aliases else []
            except json.JSONDecodeError:
                aliases = []

            if entry.alias_text not in aliases:
                aliases.append(entry.alias_text)
                concept.aliases = json.dumps(aliases)
    else:
        entry.status = "rejected"

    db.commit()

    return {
        "message": "Alias admitted." if request.approve else "Alias rejected.",
        "kb_entry_id": entry.kb_entry_id,
        "status": entry.status,
    }


@router.get("/aliases/pending")
def list_pending_aliases(db: Session = Depends(get_db)):
    entries = (
        db.query(KnowledgeBaseVersion)
        .filter(KnowledgeBaseVersion.status == "pending")
        .order_by(KnowledgeBaseVersion.kb_entry_id.desc())
        .all()
    )

    return {
        "count": len(entries),
        "entries": [
            {
                "kb_entry_id": e.kb_entry_id,
                "concept_id": e.concept_id,
                "alias_text": e.alias_text,
                "evidence_count": e.evidence_count,
                "status": e.status,
                "created_at": e.created_at.isoformat(),
            }
            for e in entries
        ],
    }


@router.get("/feedback")
def list_feedback(db: Session = Depends(get_db)):
    feedback = (
        db.query(SemanticFeedback)
        .order_by(SemanticFeedback.feedback_id.desc())
        .limit(200)
        .all()
    )

    return {
        "count": len(feedback),
        "feedback": [
            {
                "feedback_id": f.feedback_id,
                "dataset_id": f.dataset_id,
                "version_id": f.version_id,
                "table_name": f.table_name,
                "column_name": f.column_name,
                "decision": f.decision,
                "original_concept_id": f.original_concept_id,
                "corrected_concept_id": f.corrected_concept_id,
                "model_name": f.model_name,
                "created_at": f.created_at.isoformat(),
            }
            for f in feedback
        ],
    }

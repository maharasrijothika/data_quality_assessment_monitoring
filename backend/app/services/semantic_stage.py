"""Stage 04 semantic understanding: persistence + human-in-the-loop review."""

from datetime import datetime, timezone

from sqlalchemy.orm import Session

from app.config import (
    SEMANTIC_AMBIGUOUS_THRESHOLD,
    SEMANTIC_EMBEDDING_REPRESENTATION_VERSION,
    SEMANTIC_EVIDENCE_VERSION,
    SEMANTIC_MODEL_NAME,
    SEMANTIC_PROBABLE_EVIDENCE_THRESHOLD,
    SEMANTIC_PROBABLE_MARGIN,
    SEMANTIC_PROBABLE_THRESHOLD,
    SEMANTIC_SCORING_WEIGHTS,
    SEMANTIC_STRONG_MARGIN,
    SEMANTIC_STRONG_THRESHOLD,
)
from app.models import (
    Dataset,
    SemanticCandidateRecord,
    SemanticConcept,
    SemanticFeedback,
    SemanticPrediction,
    SemanticRun,
    StoredProfile,
)
from app.services import audit_artifacts
from app.services.semantic_analysis import SemanticAnalysisService
from app.services.semantic_dataset_analysis import SemanticDatasetAnalysisService
from app.services.semantic_kb import ensure_semantic_kb
from app.services.semantic_retrieval import kb_fingerprint


def confidence_level(
    score: float,
    evidence_coverage: float,
    margin: float | None = None,
    decision_kind: str | None = None,
) -> str:
    """Map an evidence-based score to an interpretable confidence category.

    The score is a weighted combination of real evidence similarities; it is
    NOT a calibrated probability. A candidate additionally needs sufficient
    evidence coverage (it must actually clear a substantial share of its
    applicable evidence) to reach "Probable" or better. Applicable-but-zero
    evidence (e.g. a description exists and disagrees) reduces coverage;
    structurally absent evidence (no description configured) does not.

    v2: when the top-2 margin is supplied, a candidate whose runner-up is
    within the margin is capped at "Ambiguous" regardless of its absolute
    score (a near-tie is never "Strong"). ``decision_kind="FAMILY_MATCH"``
    caps the level at "Ambiguous" because the SPECIFIC concept is unknown —
    the family answer is honest, but it is not a confident KB match. The
    two-argument call keeps its pre-margin behaviour for compatibility.
    """
    if decision_kind == "FAMILY_MATCH":
        if score >= SEMANTIC_AMBIGUOUS_THRESHOLD:
            return "Ambiguous"
        return "Unknown"

    strong_margin_ok = margin is None or margin >= SEMANTIC_STRONG_MARGIN
    probable_margin_ok = margin is None or margin >= SEMANTIC_PROBABLE_MARGIN

    if (
        score >= SEMANTIC_STRONG_THRESHOLD
        and evidence_coverage >= SEMANTIC_PROBABLE_EVIDENCE_THRESHOLD
        and strong_margin_ok
    ):
        return "Strong"
    if (
        score >= SEMANTIC_PROBABLE_THRESHOLD
        and evidence_coverage >= SEMANTIC_PROBABLE_EVIDENCE_THRESHOLD
        and probable_margin_ok
    ):
        return "Probable"
    if score >= SEMANTIC_AMBIGUOUS_THRESHOLD:
        return "Ambiguous"
    return "Unknown"


def get_or_create_profile(db: Session, dataset_id: int, version_id: int) -> dict:
    """Return the persisted profile for a version, profiling once if needed."""
    stored = (
        db.query(StoredProfile)
        .filter(StoredProfile.version_id == version_id)
        .first()
    )

    if stored is not None:
        return stored.profile_json

    from app.routers.profiling import get_dataset_profiling_result

    profile_result = get_dataset_profiling_result(db, dataset_id)

    db.add(
        StoredProfile(
            version_id=version_id,
            profile_json=profile_result,
            row_count=profile_result.get("total_rows", 0),
        )
    )
    db.flush()

    return profile_result


def run_semantic_analysis(
    db: Session,
    dataset: Dataset,
    version_id: int,
) -> dict:
    """Run semantic analysis over all columns and persist predictions.

    Existing pending predictions are replaced; approved decisions are kept.
    The Knowledge Base is seeded (idempotently) before analysis so a fresh
    database gets the shipped concepts, and KB embeddings are backfilled
    once per representation version.
    """
    # Seed the shipped KB baseline FIRST (idempotent) and make sure every
    # concept has a current-version embedding. Both are cheap no-ops when
    # the KB is already current.
    ensure_semantic_kb(db)

    profile_result = get_or_create_profile(db, dataset.dataset_id, version_id)

    dataset_service = SemanticDatasetAnalysisService()

    analysis = dataset_service.analyze_dataset(
        db=db,
        dataset=dataset,
        profiling_result=profile_result,
    )

    fingerprint = kb_fingerprint(db)

    run_number = (
        db.query(SemanticRun)
        .filter(SemanticRun.version_id == version_id)
        .count()
        + 1
    )

    existing = (
        db.query(SemanticPrediction)
        .filter(SemanticPrediction.version_id == version_id)
        .all()
    )

    kept: dict[tuple[str, str], SemanticPrediction] = {}
    for prediction in existing:
        if prediction.status in {"approved", "edited"}:
            kept[(prediction.table_name, prediction.column_name)] = prediction
        else:
            db.query(SemanticCandidateRecord).filter(
                SemanticCandidateRecord.prediction_id == prediction.prediction_id
            ).delete()
            db.delete(prediction)

    db.flush()

    now = datetime.now(timezone.utc)
    results = []

    for column_result in analysis["columns"]:
        key = (column_result["table_name"], column_result["column_name"])

        if key in kept:
            prediction = kept[key]
            concept = (
                db.query(SemanticConcept)
                .filter(SemanticConcept.concept_id == prediction.user_confirmed_concept_id)
                .first()
            )
            stored_evidence = prediction.evidence_json or {}
            results.append(
                {
                    "prediction_id": prediction.prediction_id,
                    "table_name": prediction.table_name,
                    "column_name": prediction.column_name,
                    "predicted_concept": concept.concept_name if concept else prediction.predicted_concept,
                    "concept_id": prediction.user_confirmed_concept_id,
                    "confidence_level": "Approved" if concept else prediction.confidence_level,
                    "status": prediction.status,
                    "source": "KB",
                    "match_status": "MATCHED",
                    "decision_kind": stored_evidence.get("decision_kind"),
                    "semantic_family": stored_evidence.get("semantic_family"),
                    "proposal": stored_evidence.get("proposal"),
                    "confidence_score": prediction.confidence_score,
                }
            )
            continue

        semantic = column_result["semantic_analysis"]
        evidence = semantic.get("evidence", {}) or {}

        # Coverage is computed by the engine from applicable evidence; the
        # stored payload keeps both values and applicability flags so the UI
        # can show WHY each piece of evidence was counted or skipped.
        coverage = semantic.get("evidence_coverage", 0.0)
        score = semantic.get("confidence_score", 0.0)
        decision_kind = semantic.get("decision_kind", "UNKNOWN")
        margin = semantic.get("margin")

        prediction = SemanticPrediction(
            version_id=version_id,
            table_name=column_result["table_name"],
            column_name=column_result["column_name"],
            predicted_concept_id=semantic.get("concept_id"),
            predicted_concept=semantic.get("semantic_concept"),
            confidence_score=score,
            confidence_level=confidence_level(
                score, coverage, margin=margin, decision_kind=decision_kind
            ),
            alternatives_json={"alternatives": semantic.get("alternatives", [])},
            evidence_json={
                "evidence_version": SEMANTIC_EVIDENCE_VERSION,
                "features": evidence,
                "evidence_coverage": coverage,
                "source": semantic.get("source", "OPEN_DISCOVERY"),
                "match_status": semantic.get("match_status", "UNKNOWN"),
                "normalization": semantic.get("normalization"),
                # v2 decision payload (new keys only; readers use .get()
                # defaults so v1 payloads keep loading).
                "decision_kind": decision_kind,
                "semantic_family": semantic.get("semantic_family"),
                "proposal": semantic.get("proposal"),
                "candidates": semantic.get("candidates", []),
                "decision_rule": semantic.get("decision_rule"),
                "margin": margin,
                "exact_stage": semantic.get("exact_stage", False),
                "exact_but_profile_conflict": semantic.get(
                    "exact_but_profile_conflict", False
                ),
                "kb_fingerprint": fingerprint,
                "run_number": run_number,
                "semantic_input": semantic.get("semantic_input"),
            },
            status="pending",
            created_at=now,
            updated_at=now,
        )
        db.add(prediction)
        db.flush()

        for alternative in semantic.get("alternatives", [])[:5]:
            db.add(
                SemanticCandidateRecord(
                    prediction_id=prediction.prediction_id,
                    concept_id=alternative.get("concept_id"),
                    concept_name=alternative.get("concept", "Unknown"),
                    confidence_score=alternative.get("score", 0.0),
                )
            )

        results.append(
            {
                "prediction_id": prediction.prediction_id,
                "table_name": prediction.table_name,
                "column_name": prediction.column_name,
                "predicted_concept": prediction.predicted_concept,
                "concept_id": prediction.predicted_concept_id,
                "confidence_level": prediction.confidence_level,
                "status": prediction.status,
                "source": semantic.get("source", "OPEN_DISCOVERY"),
                "match_status": semantic.get("match_status", "UNKNOWN"),
                "decision_kind": decision_kind,
                "semantic_family": semantic.get("semantic_family"),
                "proposal": semantic.get("proposal"),
                "confidence_score": score,
            }
        )

    db.flush()

    # ------------------------------------------------------------------
    # Append-only run record + audit artifacts. Every run is RECORDED so a
    # semantic-logic/KB/model change produces a NEW comparable run (S1, S2,
    # ...) instead of overwriting history. Approved/edited decisions kept
    # above are recorded with their human decision source.
    # ------------------------------------------------------------------
    inputs_by_column = {
        (column_result["table_name"], column_result["column_name"]): (
            column_result["semantic_analysis"].get("semantic_input")
        )
        for column_result in analysis["columns"]
    }

    run_record = SemanticRun(
        dataset_id=dataset.dataset_id,
        version_id=version_id,
        run_number=run_number,
        model_version=SEMANTIC_MODEL_NAME,
        embedding_representation_version=SEMANTIC_EMBEDDING_REPRESENTATION_VERSION,
        evidence_version=SEMANTIC_EVIDENCE_VERSION,
        kb_fingerprint=fingerprint,
        configuration_json={
            "scoring_weights": dict(SEMANTIC_SCORING_WEIGHTS),
            "strong_threshold": SEMANTIC_STRONG_THRESHOLD,
            "probable_threshold": SEMANTIC_PROBABLE_THRESHOLD,
            "ambiguous_threshold": SEMANTIC_AMBIGUOUS_THRESHOLD,
            "strong_margin": SEMANTIC_STRONG_MARGIN,
            "probable_margin": SEMANTIC_PROBABLE_MARGIN,
            "evidence_coverage_threshold": SEMANTIC_PROBABLE_EVIDENCE_THRESHOLD,
        },
        semantic_input_json={
            "columns": [
                inputs_by_column.get(
                    (column_result["table_name"], column_result["column_name"]),
                    {},
                )
                for column_result in analysis["columns"]
            ]
        },
        predictions_json={"columns": results},
    )
    db.add(run_record)
    db.flush()

    run_payload = {
        "dataset_id": dataset.dataset_id,
        "version_id": version_id,
        "run_id": run_record.run_id,
        "run_number": run_number,
        "created_at": run_record.created_at.isoformat(),
        "model_version": SEMANTIC_MODEL_NAME,
        "embedding_representation_version": (
            SEMANTIC_EMBEDDING_REPRESENTATION_VERSION
        ),
        "evidence_version": SEMANTIC_EVIDENCE_VERSION,
        "kb_fingerprint": fingerprint,
        "configuration": run_record.configuration_json,
        "semantic_input": run_record.semantic_input_json,
        "predictions": results,
    }

    try:
        audit_artifacts.export_semantic_input_artifact(
            dataset.dataset_id,
            version_id,
            run_number,
            run_payload,
        )
    except OSError:  # pragma: no cover - artifact export never breaks a run
        pass

    return {
        "dataset_id": dataset.dataset_id,
        "version_id": version_id,
        "run_id": run_record.run_id,
        "run_number": run_number,
        "column_count": len(results),
        "columns": results,
    }


def get_semantic_predictions(
    db: Session,
    version_id: int,
) -> list[SemanticPrediction]:
    return (
        db.query(SemanticPrediction)
        .filter(SemanticPrediction.version_id == version_id)
        .order_by(
            SemanticPrediction.table_name,
            SemanticPrediction.prediction_id,
        )
        .all()
    )


def apply_semantic_decision(
    db: Session,
    dataset: Dataset,
    version_id: int,
    prediction_id: int,
    decision: str,
    concept_id: int | None,
    user_defined_type: str | None = None,
) -> SemanticPrediction:
    """Record a human decision and persist feedback for offline learning.

    Decisions go to the Feedback Store only; the Knowledge Base is never
    updated directly from a single approval.

    A decision may attach an existing KB concept (concept_id) OR a free-text
    user-defined semantic type (user_defined_type) — never forced into the
    KB: user-defined types stay column-local facts and are NOT promoted into
    the production Knowledge Base by this function.
    """
    prediction = (
        db.query(SemanticPrediction)
        .filter(
            SemanticPrediction.prediction_id == prediction_id,
            SemanticPrediction.version_id == version_id,
        )
    .first()
    )

    if prediction is None:
        raise ValueError(f"Prediction {prediction_id} not found for this version.")

    if decision not in {"approved", "rejected", "edited"}:
        raise ValueError(f"Unsupported decision: {decision}")

    now = datetime.now(timezone.utc)

    # Capture the ORIGINAL recommendation BEFORE any mutation so the
    # feedback history always records what the system proposed.
    original_concept_id = prediction.predicted_concept_id

    concept: SemanticConcept | None = None
    if decision in {"approved", "edited"} and concept_id is not None:
        concept = (
            db.query(SemanticConcept)
            .filter(SemanticConcept.concept_id == concept_id)
            .first()
        )
        if concept is None:
            raise ValueError(f"Concept {concept_id} does not exist.")

    if decision == "edited" and concept is None:
        trimmed = (user_defined_type or "").strip()
        if not trimmed:
            raise ValueError(
                "An edit decision requires concept_id or user_defined_type."
            )

        # User-defined semantic type: stored on the prediction, kept out of
        # the KB. The original recommendation stays untouched for audit.
        prediction.user_defined_type = trimmed
        prediction.user_confirmed_concept_id = None
        prediction.predicted_concept_id = None
        # Final semantic type display value. predicted_concept (the ORIGINAL
        # system recommendation) would be overwritten here — instead the
        # user-defined value is stored separately and exposed by the router
        # as final_semantic_type.
        prediction.decision_source = "user_defined"
    elif decision in {"approved", "edited"}:
        prediction.user_defined_type = None
        prediction.user_confirmed_concept_id = concept.concept_id
        prediction.predicted_concept_id = concept.concept_id
        prediction.predicted_concept = concept.concept_name
        prediction.decision_source = "kb_approval"
    elif decision == "rejected":
        prediction.user_confirmed_concept_id = None
        prediction.decision_source = "rejection"

    prediction.status = decision
    prediction.decided_at = now
    prediction.updated_at = now

    # features_json keeps the FULL ranked candidate list (with each
    # candidate's evidence payload) so offline learning can see not only
    # what won but what it beat. falls back to the stored evidence payload.
    stored_evidence = prediction.evidence_json or {}
    features_payload = stored_evidence.get("candidates") or stored_evidence

    db.add(
        SemanticFeedback(
            dataset_id=dataset.dataset_id,
            version_id=version_id,
            prediction_id=prediction.prediction_id,
            table_name=prediction.table_name,
            column_name=prediction.column_name,
            original_concept_id=(
                original_concept_id if decision in {"rejected", "edited"} else None
            ),
            corrected_concept_id=(
                concept.concept_id
                if (decision in {"approved", "edited"} and concept is not None)
                else None
            ),
            decision=decision,
            features_json=features_payload,
            model_name=SEMANTIC_MODEL_NAME,
        )
    )

    db.flush()
    return prediction


def review_semantic_kb_candidate(
    db: Session,
    column_name: str,
    concept_id: int | None = None,
) -> dict:
    """Show evidence for an unseen column: candidates or UNKNOWN handling."""
    service = SemanticAnalysisService()
    return service.analyze_column(
        db=db,
        column_name=column_name,
    )


def semantic_feedback_summary(db: Session, dataset_id: int) -> dict:
    feedback = (
        db.query(SemanticFeedback)
        .filter(SemanticFeedback.dataset_id == dataset_id)
        .order_by(SemanticFeedback.feedback_id.desc())
        .all()
    )
    decisions = {"approved": 0, "rejected": 0, "edited": 0}
    for item in feedback:
        decisions[item.decision] = decisions.get(item.decision) + 1

    return {
        "total": len(feedback),
        "decisions": decisions,
        "recent": [
            {
                "feedback_id": item.feedback_id,
                "table_name": item.table_name,
                "column_name": item.column_name,
                "decision": item.decision,
                "original_concept_id": item.original_concept_id,
                "corrected_concept_id": item.corrected_concept_id,
                "created_at": item.created_at.isoformat(),
            }
            for item in feedback[:20]
        ],
    }

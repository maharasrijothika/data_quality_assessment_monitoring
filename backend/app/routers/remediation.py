from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import (
    DatasetVersion,
    RemediationApproval,
    RemediationRecord,
)
from app.services.stage_state import (
    get_dataset_or_none,
    get_latest_version,
    mark_stage_complete,
)
from app.services.remediation import (
    apply_remediation,
    propose_remediation,
)
from app.services.scoring import compute_dq_score, get_latest_score
from datetime import datetime, timezone

router = APIRouter(prefix="/datasets", tags=["remediation"])


class RemediationDecisionRequest(BaseModel):
    approve: bool


def _dataset_or_404(db: Session, dataset_id: int):
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


@router.post("/{dataset_id}/remediation/propose")
def propose(
    dataset_id: int,
    db: Session = Depends(get_db),
):
    dataset = _dataset_or_404(db, dataset_id)
    version = _version_or_404(db, dataset_id)

    proposal = propose_remediation(
        db,
        dataset.dataset_id,
        version.version_id,
        version.version_number,
    )

    record = RemediationRecord(
        dataset_id=dataset_id,
        source_version_id=version.version_id,
        remediation_type=proposal["remediation_type"],
        status="proposed",
        proposal_json=proposal,
        affected_rows=proposal["affected_rows"],
        correction_count=proposal["total_corrections"],
        before_score=(
            get_latest_score(db, version.version_id).overall_score
            if get_latest_score(db, version.version_id)
            else None
        ),
    )

    db.add(record)
    db.commit()

    return {
        "remediation_id": record.remediation_id,
        "status": record.status,
        "proposal": proposal,
    }


@router.get("/{dataset_id}/remediation")
def list_remediation(
    dataset_id: int,
    db: Session = Depends(get_db),
):
    dataset = _dataset_or_404(db, dataset_id)

    records = (
        db.query(RemediationRecord)
        .filter(RemediationRecord.dataset_id == dataset_id)
        .order_by(RemediationRecord.remediation_id.desc())
        .all()
    )

    return {
        "dataset_id": dataset_id,
        "count": len(records),
        "remediations": [
            {
                "remediation_id": r.remediation_id,
                "source_version_id": r.source_version_id,
                "resulting_version_id": r.resulting_version_id,
                "remediation_type": r.remediation_type,
                "status": r.status,
                "affected_rows": r.affected_rows,
                "correction_count": r.correction_count,
                "before_score": r.before_score,
                "after_score": r.after_score,
                "proposal": r.proposal_json,
            }
            for r in records
        ],
    }


@router.post("/{dataset_id}/remediation/{remediation_id}/decision")
def decide_remediation(
    dataset_id: int,
    remediation_id: int,
    request: RemediationDecisionRequest,
    db: Session = Depends(get_db),
):
    """Human decision on a remediation proposal, then apply + reassess."""
    dataset = _dataset_or_404(db, dataset_id)
    version = _version_or_404(db, dataset_id)

    record = (
        db.query(RemediationRecord)
        .filter(
            RemediationRecord.remediation_id == remediation_id,
            RemediationRecord.dataset_id == dataset_id,
        )
        .first()
    )

    if record is None:
        raise HTTPException(status_code=404, detail="Remediation record not found.")

    if record.status != "proposed":
        raise HTTPException(
            status_code=400,
            detail="Remediation has already been decided.",
        )

    if not request.approve:
        record.status = "rejected"

        db.add(
            RemediationApproval(
                remediation_id=record.remediation_id,
                decision="rejected",
                decided_at=datetime.now(timezone.utc),
            )
        )
        db.commit()

        return {
            "message": "Remediation rejected.",
            "remediation_id": remediation_id,
            "status": "rejected",
        }

    db.add(
        RemediationApproval(
            remediation_id=record.remediation_id,
            decision="approved",
            decided_at=datetime.now(timezone.utc),
        )
    )

    proposal = record.proposal_json

    # Carry approved semantic decisions into the new version so the
    # pipeline (recommendations, rules, scores) continues seamlessly.
    from app.models import SemanticPrediction

    approved_predictions = (
        db.query(SemanticPrediction)
        .filter(
            SemanticPrediction.version_id == record.source_version_id,
            SemanticPrediction.status.in_(["approved", "edited"]),
        )
        .all()
    )

    result = apply_remediation(
        db=db,
        dataset_id=dataset_id,
        version_id=record.source_version_id,
        version_number=(
            db.get(DatasetVersion, record.source_version_id).version_number
        ),
        proposal=proposal,
    )

    record.resulting_version_id = result["new_version_id"]
    record.correction_count = result["correction_count"]
    record.status = "applied"

    # Reassess: re-run approved rules on the new version and re-score.
    from app.models import Rule
    from app.services.rule_execution import execute_rule, RuleExecutionError

    new_version_id = result["new_version_id"]
    new_version = db.get(DatasetVersion, new_version_id)

    rules = (
        db.query(Rule)
        .filter(
            Rule.version_id == record.source_version_id,
            Rule.status == "approved",
        )
        .all()
    )

    reassessment = []

    for rule in rules:
        new_rule = Rule(
            dataset_id=dataset_id,
            version_id=new_version_id,
            table_name=rule.table_name,
            rule_name=rule.rule_name,
            rule_json=rule.rule_json,
            metric=rule.metric,
            source=rule.source,
            risk=rule.risk,
            validation_status=rule.validation_status,
            validation_issues_json=rule.validation_issues_json,
            status="approved",
        )
        db.add(new_rule)
        db.flush()

        try:
            execution = execute_rule(
                db=db,
                rule=new_rule,
                dataset_id=dataset_id,
                version_id=new_version_id,
                version_number=new_version.version_number,
            )
            reassessment.append(
                {
                    "rule_name": rule.rule_name,
                    "failed_rows": execution.failed_rows,
                    "pass_rate": execution.pass_rate,
                }
            )
        except RuleExecutionError as exc:
            reassessment.append(
                {
                    "rule_name": rule.rule_name,
                    "error": str(exc),
                }
            )

    after_score = None
    if rules:
        score = compute_dq_score(db, new_version_id)
        after_score = score.overall_score

    record.after_score = after_score

    mark_stage_complete(db, dataset_id, "remediation")
    db.commit()

    before_score = record.before_score

    return {
        "message": "Remediation applied and reassessed.",
        "remediation_id": remediation_id,
        "new_version_id": new_version_id,
        "new_version_number": new_version.version_number,
        "correction_count": result["correction_count"],
        "before_score": before_score,
        "after_score": after_score,
        "score_delta": (
            round(after_score - before_score, 2)
            if after_score is not None and before_score is not None
            else None
        ),
        "reassessment": reassessment,
    }


@router.get("/{dataset_id}/versions")
def list_versions(
    dataset_id: int,
    db: Session = Depends(get_db),
):
    dataset = _dataset_or_404(db, dataset_id)

    versions = (
        db.query(DatasetVersion)
        .filter(DatasetVersion.dataset_id == dataset_id)
        .order_by(DatasetVersion.version_number)
        .all()
    )

    from app.models import DQScore

    version_list = []

    for version in versions:
        score = get_latest_score(db, version.version_id)

        version_list.append(
            {
                "version_id": version.version_id,
                "version_number": version.version_number,
                "parent_version_id": version.parent_version_id,
                "schema_fingerprint": version.schema_fingerprint,
                "content_fingerprint": version.content_fingerprint,
                "uploaded_at": version.uploaded_at.isoformat(),
                "overall_score": score.overall_score if score else None,
            }
        )

    return {
        "dataset_id": dataset_id,
        "count": len(version_list),
        "versions": version_list,
    }

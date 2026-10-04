from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import (
    ColumnMetricContext,
    Dataset,
    RCAFinding,
    Rule,
    RuleApproval,
    RuleExecution,
    RuleFeedback,
    RuleVersion,
)
from app.services.stage_state import (
    get_dataset_or_none,
    get_latest_version,
    mark_stage_complete,
)
from app.services.metric_applicability import MetricApplicabilityEngine
from app.services.rule_interpretation import interpret_business_rule
from app.services.rule_recommendation import (
    recommend_rules_for_version,
    save_recommendations,
)
from app.services.rule_templates import (
    TEMPLATES,
    METRICS,
    METRIC_LABELS,
    templates_for_metric,
)
from app.services.rule_validation import validate_rule_dict
from app.services.rule_execution import (
    RuleExecutionError,
    _find_stored_file_for_table,
    execute_rule,
)
from app.services.rca import analyze_rule_failures
from app.services.scoring import compute_dq_score, get_latest_score
from datetime import datetime, timezone

router = APIRouter(tags=["rules"])


class ApproveRuleRequest(BaseModel):
    decision: str
    edited_rule_json: dict | None = None
    edited_rule_name: str | None = None
    approved_by: str | None = None


class ExecuteRulesRequest(BaseModel):
    rule_ids: list[int] = Field(default_factory=list)


class InterpretRuleRequest(BaseModel):
    text: str
    table_name: str | None = None
    column_name: str | None = None


class ManualRuleRequest(BaseModel):
    metric: str
    rule_template: str
    table_name: str
    column_name: str | None = None
    columns: list[str] = Field(default_factory=list)
    parameters: dict = Field(default_factory=dict)
    rule_name: str | None = None
    original_text: str | None = None


class MetricContextRequest(BaseModel):
    table_name: str
    column_name: str
    metric: str
    required: bool | None = None
    business_note: str | None = None


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


def _candidate_key(rule_json: dict) -> str:
    params = rule_json.get("parameters", {}) or {}
    columns = rule_json.get("columns") or ([rule_json["column"]] if rule_json.get("column") else [])
    return "|".join(
        [
            str(rule_json.get("rule_template") or rule_json.get("type") or "?"),
            str(rule_json.get("table") or "?"),
            ",".join(str(c) for c in columns),
            ",".join(f"{k}={params[k]}" for k in sorted(params)),
        ]
    )


def _store_rule_feedback(
    db: Session,
    rule: Rule,
    decision: str,
    edited_rule_json: dict | None = None,
) -> None:
    """Feedback-store write: candidate + evidence + human decision.

    This NEVER changes current-run behavior - it only records labels for
    future offline ranker training (XGBRanker), exactly like semantic and
    relationship feedback do. Features are captured at decision time from
    the same stored evidence the recommendation used (profile + semantic
    final type), so future training needs no recomputation.
    """
    rule_json = edited_rule_json or rule.rule_json or {}
    params = rule_json.get("parameters", {}) or {}
    columns = rule_json.get("columns") or (
        [rule_json["column"]] if rule_json.get("column") else []
    )
    column = columns[0] if columns else params.get("child_column")

    # Decision-time features from stored evidence (deterministic re-read).
    features: dict = {
        "metric": rule_json.get("metric") or rule.metric,
        "rule_template": str(
            rule_json.get("rule_template") or rule_json.get("type") or "unknown"
        ),
        "source": rule.source,
    }
    evidence: dict = {
        "validation_status": rule.validation_status,
        "validation_issues": rule.validation_issues_json.get("issues", []),
    }

    if column:
        try:
            engine = MetricApplicabilityEngine(db, rule.dataset_id, rule.version_id)
            record = engine.applicability_for_column(rule.table_name, column)
            features["semantic_type"] = record.get("semantic_type")
            features["semantic_confidence"] = record.get("semantic_confidence")
            features["data_type"] = record.get("data_type")
            metric_ev = record.get("metrics", {}).get(rule.metric, {})
            evidence["applicability_status"] = metric_ev.get("status")
            evidence["applicability_evidence"] = metric_ev.get("evidence", [])

            column_profile = engine._column_profile_lookup(rule.table_name, column)
            if column_profile:
                features["null_percentage"] = column_profile.get("null_percentage")
                features["distinct_percentage"] = column_profile.get("distinct_percentage")
                features["identifier_signal"] = column_profile.get("identifier_signal")
        except Exception:  # pragma: no cover - feedback must never break approval
            pass

    db.add(
        RuleFeedback(
            rule_id=rule.rule_id,
            dataset_id=rule.dataset_id,
            version_id=rule.version_id,
            candidate_key=_candidate_key(rule_json),
            metric=rule_json.get("metric") or rule.metric,
            rule_template=str(
                rule_json.get("rule_template") or rule_json.get("type") or "unknown"
            ),
            target_table=rule.table_name,
            target_columns=columns,
            decision=decision,
            evidence_json=evidence,
            features_json=features,
            recommendation_source="system" if rule.source == "system" else rule.source,
            edited_rule_json=edited_rule_json,
        )
    )


def _record_rule_version(
    db: Session,
    rule: Rule,
    change_kind: str,
    approved_by: str | None = None,
) -> None:
    db.add(
        RuleVersion(
            rule_id=rule.rule_id,
            version_number=rule.version_number,
            rule_code=rule.rule_code,
            rule_name=rule.rule_name,
            rule_json=rule.rule_json,
            metric=rule.metric,
            change_kind=change_kind,
            approved_by=approved_by,
        )
    )


def _next_rule_code(db: Session, dataset_id: int, version_id: int) -> str:
    count = (
        db.query(Rule)
        .filter(
            Rule.dataset_id == dataset_id,
            Rule.version_id == version_id,
            Rule.rule_code.isnot(None),
        )
        .count()
    )
    return f"R{count + 1:03d}"


def _rule_response(rule: Rule) -> dict:
    rule_json = rule.rule_json or {}
    return {
        "rule_id": rule.rule_id,
        "table_name": rule.table_name,
        "rule_name": rule.rule_name,
        "rule_json": rule_json,
        "metric": rule.metric,
        "rule_template": rule_json.get("rule_template") or rule_json.get("type"),
        "parameters": rule_json.get("parameters", {}),
        "source": rule.source,
        "risk": rule.risk,
        "rule_code": rule.rule_code,
        "version_number": rule.version_number,
        "validation_status": rule.validation_status,
        "validation_issues": rule.validation_issues_json.get("issues", []),
        "status": rule.status,
        "created_at": rule.created_at.isoformat() if rule.created_at else None,
        "updated_at": rule.updated_at.isoformat() if rule.updated_at else None,
    }


# ---------------------------------------------------------------------------
# Metric applicability
# ---------------------------------------------------------------------------


@router.get("/datasets/{dataset_id}/applicability")
def get_applicability(
    dataset_id: int,
    db: Session = Depends(get_db),
):
    """Evidence-based metric applicability matrix for the latest version."""
    _dataset_or_404(db, dataset_id)
    version = _version_or_404(db, dataset_id)

    try:
        engine = MetricApplicabilityEngine(db, dataset_id, version.version_id)
        matrix = engine.applicability_matrix()
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return {
        "dataset_id": dataset_id,
        "version_id": version.version_id,
        **matrix,
    }


# ---------------------------------------------------------------------------
# Metric contexts (business requiredness per column/metric)
# ---------------------------------------------------------------------------


@router.post("/datasets/{dataset_id}/metric-contexts")
def set_metric_context(
    dataset_id: int,
    request: MetricContextRequest,
    db: Session = Depends(get_db),
):
    """Record a business-requiredness decision for a column/metric pair."""
    _dataset_or_404(db, dataset_id)
    version = _version_or_404(db, dataset_id)

    if request.metric not in METRICS:
        raise HTTPException(status_code=400, detail=f"Unknown metric: {request.metric}")

    context = (
        db.query(ColumnMetricContext)
        .filter(
            ColumnMetricContext.dataset_id == dataset_id,
            ColumnMetricContext.version_id == version.version_id,
            ColumnMetricContext.table_name == request.table_name,
            ColumnMetricContext.column_name == request.column_name,
            ColumnMetricContext.metric == request.metric,
        )
        .first()
    )

    if context is None:
        context = ColumnMetricContext(
            dataset_id=dataset_id,
            version_id=version.version_id,
            table_name=request.table_name,
            column_name=request.column_name,
            metric=request.metric,
        )
        db.add(context)

    context.required = request.required
    context.business_note = request.business_note
    db.commit()

    return {
        "message": "Metric context saved.",
        "context": {
            "table_name": context.table_name,
            "column_name": context.column_name,
            "metric": context.metric,
            "required": context.required,
            "business_note": context.business_note,
        },
    }


@router.get("/datasets/{dataset_id}/metric-contexts")
def list_metric_contexts(
    dataset_id: int,
    db: Session = Depends(get_db),
):
    _dataset_or_404(db, dataset_id)
    version = _version_or_404(db, dataset_id)

    contexts = (
        db.query(ColumnMetricContext)
        .filter(ColumnMetricContext.version_id == version.version_id)
        .all()
    )

    return {
        "contexts": [
            {
                "table_name": c.table_name,
                "column_name": c.column_name,
                "metric": c.metric,
                "required": c.required,
                "business_note": c.business_note,
            }
            for c in contexts
        ]
    }


# ---------------------------------------------------------------------------
# Business-rule interpretation (text -> structured candidate; never executed)
# ---------------------------------------------------------------------------


@router.post("/datasets/{dataset_id}/rules/interpret")
def interpret_rule(
    dataset_id: int,
    request: InterpretRuleRequest,
    db: Session = Depends(get_db),
):
    _dataset_or_404(db, dataset_id)

    result = interpret_business_rule(
        request.text,
        default_table=request.table_name,
        default_column=request.column_name,
    )

    # If resolved, immediately validate the structural candidate so the UI
    # can show VALID/NEEDS_REVIEW/INVALID next to the interpretation.
    validation = None
    if result["status"] == "RESOLVED" and result["target"].get("table"):
        candidate_json = {
            "rule_template": result["rule_template"],
            "metric": result["metric"],
            "table": result["target"]["table"],
            "column": result["target"]["columns"][0] if result["target"]["columns"] else None,
            "columns": result["target"]["columns"] or None,
            "parameters": result["parameters"],
        }
        version = _version_or_404(db, dataset_id)
        validation = validate_rule_dict(db, dataset_id, version.version_id, candidate_json)
        result["candidate_json"] = candidate_json
        result["validation"] = validation

    return result


# ---------------------------------------------------------------------------
# Manual structured rule creation (user chooses metric/template/parameters)
# ---------------------------------------------------------------------------


@router.post("/datasets/{dataset_id}/rules/manual")
def create_manual_rule(
    dataset_id: int,
    request: ManualRuleRequest,
    db: Session = Depends(get_db),
):
    _dataset_or_404(db, dataset_id)
    version = _version_or_404(db, dataset_id)

    if request.metric not in METRICS:
        raise HTTPException(status_code=400, detail=f"Unknown metric: {request.metric}")

    if request.rule_template not in TEMPLATES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown rule template: {request.rule_template}",
        )

    spec = TEMPLATES[request.rule_template]
    if spec.metric != request.metric:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Template {request.rule_template} belongs to metric {spec.metric}, "
                f"not {request.metric}."
            ),
        )

    rule_json = {
        "rule_template": request.rule_template,
        "metric": request.metric,
        "table": request.table_name,
        "column": request.column_name,
        "columns": request.columns or None,
        "parameters": request.parameters,
        "source_text": request.original_text,
    }

    rule_name = request.rule_name or (
        f"{request.column_name or request.table_name}: {request.rule_template}"
    )

    validation = validate_rule_dict(db, dataset_id, version.version_id, rule_json)

    rule = Rule(
        dataset_id=dataset_id,
        version_id=version.version_id,
        table_name=request.table_name,
        rule_name=rule_name,
        rule_json=rule_json,
        metric=request.metric,
        source="user",
        risk="low",
        validation_status=validation["status"],
        validation_issues_json={"issues": validation["issues"]},
        status="recommended",
    )
    db.add(rule)
    db.flush()

    mark_stage_complete(db, dataset_id, "recommendations")
    db.commit()

    return {
        "message": f"Rule created (validation: {validation['status']}).",
        "rule": _rule_response(rule),
        "validation": validation,
    }


# ---------------------------------------------------------------------------
# Template registry introspection (UI form building)
# ---------------------------------------------------------------------------


@router.get("/datasets/{dataset_id}/rule-templates")
def list_rule_templates(
    dataset_id: int,
    db: Session = Depends(get_db),
):
    _dataset_or_404(db, dataset_id)

    return {
        "metrics": [
            {"name": metric, "label": METRIC_LABELS.get(metric, metric)}
            for metric in METRICS
        ],
        "templates": [
            {
                "name": spec.name,
                "metric": spec.metric,
                "target": spec.target,
                "parameters": [
                    {
                        "name": p.name,
                        "kind": p.kind,
                        "required": p.required,
                    }
                    for p in spec.parameters
                ],
                "requires": list(spec.requires),
                "description": spec.description,
            }
            for spec in TEMPLATES.values()
        ],
    }


# ---------------------------------------------------------------------------
# Recommendations
# ---------------------------------------------------------------------------


@router.post("/datasets/{dataset_id}/recommendations")
def generate_recommendations(
    dataset_id: int,
    db: Session = Depends(get_db),
):
    """Generate metric applicability + rule candidates for the latest version."""
    dataset = _dataset_or_404(db, dataset_id)
    version = _version_or_404(db, dataset_id)

    try:
        recommendations = recommend_rules_for_version(
            db, dataset.dataset_id, version.version_id
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    # Replace previous unapproved recommendations; approved rules are kept.
    existing = (
        db.query(Rule)
        .filter(
            Rule.version_id == version.version_id,
            Rule.status == "recommended",
        )
        .all()
    )

    for rule in existing:
        db.delete(rule)

    db.flush()

    saved = save_recommendations(
        db, dataset.dataset_id, version.version_id, recommendations
    )

    mark_stage_complete(db, dataset_id, "recommendations")
    db.commit()

    return {
        "dataset_id": dataset_id,
        "version_id": version.version_id,
        "recommendation_count": len(saved),
        "rules": [_rule_response(rule) for rule in saved],
    }


@router.get("/datasets/{dataset_id}/rules")
def list_rules(
    dataset_id: int,
    status: str | None = None,
    db: Session = Depends(get_db),
):
    dataset = _dataset_or_404(db, dataset_id)
    version = _version_or_404(db, dataset_id)

    query = db.query(Rule).filter(Rule.version_id == version.version_id)

    if status:
        query = query.filter(Rule.status == status)

    rules = query.order_by(Rule.rule_id).all()

    return {
        "dataset_id": dataset_id,
        "version_id": version.version_id,
        "count": len(rules),
        "rules": [_rule_response(rule) for rule in rules],
    }


@router.get("/rules/{rule_id}/versions")
def list_rule_versions(
    rule_id: int,
    db: Session = Depends(get_db),
):
    rule = db.get(Rule, rule_id)
    if rule is None:
        raise HTTPException(status_code=404, detail="Rule not found.")

    versions = (
        db.query(RuleVersion)
        .filter(RuleVersion.rule_id == rule_id)
        .order_by(RuleVersion.version_number)
        .all()
    )

    return {
        "rule_id": rule_id,
        "current_version_number": rule.version_number,
        "versions": [
            {
                "version_number": v.version_number,
                "rule_code": v.rule_code,
                "rule_name": v.rule_name,
                "rule_json": v.rule_json,
                "metric": v.metric,
                "change_kind": v.change_kind,
                "approved_by": v.approved_by,
                "created_at": v.created_at.isoformat(),
            }
            for v in versions
        ],
    }


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@router.post("/rules/{rule_id}/validate")
def validate_rule(
    rule_id: int,
    db: Session = Depends(get_db),
):
    rule = db.get(Rule, rule_id)

    if rule is None:
        raise HTTPException(status_code=404, detail="Rule not found.")

    validation = validate_rule_dict(
        db, rule.dataset_id, rule.version_id, rule.rule_json
    )

    rule.validation_status = validation["status"]
    rule.validation_issues_json = {"issues": validation["issues"]}

    mark_stage_complete(db, rule.dataset_id, "validation")
    db.commit()

    return {
        "rule_id": rule.rule_id,
        "validation_status": validation["status"],
        "issues": validation["issues"],
    }


@router.post("/datasets/{dataset_id}/rules/validate-all")
def validate_all_rules(
    dataset_id: int,
    db: Session = Depends(get_db),
):
    dataset = _dataset_or_404(db, dataset_id)
    version = _version_or_404(db, dataset_id)

    rules = (
        db.query(Rule)
        .filter(Rule.version_id == version.version_id)
        .all()
    )

    results = []
    for rule in rules:
        validation = validate_rule_dict(db, dataset_id, version.version_id, rule.rule_json)
        rule.validation_status = validation["status"]
        rule.validation_issues_json = {"issues": validation["issues"]}
        results.append(
            {
                "rule_id": rule.rule_id,
                "validation_status": validation["status"],
                "issues": validation["issues"],
            }
        )

    mark_stage_complete(db, dataset_id, "validation")
    db.commit()

    return {"validated": len(results), "results": results}


# ---------------------------------------------------------------------------
# Approval (Accept / Reject / Edit) with versioning + feedback
# ---------------------------------------------------------------------------


@router.post("/rules/{rule_id}/approve")
def approve_rule(
    rule_id: int,
    request: ApproveRuleRequest,
    db: Session = Depends(get_db),
):
    """Human decision: approve / reject, optionally with an edited definition.

    Editing bumps the rule version and records an immutable snapshot; an
    INVALID edit can never be approved.
    """
    rule = db.get(Rule, rule_id)

    if rule is None:
        raise HTTPException(status_code=404, detail="Rule not found.")

    if request.decision not in {"approved", "rejected"}:
        raise HTTPException(
            status_code=400,
            detail="Decision must be 'approved' or 'rejected'.",
        )

    if request.edited_rule_json is not None:
        if request.decision != "approved":
            raise HTTPException(
                status_code=400,
                detail="Edited rule definition requires approval.",
            )

        edited = dict(request.edited_rule_json)

        validation = validate_rule_dict(
            db, rule.dataset_id, rule.version_id, edited
        )

        if validation["status"] == "INVALID":
            raise HTTPException(
                status_code=400,
                detail={
                    "message": "Edited rule is invalid and cannot be approved.",
                    "issues": validation["issues"],
                },
            )

        if validation["status"] == "NEEDS_REVIEW":
            # NEEDS_REVIEW edits require the user to approve knowingly; the
            # API accepts them but returns the issues in the response.
            pass

        was_approved = rule.status == "approved"

        if was_approved:
            # The outgoing definition is already immortalized by its
            # initial_approval snapshot; just advance the version counter.
            rule.version_number = (rule.version_number or 1) + 1

        rule.rule_json = edited
        rule.metric = edited.get("metric") or rule.metric
        if request.edited_rule_name:
            rule.rule_name = request.edited_rule_name
        rule.validation_status = validation["status"]
        rule.validation_issues_json = {"issues": validation["issues"]}

        _record_rule_version(db, rule, change_kind="edited", approved_by=request.approved_by)

    if rule.status == "recommended" and request.decision == "approved" and not rule.rule_code:
        rule.rule_code = _next_rule_code(db, rule.dataset_id, rule.version_id)
        _record_rule_version(
            db, rule, change_kind="initial_approval", approved_by=request.approved_by
        )

    rule.status = request.decision

    _store_rule_feedback(db, rule, "accepted" if request.decision == "approved" else "rejected", request.edited_rule_json)

    db.add(
        RuleApproval(
            rule_id=rule.rule_id,
            decision=request.decision,
            edited_rule_json=request.edited_rule_json,
            decided_at=datetime.now(timezone.utc),
        )
    )

    db.commit()

    return {
        "message": f"Rule {request.decision} (version {rule.version_number}).",
        "rule_id": rule.rule_id,
        "rule_code": rule.rule_code,
        "version_number": rule.version_number,
        "status": rule.status,
    }


@router.get("/datasets/{dataset_id}/rule-feedback")
def list_rule_feedback(
    dataset_id: int,
    db: Session = Depends(get_db),
):
    """Feedback-store contents for this dataset (future offline training)."""
    _dataset_or_404(db, dataset_id)

    feedback = (
        db.query(RuleFeedback)
        .filter(RuleFeedback.dataset_id == dataset_id)
        .order_by(RuleFeedback.feedback_id.desc())
        .limit(500)
        .all()
    )

    return {
        "count": len(feedback),
        "feedback": [
            {
                "feedback_id": f.feedback_id,
                "rule_id": f.rule_id,
                "candidate_key": f.candidate_key,
                "metric": f.metric,
                "rule_template": f.rule_template,
                "decision": f.decision,
                "features": f.features_json,
                "recommendation_score": f.recommendation_score,
                "confidence": f.confidence,
                "created_at": f.created_at.isoformat(),
            }
            for f in feedback
        ],
    }


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


@router.post("/datasets/{dataset_id}/execute")
def execute_rules(
    dataset_id: int,
    request: ExecuteRulesRequest,
    db: Session = Depends(get_db),
):
    """Execute approved rules for the latest version."""
    dataset = _dataset_or_404(db, dataset_id)
    version = _version_or_404(db, dataset_id)

    query = db.query(Rule).filter(
        Rule.version_id == version.version_id,
        Rule.status == "approved",
    )

    if request.rule_ids:
        query = query.filter(Rule.rule_id.in_(request.rule_ids))

    rules = query.all()

    if not rules:
        raise HTTPException(
            status_code=400,
            detail="No approved rules to execute.",
        )

    results = []
    errors = 0

    for rule in rules:
        try:
            execution = execute_rule(
                db=db,
                rule=rule,
                dataset_id=dataset.dataset_id,
                version_id=version.version_id,
                version_number=version.version_number,
            )
            db.flush()
            results.append(
                {
                    "rule_id": rule.rule_id,
                    "rule_code": rule.rule_code,
                    "version_number": rule.version_number,
                    "rule_name": rule.rule_name,
                    "execution_id": execution.execution_id,
                    "status": execution.status,
                    "total_rows": execution.total_rows,
                    "applicable_rows": execution.applicable_rows,
                    "passed_rows": execution.passed_rows,
                    "failed_rows": execution.failed_rows,
                    "pass_rate": execution.evidence_json.get("pass_rate_not_applicable")
                    and None
                    or execution.pass_rate,
                    "violation_rate": execution.evidence_json.get(
                        "violation_rate_not_applicable"
                    )
                    and None
                    or execution.violation_rate,
                }
            )
        except RuleExecutionError as exc:
            # An execution error MUST be persisted (status="error") so
            # scoring can exclude it from pass rates instead of silently
            # treating the rule as if it never ran.
            errors += 1

            db.add(
                RuleExecution(
                    rule_id=rule.rule_id,
                    version_id=version.version_id,
                    status="error",
                    error_message=str(exc),
                )
            )
            db.flush()

            results.append(
                {
                    "rule_id": rule.rule_id,
                    "rule_code": rule.rule_code,
                    "rule_name": rule.rule_name,
                    "status": "error",
                    "message": str(exc),
                }
            )

    mark_stage_complete(db, dataset_id, "execution")
    db.commit()

    return {
        "dataset_id": dataset_id,
        "version_id": version.version_id,
        "executed": len([r for r in results if r["status"] != "error"]),
        "errors": errors,
        "results": results,
    }


@router.get("/datasets/{dataset_id}/executions")
def list_executions(
    dataset_id: int,
    db: Session = Depends(get_db),
):
    dataset = _dataset_or_404(db, dataset_id)
    version = _version_or_404(db, dataset_id)

    rules = (
        db.query(Rule)
        .filter(Rule.version_id == version.version_id)
        .all()
    )

    rule_map = {rule.rule_id: rule for rule in rules}

    executions = (
        db.query(RuleExecution)
        .filter(RuleExecution.version_id == version.version_id)
        .order_by(RuleExecution.execution_id.desc())
        .all()
    )

    return {
        "dataset_id": dataset_id,
        "version_id": version.version_id,
        "count": len(executions),
        "executions": [
            {
                "execution_id": e.execution_id,
                "rule_id": e.rule_id,
                "rule_code": rule_map[e.rule_id].rule_code if e.rule_id in rule_map else None,
                "rule_name": rule_map[e.rule_id].rule_name if e.rule_id in rule_map else None,
                "metric": rule_map[e.rule_id].metric if e.rule_id in rule_map else None,
                "rule_version_number": e.evidence_json.get("rule_version_number"),
                "status": e.status,
                "error_message": e.error_message,
                "total_rows": e.total_rows,
                "applicable_rows": e.applicable_rows,
                "passed_rows": e.passed_rows,
                "failed_rows": e.failed_rows,
                "pass_rate": None
                if e.evidence_json.get("pass_rate_not_applicable")
                else e.pass_rate,
                "violation_rate": None
                if e.evidence_json.get("violation_rate_not_applicable")
                else e.violation_rate,
                "failure_examples": e.evidence_json.get("failure_examples", []),
                "executed_at": e.executed_at.isoformat(),
            }
            for e in executions
        ],
    }


# ---------------------------------------------------------------------------
# RCA
# ---------------------------------------------------------------------------


@router.post("/datasets/{dataset_id}/rca")
def run_rca(
    dataset_id: int,
    db: Session = Depends(get_db),
):
    """Run evidence-based root cause analysis on the latest executions."""
    dataset = _dataset_or_404(db, dataset_id)
    version = _version_or_404(db, dataset_id)

    rules = (
        db.query(Rule)
        .filter(Rule.version_id == version.version_id)
        .all()
    )

    rule_map = {rule.rule_id: rule for rule in rules}

    executions = (
        db.query(RuleExecution)
        .filter(
            RuleExecution.version_id == version.version_id,
            RuleExecution.failed_rows > 0,
        )
        .all()
    )

    if not executions:
        return {
            "dataset_id": dataset_id,
            "version_id": version.version_id,
            "findings": [],
            "message": "No rule failures to analyze.",
        }

    db.query(RCAFinding).filter(RCAFinding.version_id == version.version_id).delete()
    db.flush()

    findings = []

    for execution in executions:
        rule = rule_map.get(execution.rule_id)

        if rule is None:
            continue

        stored_filename = _find_stored_file_for_table(
            db, version.version_id, rule.table_name
        )

        if stored_filename is None:
            continue

        analysis = analyze_rule_failures(
            db=db,
            rule=rule,
            execution=execution,
            dataset_id=dataset.dataset_id,
            version_number=version.version_number,
            stored_filename=stored_filename,
        )

        finding = RCAFinding(
            version_id=version.version_id,
            rule_id=rule.rule_id,
            failure_count=execution.failed_rows,
            analysis_json=analysis,
        )

        db.add(finding)
        findings.append(
            {
                "finding_id": finding.finding_id,
                "rule_id": rule.rule_id,
                "rule_name": rule.rule_name,
                "failure_count": execution.failed_rows,
                "analysis": analysis,
            }
        )

    mark_stage_complete(db, dataset_id, "rca")
    db.commit()

    return {
        "dataset_id": dataset_id,
        "version_id": version.version_id,
        "findings": findings,
    }


@router.get("/datasets/{dataset_id}/rca")
def get_rca(
    dataset_id: int,
    db: Session = Depends(get_db),
):
    dataset = _dataset_or_404(db, dataset_id)
    version = _version_or_404(db, dataset_id)

    findings = (
        db.query(RCAFinding)
        .filter(RCAFinding.version_id == version.version_id)
        .all()
    )

    rules = (
        db.query(Rule)
        .filter(Rule.version_id == version.version_id)
        .all()
    )

    rule_map = {rule.rule_id: rule for rule in rules}

    return {
        "dataset_id": dataset_id,
        "version_id": version.version_id,
        "findings": [
            {
                "finding_id": f.finding_id,
                "rule_id": f.rule_id,
                "rule_name": rule_map[f.rule_id].rule_name if f.rule_id in rule_map else None,
                "failure_count": f.failure_count,
                "analysis": f.analysis_json,
            }
            for f in findings
        ],
    }


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


@router.post("/datasets/{dataset_id}/scores")
def calculate_scores(
    dataset_id: int,
    db: Session = Depends(get_db),
):
    """Calculate DQ scores from the latest executions."""
    dataset = _dataset_or_404(db, dataset_id)
    version = _version_or_404(db, dataset_id)

    approved_rules = (
        db.query(Rule)
        .filter(
            Rule.version_id == version.version_id,
            Rule.status == "approved",
        )
        .count()
    )

    if approved_rules == 0:
        raise HTTPException(
            status_code=400,
            detail="No approved rules exist. Approve and execute rules first.",
        )

    score = compute_dq_score(db, version.version_id)

    mark_stage_complete(db, dataset_id, "scoring")
    db.commit()

    return {
        "dataset_id": dataset_id,
        "version_id": version.version_id,
        "overall_score": score.overall_score,
        "metric_count": score.metric_count,
        "excluded_metrics": score.excluded_metrics,
        "details": score.details_json,
    }


@router.get("/datasets/{dataset_id}/scores")
def get_scores(
    dataset_id: int,
    db: Session = Depends(get_db),
):
    dataset = _dataset_or_404(db, dataset_id)
    version = _version_or_404(db, dataset_id)

    score = get_latest_score(db, version.version_id)

    if score is None:
        raise HTTPException(
            status_code=404,
            detail="No DQ score computed for the current version yet.",
        )

    return {
        "dataset_id": dataset_id,
        "version_id": version.version_id,
        "overall_score": score.overall_score,
        "weighted": bool(score.weighted),
        "metric_count": score.metric_count,
        "excluded_metrics": score.excluded_metrics,
        "details": score.details_json,
        "computed_at": score.computed_at.isoformat(),
    }

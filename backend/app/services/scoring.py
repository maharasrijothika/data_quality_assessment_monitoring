"""Stage 09: DQ scoring.

Each metric produces an independent score from its rule executions. The
overall score is the mean of available metric scores. Execution errors and
N/A metrics are never treated as passing - they are surfaced explicitly.
"""

from datetime import datetime, timezone

from sqlalchemy.orm import Session

from app.models import DQScore, MetricResult, Rule, RuleExecution

DEFAULT_METRIC_WEIGHTS: dict[str, float] = {
    "completeness": 1.0,
    "uniqueness": 1.0,
    "validity": 1.0,
    "consistency": 1.0,
    "referential_integrity": 1.0,
}


def compute_metric_results(
    db: Session,
    version_id: int,
    weights: dict[str, float] | None = None,
) -> list[MetricResult]:
    """Aggregate execution results into per-metric scores."""
    weights = weights or DEFAULT_METRIC_WEIGHTS

    rules = (
        db.query(Rule)
        .filter(
            Rule.version_id == version_id,
            Rule.status == "approved",
        )
        .all()
    )

    rule_ids = [rule.rule_id for rule in rules]

    executions = (
        db.query(RuleExecution)
        .filter(RuleExecution.version_id == version_id)
        .all()
    )

    executions = [e for e in executions if e.rule_id in set(rule_ids)]

    # Remove previous results for this version before recomputation.
    db.query(MetricResult).filter(MetricResult.version_id == version_id).delete()

    metric_results: list[MetricResult] = []
    now = datetime.now(timezone.utc)

    by_metric: dict[str, list[RuleExecution]] = {}

    for execution in executions:
        rule = next(r for r in rules if r.rule_id == execution.rule_id)
        by_metric.setdefault(rule.metric, []).append(execution)

    for metric, metric_executions in by_metric.items():
        # Execution errors are NEVER treated as passing. Errored executions
        # are excluded from the numerator/denominator of the pass rate, and
        # if EVERY execution of the metric errored, the metric is reported
        # with status EXECUTION_ERROR instead of a fake 100%.
        successful = [e for e in metric_executions if e.status != "error"]
        errored = [e for e in metric_executions if e.status == "error"]

        total_applicable = sum(e.applicable_rows for e in successful)
        total_failed = sum(e.failed_rows for e in successful)
        error_count = len(errored)

        if not successful:
            # All executions errored: no evidence of quality at all.
            score = 0.0
            status = "execution_error"
        elif total_applicable == 0:
            score = 0.0
            status = "not_applicable"
        else:
            score = round(
                (total_applicable - total_failed) / total_applicable * 100,
                2,
            )
            status = "available"

        metric_results.append(
            MetricResult(
                version_id=version_id,
                metric=metric,
                score=score,
                applicable_records=total_applicable,
                failed_records=total_failed,
                rule_count=len(metric_executions),
                status=status,
                computed_at=now,
            )
        )

    for result in metric_results:
        db.add(result)

    db.flush()
    return metric_results


def compute_dq_score(
    db: Session,
    version_id: int,
    weighted: bool = False,
    weights: dict[str, float] | None = None,
) -> DQScore:
    """Compute the overall DQ score for a version."""
    metric_results = compute_metric_results(db, version_id, weights)

    now = datetime.now(timezone.utc)

    available = [m for m in metric_results if m.status == "available"]
    not_applicable = [m for m in metric_results if m.status == "not_applicable"]
    errored_metrics = [m for m in metric_results if m.status == "execution_error"]

    # Count every persisted errored execution of approved rules. Execution
    # errors must never be silently absorbed into a 100% score.
    approved_rule_ids = {
        rule_id
        for (rule_id,) in db.query(Rule.rule_id)
        .filter(
            Rule.version_id == version_id,
            Rule.status == "approved",
        )
        .all()
    }
    execution_errors = (
        db.query(RuleExecution)
        .filter(
            RuleExecution.version_id == version_id,
            RuleExecution.status == "error",
            RuleExecution.rule_id.in_(approved_rule_ids),
        )
        .count()
        if approved_rule_ids
        else 0
    )

    if weighted:
        used_weights = {
            m.metric: weights.get(m.metric, 1.0) if weights else DEFAULT_METRIC_WEIGHTS.get(m.metric, 1.0)
            for m in available
        }
        total_weight = sum(used_weights.values())
        if total_weight > 0:
            overall = sum(
                m.score * used_weights[m.metric] for m in available
            ) / total_weight
        else:
            overall = 0.0
    else:
        overall = (
            sum(m.score for m in available) / len(available)
            if available
            else 0.0
        )

    # Failures that must not be hidden by the overall score.
    critical_failures = [
        {
            "metric": m.metric,
            "score": m.score,
            "failed_records": m.failed_records,
            "applicable_records": m.applicable_records,
        }
        for m in available
        if m.score < 90.0
    ]

    details = {
        "metrics": [
            {
                "metric": m.metric,
                "score": m.score,
                "status": m.status,
                "applicable_records": m.applicable_records,
                "failed_records": m.failed_records,
                "rule_count": m.rule_count,
            }
            for m in metric_results
        ],
        "not_applicable_metrics": [m.metric for m in not_applicable],
        "execution_error_metrics": [m.metric for m in errored_metrics],
        "execution_error_count": execution_errors,
        "critical_failures": critical_failures,
    }

    # Column / table / dataset breakdown with N/A-aware aggregation.
    try:
        from app.services.column_scoring import compute_column_breakdown

        details["column_breakdown"] = compute_column_breakdown(db, version_id)
    except Exception as exc:  # pragma: no cover - breakdown must never break scoring
        details["column_breakdown"] = {"error": str(exc)}

    score = DQScore(
        version_id=version_id,
        overall_score=round(overall, 2),
        weighted=1 if weighted else 0,
        metric_count=len(available),
        excluded_metrics=len(not_applicable),
        execution_errors=execution_errors,
        details_json=details,
        computed_at=now,
    )

    db.add(score)
    db.flush()

    return score


def get_latest_score(db: Session, version_id: int) -> DQScore | None:
    return (
        db.query(DQScore)
        .filter(DQScore.version_id == version_id)
        .order_by(DQScore.score_id.desc())
        .first()
    )

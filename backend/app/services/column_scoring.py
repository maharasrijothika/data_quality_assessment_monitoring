"""Column / table / dataset-level DQ scoring (Stage 09).

Aggregation rules (spec 18-23):
  - Rule results are the ONLY source of pass/fail (validation-time stats are
    evidence, never results).
  - Metric score per column = combined applicable/passing population across
    that column's rules of the metric (record-level, no double counting).
  - N/A never becomes 0: metrics with no approved executed rules are "N/A"
    and are excluded from column/dataset denominators.
  - Dataset score = mean of applicable metric scores across tables (equal
    weighting, no silent business weights).
  - Execution errors are surfaced, never absorbed.
"""

from __future__ import annotations

from sqlalchemy.orm import Session

from app.models import MetricResult, Rule, RuleExecution

# The full metric dimension list, in canonical order.
METRIC_ORDER = (
    "completeness",
    "uniqueness",
    "validity",
    "accuracy",
    "consistency",
    "referential_integrity",
    "timeliness",
)


def _execution_rows(execution: RuleExecution) -> dict:
    return {
        "execution_id": execution.execution_id,
        "rule_id": execution.rule_id,
        "status": execution.status,
        "applicable": execution.applicable_rows,
        "passed": execution.passed_rows,
        "failed": execution.failed_rows,
        "total": execution.total_rows,
    }


def _score_from_counts(applicable: int, failed: int) -> float | None:
    """Score = (A - F) / A. A == 0 -> None (N/A), never 0 or 100."""
    if applicable <= 0:
        return None
    return round((applicable - failed) / applicable * 100, 2)


def _latest_execution_per_rule(
    executions: list[RuleExecution],
) -> dict[int, RuleExecution]:
    """One execution per rule: the newest run wins (re-runs supersede)."""
    latest: dict[int, RuleExecution] = {}
    for execution in sorted(executions, key=lambda e: e.execution_id):
        latest[execution.rule_id] = execution
    return latest


def compute_column_breakdown(
    db: Session,
    version_id: int,
) -> dict:
    """Build the column -> metric -> rule breakdown for a version.

    Returns a JSON-serializable dict embedded in DQScore.details_json.
    """
    rules = (
        db.query(Rule)
        .filter(Rule.version_id == version_id, Rule.status == "approved")
        .all()
    )
    rules_by_id = {rule.rule_id: rule for rule in rules}

    executions = (
        db.query(RuleExecution)
        .filter(RuleExecution.version_id == version_id)
        .order_by(RuleExecution.execution_id)
        .all()
    )
    latest_per_rule = _latest_execution_per_rule(executions)

    # table -> column -> metric -> list of (rule, execution)
    structure: dict[str, dict[str, dict[str, list]]] = {}
    for rule in rules:
        rule_json = rule.rule_json or {}
        params = rule_json.get("parameters", {}) or {}
        column = (
            rule_json.get("column")
            or (rule_json.get("columns") or [None])[0]
            or params.get("child_column")
            or params.get("left_column")
            or params.get("determinant_column")
        )
        if not column:
            # Table-level rules (none today) would aggregate at table level.
            column = "(table)"
        structure.setdefault(rule.table_name, {}).setdefault(column, {}).setdefault(
            rule.metric, []
        ).append(rule)

    columns_out: list[dict] = []
    execution_errors: list[dict] = []

    for table_name in sorted(structure):
        for column_name in sorted(structure[table_name]):
            metric_entries: dict[str, dict] = {}
            column_applicable_total = 0
            column_failed_total = 0

            for metric in METRIC_ORDER:
                metric_rules = structure[table_name][column_name].get(metric, [])
                if not metric_rules:
                    metric_entries[metric] = {"status": "N/A", "score": None}
                    continue

                applicable_total = 0
                failed_total = 0
                rule_rows: list[dict] = []
                has_error = False

                for rule in metric_rules:
                    execution = latest_per_rule.get(rule.rule_id)
                    if execution is None:
                        # Approved but not executed: counts as NOT evaluated.
                        rule_rows.append(
                            {
                                "rule_id": rule.rule_id,
                                "rule_code": rule.rule_code,
                                "rule_name": rule.rule_name,
                                "status": "NOT_EXECUTED",
                                "applicable": None,
                                "failed": None,
                                "pass_rate": None,
                            }
                        )
                        has_error = True
                        continue

                    if execution.status == "error":
                        has_error = True
                        rule_rows.append(
                            {
                                "rule_id": rule.rule_id,
                                "rule_code": rule.rule_code,
                                "rule_name": rule.rule_name,
                                "status": "EXECUTION_ERROR",
                                "applicable": None,
                                "failed": None,
                                "pass_rate": None,
                                "error_message": execution.error_message,
                            }
                        )
                        continue

                    applicable_total += execution.applicable_rows
                    failed_total += execution.failed_rows
                    rule_rows.append(
                        {
                            "rule_id": rule.rule_id,
                            "rule_code": rule.rule_code,
                            "rule_name": rule.rule_name,
                            "status": "ok",
                            "applicable": execution.applicable_rows,
                            "failed": execution.failed_rows,
                            "pass_rate": (
                                round(execution.passed_rows / execution.applicable_rows * 100, 2)
                                if execution.applicable_rows
                                else None
                            ),
                        }
                    )

                if has_error and applicable_total == 0:
                    # Every rule errored or unexecuted: metric cannot be scored.
                    metric_entries[metric] = {
                        "status": "EXECUTION_ERROR",
                        "score": None,
                        "rules": rule_rows,
                    }
                elif applicable_total == 0:
                    metric_entries[metric] = {
                        "status": "N/A",
                        "score": None,
                        "rules": rule_rows,
                    }
                else:
                    score = _score_from_counts(applicable_total, failed_total)
                    metric_entries[metric] = {
                        "status": "ok",
                        "score": score,
                        "applicable": applicable_total,
                        "failed": failed_total,
                        "rules": rule_rows,
                    }
                    column_applicable_total += applicable_total
                    column_failed_total += failed_total

            evaluated = [
                m for m, entry in metric_entries.items() if entry["status"] == "ok"
            ]
            if evaluated:
                column_score = sum(metric_entries[m]["score"] for m in evaluated) / len(evaluated)
            else:
                column_score = None

            columns_out.append(
                {
                    "table_name": table_name,
                    "column_name": column_name,
                    "metric_results": metric_entries,
                    "column_score": round(column_score, 2) if column_score is not None else None,
                    "evaluated_metrics": len(evaluated),
                    "applicable_rows_total": column_applicable_total,
                    "failed_rows_total": column_failed_total,
                }
            )

    # ------------------------------------------------------------------
    # Table level: aggregate each metric across its columns (record-level:
    # sums of applicable/failed across rules - rows are never double counted
    # because each rule's applicability is per-row and metrics are separate).
    tables_out: list[dict] = []
    for table_name in sorted({c["table_name"] for c in columns_out}):
        table_columns = [c for c in columns_out if c["table_name"] == table_name]
        table_metric_entries: dict[str, dict] = {}

        for metric in METRIC_ORDER:
            applicable = 0
            failed = 0
            any_error = False
            for column in table_columns:
                entry = column["metric_results"].get(metric, {})
                if entry.get("status") == "ok":
                    applicable += entry.get("applicable", 0)
                    failed += entry.get("failed", 0)
                elif entry.get("status") == "EXECUTION_ERROR":
                    any_error = True

            if applicable > 0:
                table_metric_entries[metric] = {
                    "status": "ok",
                    "score": _score_from_counts(applicable, failed),
                    "applicable": applicable,
                    "failed": failed,
                }
            elif any_error:
                table_metric_entries[metric] = {"status": "EXECUTION_ERROR", "score": None}
            else:
                table_metric_entries[metric] = {"status": "N/A", "score": None}

        ok_metrics = [e for e in table_metric_entries.values() if e["status"] == "ok"]
        table_score = (
            round(sum(e["score"] for e in ok_metrics) / len(ok_metrics), 2)
            if ok_metrics
            else None
        )

        tables_out.append(
            {
                "table_name": table_name,
                "column_count": len(table_columns),
                "metric_results": table_metric_entries,
                "table_score": table_score,
                "evaluated_metrics": len(ok_metrics),
            }
        )

    # ------------------------------------------------------------------
    # Dataset level: mean of table metric scores (equal weights; N/A
    # excluded from both numerator and denominator; N/A != 0).
    dataset_metric_entries: dict[str, dict] = {}
    for metric in METRIC_ORDER:
        applicable = sum(
            entry.get("applicable", 0)
            for table in tables_out
            for entry in [table["metric_results"].get(metric, {})]
            if entry.get("status") == "ok"
        )
        failed = sum(
            entry.get("failed", 0)
            for table in tables_out
            for entry in [table["metric_results"].get(metric, {})]
            if entry.get("status") == "ok"
        )
        any_error = any(
            table["metric_results"].get(metric, {}).get("status") == "EXECUTION_ERROR"
            for table in tables_out
        )
        if applicable > 0:
            dataset_metric_entries[metric] = {
                "status": "ok",
                "score": _score_from_counts(applicable, failed),
                "applicable": applicable,
                "failed": failed,
            }
        elif any_error:
            dataset_metric_entries[metric] = {"status": "EXECUTION_ERROR", "score": None}
        else:
            dataset_metric_entries[metric] = {"status": "N/A", "score": None}

    dataset_ok = [e for e in dataset_metric_entries.values() if e["status"] == "ok"]
    dataset_score = (
        round(sum(e["score"] for e in dataset_ok) / len(dataset_ok), 2)
        if dataset_ok
        else None
    )

    return {
        "columns": columns_out,
        "tables": tables_out,
        "dataset": {
            "metric_results": dataset_metric_entries,
            "overall_score": dataset_score,
            "evaluated_metrics": len(dataset_ok),
        },
    }


def get_column_breakdown(db: Session, version_id: int) -> dict:
    """Convenience wrapper for routers that only need the breakdown."""
    return compute_column_breakdown(db, version_id)

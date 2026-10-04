"""Rule validation (Stage 07).

Validation asks whether a rule CAN be safely executed and whether it makes
sense given the current evidence. It never executes the rule.

Validation-time statistics are EVIDENCE for reviewers only - the authoritative
pass/fail result always comes from Stage 08 execution.

Statuses: VALID / NEEDS_REVIEW / INVALID.
"""

from __future__ import annotations

import re
from sqlalchemy.orm import Session

from app.models import RelationshipCandidate, StoredProfile
from app.services.rule_templates import (
    TEMPLATES,
    legacy_rule_template,
    template_spec,
)

# Operators allowed by constrained comparison/arithmetic templates. No eval,
# no dynamic code - execution dispatches on these fixed tokens.
COMPARISON_OPERATORS = {"<=", ">=", "<", ">", "==", "!="}
ARITHMETIC_OPERATORS = {"*", "+"}

_REFERENCE_SOURCE_TEMPLATE_PREFIXES = (
    "REFERENCE_HIERARCHY",
    "REFERENCE_LOOKUP",
    "EXTERNAL_REFERENCE_VERIFICATION",
)


def _find_column_profile(
    profile_result: dict,
    table_name: str,
    column_name: str,
) -> dict | None:
    for table in profile_result.get("tables", []):
        if table.get("table_name") == table_name:
            for column in table.get("columns", []):
                if column.get("column_name") == column_name:
                    return column
    return None


def _issue(severity: str, message: str) -> dict:
    return {"severity": severity, "message": message}


def _status_from_issues(issues: list[dict]) -> str:
    if any(i["severity"] == "error" for i in issues):
        return "INVALID"
    if any(i["severity"] == "warning" for i in issues):
        return "NEEDS_REVIEW"
    return "VALID"


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _validate_params_against_spec(
    template: str,
    params: dict,
    issues: list[dict],
) -> None:
    spec = TEMPLATES[template]

    provided = set(params.keys())
    allowed = {p.name for p in spec.parameters}

    for unknown in sorted(provided - allowed):
        issues.append(
            _issue("warning", f"Parameter '{unknown}' is not defined by template {template}; it will be ignored.")
        )

    for param in spec.parameters:
        present = param.name in params
        value = params.get(param.name)

        if not present or value is None:
            # number_or_none parameters accept explicit None (open-ended
            # ranges); every other required parameter must have a value.
            if param.required and param.kind != "number_or_none":
                issues.append(
                    _issue("error", f"Missing required parameter '{param.name}' for {template}.")
                )
            continue

        kind = param.kind
        if kind in {"number", "number_or_none"} and not _is_number(value):
            issues.append(_issue("error", f"Parameter '{param.name}' must be a number."))
        elif kind == "boolean" and not isinstance(value, bool):
            issues.append(_issue("error", f"Parameter '{param.name}' must be true/false."))
        elif kind == "string_list":
            if (
                not isinstance(value, list)
                or not value
                or not all(isinstance(v, str) for v in value)
            ):
                issues.append(_issue("error", f"Parameter '{param.name}' must be a non-empty list of strings."))
        elif kind == "regex":
            if not isinstance(value, str) or not value:
                issues.append(_issue("error", f"Parameter '{param.name}' must be a non-empty string."))
            else:
                try:
                    re.compile(value)
                except re.error as exc:
                    issues.append(_issue("error", f"Parameter '{param.name}' is not a valid regular expression: {exc}"))
        elif kind == "interval":
            if (
                not isinstance(value, dict)
                or not _is_number(value.get("amount"))
                or value.get("amount", 0) <= 0
                or value.get("unit") not in {"minutes", "hours", "days", "weeks"}
            ):
                issues.append(
                    _issue("error", f"Parameter '{param.name}' must be {{amount, unit}} with unit in minutes/hours/days/weeks.")
                )
        elif kind == "datetime":
            if not isinstance(value, str) or not value:
                issues.append(_issue("error", f"Parameter '{param.name}' must be an ISO timestamp string."))
        elif kind == "string":
            if not isinstance(value, str) or not value:
                issues.append(_issue("error", f"Parameter '{param.name}' must be a non-empty string."))

    # Cross-parameter checks.
    if template == "NUMERIC_RANGE":
        lo, hi = params.get("min"), params.get("max")
        if _is_number(lo) and _is_number(hi) and lo > hi:
            issues.append(_issue("error", "NUMERIC_RANGE min must not exceed max."))

    if template == "REGEX":
        if not params.get("pattern") and not params.get("pattern_name"):
            issues.append(
                _issue("error", "REGEX requires parameters.pattern or parameters.pattern_name (e.g. email_syntax).")
            )


def _validate_targets(
    db: Session,
    rule_json: dict,
    template: str,
    profile_result: dict,
    issues: list[dict],
) -> None:
    spec = TEMPLATES[template]
    table = rule_json.get("table")
    params = rule_json.get("parameters", {}) or {}

    # Column targets may live in rule_json.column (single), rule_json.columns
    # (list, both new and legacy rules) or template parameters (cross-column
    # templates). Parameters are authoritative for cross-column templates.
    param_columns = [
        params.get(name)
        for name in (
            "left_column", "right_column", "expected_column",
            "determinant_column", "dependent_column",
            "parent_column", "child_column",
        )
        if params.get(name)
    ]

    candidate_columns: list[str] = []
    if rule_json.get("column"):
        candidate_columns.append(rule_json["column"])
    candidate_columns.extend(str(c) for c in (rule_json.get("columns") or []))
    candidate_columns.extend(str(c) for c in param_columns)

    # De-duplicate, preserving order.
    seen_cols: set[str] = set()
    target_columns = [
        c for c in candidate_columns if not (c in seen_cols or seen_cols.add(c))
    ]

    if spec.target in {"column", "columns"} and not table:
        issues.append(_issue("error", "Rule does not reference a table."))
        return

    if spec.target in {"column", "columns"} and not target_columns:
        issues.append(_issue("error", "Rule does not reference any column."))
        return

    if spec.target == "cross_table":
        for required in ("parent_table", "parent_column", "child_table", "child_column"):
            if not params.get(required):
                issues.append(_issue("error", f"FOREIGN_KEY_EXISTS requires parameter '{required}'."))
        return

    for column in target_columns:
        if _find_column_profile(profile_result, table, column) is None:
            issues.append(_issue("error", f"Column {column} not found in table {table}."))
        else:
            if template in {"UNIQUE", "COMPOSITE_UNIQUE"}:
                column_profile = _find_column_profile(profile_result, table, column)
                distinct_pct = column_profile.get("distinct_percentage", 0.0)
                if distinct_pct < 95.0:
                    issues.append(
                        _issue(
                            "warning",
                            f"Column {column} has observed distinct ratio {distinct_pct:.1f}% - uniqueness may fail.",
                        )
                    )

            if template == "NUMERIC_TYPE" or template == "NUMERIC_RANGE":
                column_profile = _find_column_profile(profile_result, table, column)
                if column_profile and not column_profile.get("numeric"):
                    text_numeric = (column_profile.get("text", {}) or {}).get("patterns", {}).get("numeric_like", 0)
                    if not text_numeric:
                        issues.append(
                            _issue(
                                "warning",
                                f"Column {column} shows no numeric evidence in the profile; expect broad failures.",
                            )
                        )


def _validate_evidence_requirements(
    db: Session,
    dataset_id: int,
    version_id: int,
    rule_json: dict,
    template: str,
    issues: list[dict],
) -> None:
    spec = TEMPLATES[template]
    params = rule_json.get("parameters", {}) or {}

    if "approved_relationship" in spec.requires:
        parent_table = params.get("parent_table")
        parent_column = params.get("parent_column")
        child_table = params.get("child_table")
        child_column = params.get("child_column")

        approved = (
            db.query(RelationshipCandidate)
            .filter(
                RelationshipCandidate.version_id == version_id,
                RelationshipCandidate.candidate_kind == "relationship",
                RelationshipCandidate.status.in_({"approved", "edited", "manual"}),
                RelationshipCandidate.parent_table == parent_table,
                RelationshipCandidate.parent_column == parent_column,
                RelationshipCandidate.child_table == child_table,
                RelationshipCandidate.child_column == child_column,
            )
            .first()
        )

        if approved is None:
            issues.append(
                _issue(
                    "error",
                    "No human-approved relationship matches this RI rule. Approve the relationship in Stage 05 first "
                    "(RI is never generated from name similarity alone).",
                )
            )

    if "user_sla" in spec.requires and template in {"FRESHNESS_THRESHOLD", "DEADLINE"}:
        sla_present = bool(params.get("max_age")) or bool(params.get("deadline")) or bool(params.get("time_of_day"))
        if not sla_present:
            issues.append(
                _issue("error", "Timeliness rules require a user-supplied SLA (threshold or deadline); it cannot be invented.")
            )

    if "reference_source" in spec.requires:
        # A reference source is user-configured; until the user supplies one
        # (via parameters.reference_name/provider or dataset context) the
        # rule stays unexecutable.
        reference_named = bool(params.get("hierarchy")) or bool(params.get("reference_name")) or bool(params.get("provider"))
        if not reference_named:
            issues.append(
                _issue(
                    "error",
                    f"{template} requires a reference source (hierarchy values, reference dataset or provider) configured by the user.",
                )
            )


def validate_rule_dict(
    db: Session,
    dataset_id: int,
    version_id: int,
    rule_json: dict,
) -> dict:
    """Validate one structured rule definition against registry + evidence."""
    issues: list[dict] = []

    template = rule_json.get("rule_template") or rule_json.get("template")
    rule_type = rule_json.get("type")

    # Legacy DSL adapter: old rules (type=...) map onto registry templates.
    if not template and rule_type:
        mapped = legacy_rule_template(rule_json)
        if mapped is None:
            return {
                "status": "INVALID",
                "issues": [_issue("error", f"Unsupported rule type: {rule_type}")],
            }
        template = mapped

    if template == "NOT_WHITESPACE_COMBINED":
        # Legacy combined completeness (not_null + empty + whitespace flags).
        # Keep validating the legacy shape directly.
        column = rule_json.get("column")
        if not rule_json.get("table"):
            issues.append(_issue("error", "Rule does not reference a table."))
        if not column:
            issues.append(_issue("error", "Rule does not reference any column."))
        status = _status_from_issues(issues)
        return {"status": status, "issues": issues}

    if not template or template not in TEMPLATES:
        return {
            "status": "INVALID",
            "issues": [_issue("error", f"Unknown rule template: {template}")],
        }

    # Structural + parameter validation against the registry spec.
    _validate_params_against_spec(template, rule_json.get("parameters", {}) or {}, issues)
    _validate_evidence_requirements(db, dataset_id, version_id, rule_json, template, issues)

    stored_profile = (
        db.query(StoredProfile)
        .filter(StoredProfile.version_id == version_id)
        .first()
    )
    profile_result = stored_profile.profile_json if stored_profile else {}

    if not profile_result:
        issues.append(_issue("warning", "No stored profile available for validation."))
        status = _status_from_issues(issues)
        return {"status": status, "issues": issues}

    _validate_targets(db, rule_json, template, profile_result, issues)

    status = _status_from_issues(issues)
    return {"status": status, "issues": issues}


def validate_rule_dict_full(
    db: Session,
    dataset_id: int,
    version_id: int,
    rule_json: dict,
) -> dict:
    """Backwards-compatible alias."""
    return validate_rule_dict(db, dataset_id, version_id, rule_json)

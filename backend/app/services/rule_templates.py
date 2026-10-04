"""Controlled rule template registry (Stage 06+).

Every rule in the system is an instance of one template from this registry.
Templates define:
  - which metric they belong to
  - which parameters they require/accept (name, type, required/optional)
  - whether they need extra evidence (approved relationship, reference
    source, user SLA) to be executable

The registry is intentionally constrained and explainable. There is no
arbitrary-code template and there never will be one: execution dispatches to
one handler per template (see rule_execution.py).

Legacy rules (type=completeness/uniqueness/validity + check=...) created
before this registry are supported through ``legacy_rule_template`` which
maps them onto registry template names.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

METRICS = (
    "completeness",
    "uniqueness",
    "validity",
    "accuracy",
    "consistency",
    "referential_integrity",
    "timeliness",
)

METRIC_LABELS = {
    "completeness": "Completeness",
    "uniqueness": "Uniqueness",
    "validity": "Validity",
    "accuracy": "Accuracy",
    "consistency": "Consistency",
    "referential_integrity": "Referential Integrity",
    "timeliness": "Timeliness",
}


@dataclass(frozen=True)
class ParamSpec:
    """One structured rule parameter."""

    name: str
    kind: str  # "string" | "number" | "number_or_none" | "string_list" | "regex" | "boolean" | "threshold" | "interval" | "datetime"
    required: bool = True


@dataclass(frozen=True)
class TemplateSpec:
    """One controlled rule template."""

    name: str
    metric: str
    target: str  # "column" | "columns" | "table" | "cross_table"
    parameters: tuple[ParamSpec, ...] = field(default_factory=tuple)
    # Extra evidence the rule REQUIRES before it may execute:
    #   approved_relationship - needs a human-approved relationship
    #   reference_source      - needs a reference/authoritative source
    #   user_sla              - needs a user-supplied threshold/deadline
    requires: tuple[str, ...] = field(default_factory=tuple)
    description: str = ""


TEMPLATES: dict[str, TemplateSpec] = {
    # Completeness -----------------------------------------------------------
    "NOT_NULL": TemplateSpec(
        "NOT_NULL",
        "completeness",
        "column",
        description="Values must not be NULL.",
    ),
    "NOT_EMPTY": TemplateSpec(
        "NOT_EMPTY",
        "completeness",
        "column",
        description="Values must not be empty strings.",
    ),
    "NOT_WHITESPACE": TemplateSpec(
        "NOT_WHITESPACE",
        "completeness",
        "column",
        description="Values must not be whitespace-only strings.",
    ),
    # Uniqueness -------------------------------------------------------------
    "UNIQUE": TemplateSpec(
        "UNIQUE",
        "uniqueness",
        "column",
        (
            ParamSpec("ignore_nulls", "boolean", required=False),
            ParamSpec("normalize_whitespace", "boolean", required=False),
        ),
        description="Values must be unique within the column.",
    ),
    "COMPOSITE_UNIQUE": TemplateSpec(
        "COMPOSITE_UNIQUE",
        "uniqueness",
        "columns",
        (
            ParamSpec("ignore_nulls", "boolean", required=False),
            ParamSpec("normalize_whitespace", "boolean", required=False),
        ),
        description="The combination of values must be unique across rows.",
    ),
    # Validity ---------------------------------------------------------------
    "REGEX": TemplateSpec(
        "REGEX",
        "validity",
        "column",
        (
            ParamSpec("pattern", "regex", required=False),
            # Named fixed patterns (email_syntax) avoid hand-written regex in
            # rule definitions; exactly one of pattern/pattern_name is used.
            ParamSpec("pattern_name", "string", required=False),
        ),
        description="Values must match a fixed regular expression or a named pattern (e.g. email_syntax).",
    ),
    "ALLOWED_VALUES": TemplateSpec(
        "ALLOWED_VALUES",
        "validity",
        "column",
        (ParamSpec("allowed_values", "string_list"),),
        description="Values must be one of a controlled value list.",
    ),
    "NUMERIC_TYPE": TemplateSpec(
        "NUMERIC_TYPE",
        "validity",
        "column",
        description="Values must be numeric (parseable as a number).",
    ),
    "NUMERIC_RANGE": TemplateSpec(
        "NUMERIC_RANGE",
        "validity",
        "column",
        (
            ParamSpec("min", "number_or_none"),
            ParamSpec("max", "number_or_none"),
        ),
        description="Values must lie inside an explicit business range.",
    ),
    "DATE_FORMAT": TemplateSpec(
        "DATE_FORMAT",
        "validity",
        "column",
        (ParamSpec("date_format", "string", required=False),),
        description="Values must be parseable dates (optionally one exact format).",
    ),
    "DATATYPE_COMPATIBILITY": TemplateSpec(
        "DATATYPE_COMPATIBILITY",
        "validity",
        "column",
        (ParamSpec("expected_datatype", "string"),),
        description="Stored values must be representable as the expected datatype.",
    ),
    # Consistency ------------------------------------------------------------
    "COLUMN_COMPARISON": TemplateSpec(
        "COLUMN_COMPARISON",
        "consistency",
        "columns",
        (
            ParamSpec("left_column", "string"),
            ParamSpec("operator", "string"),  # <= < == != >= >
            ParamSpec("right_column", "string"),
        ),
        description="Two columns of the same row must satisfy a comparison.",
    ),
    "DATE_ORDER": TemplateSpec(
        "DATE_ORDER",
        "consistency",
        "columns",
        (
            ParamSpec("left_column", "string"),
            ParamSpec("right_column", "string"),
            ParamSpec("allow_equal", "boolean", required=False),
        ),
        description="One date must precede (or equal) another on the same row.",
    ),
    "ARITHMETIC_RELATION": TemplateSpec(
        "ARITHMETIC_RELATION",
        "consistency",
        "columns",
        (
            ParamSpec("left_column", "string"),
            ParamSpec("operator", "string"),  # "*" | "+"
            ParamSpec("right_column", "string"),
            ParamSpec("expected_column", "string"),
            ParamSpec("tolerance", "number", required=False),
        ),
        description="A computed relation between columns must equal a result column within tolerance.",
    ),
    "FUNCTIONAL_DEPENDENCY": TemplateSpec(
        "FUNCTIONAL_DEPENDENCY",
        "consistency",
        "columns",
        (
            ParamSpec("determinant_column", "string"),
            ParamSpec("dependent_column", "string"),
        ),
        description="A determinant column must uniquely determine a dependent column.",
    ),
    "REFERENCE_HIERARCHY": TemplateSpec(
        "REFERENCE_HIERARCHY",
        "consistency",
        "columns",
        (
            ParamSpec("parent_column", "string"),
            ParamSpec("child_column", "string"),
            ParamSpec("hierarchy", "string_list"),
        ),
        requires=("reference_source",),
        description="Child values must be consistent with parent values per a reference hierarchy.",
    ),
    # Referential integrity ---------------------------------------------------
    "FOREIGN_KEY_EXISTS": TemplateSpec(
        "FOREIGN_KEY_EXISTS",
        "referential_integrity",
        "cross_table",
        (
            ParamSpec("parent_table", "string"),
            ParamSpec("parent_column", "string"),
            ParamSpec("child_table", "string"),
            ParamSpec("child_column", "string"),
        ),
        requires=("approved_relationship",),
        description="Child values must exist in the referenced parent column.",
    ),
    # Timeliness ---------------------------------------------------------------
    "FRESHNESS_THRESHOLD": TemplateSpec(
        "FRESHNESS_THRESHOLD",
        "timeliness",
        "column",
        (ParamSpec("max_age", "interval"),),
        requires=("user_sla",),
        description="Age since the recorded timestamp must stay within a user SLA.",
    ),
    "DEADLINE": TemplateSpec(
        "DEADLINE",
        "timeliness",
        "column",
        (
            ParamSpec("deadline", "datetime", required=False),
            ParamSpec("time_of_day", "string", required=False),
            ParamSpec("recurring", "boolean", required=False),
        ),
        requires=("user_sla",),
        description="Timestamps must meet a fixed deadline or a recurring daily time-of-day SLA.",
    ),
    # Accuracy -----------------------------------------------------------------
    "REFERENCE_LOOKUP": TemplateSpec(
        "REFERENCE_LOOKUP",
        "accuracy",
        "column",
        (
            ParamSpec("reference_name", "string"),
            ParamSpec("match_mode", "string", required=False),
        ),
        requires=("reference_source",),
        description="Values must exist in an authoritative reference dataset.",
    ),
    "EXTERNAL_REFERENCE_VERIFICATION": TemplateSpec(
        "EXTERNAL_REFERENCE_VERIFICATION",
        "accuracy",
        "column",
        (
            ParamSpec("provider", "string"),
            ParamSpec("match_mode", "string", required=False),
        ),
        requires=("reference_source",),
        description="Values are verified against an external provider (cachable, rate-limited).",
    ),
}

TEMPLATES_BY_METRIC: dict[str, list[str]] = {}
for _name, _spec in TEMPLATES.items():
    TEMPLATES_BY_METRIC.setdefault(_spec.metric, []).append(_name)


def template_spec(name: str) -> TemplateSpec | None:
    return TEMPLATES.get(name)


def templates_for_metric(metric: str) -> list[str]:
    return list(TEMPLATES_BY_METRIC.get(metric, []))


# ---------------------------------------------------------------------------
# Legacy DSL adapter (rules created before the template registry)
# ---------------------------------------------------------------------------


def legacy_rule_template(rule_json: dict) -> str | None:
    """Map a legacy rule_json onto a registry template name, or None."""

    rule_type = rule_json.get("type")

    if rule_type == "completeness":
        treat_empty = bool(rule_json.get("treat_empty_string_as_null"))
        treat_ws = bool(rule_json.get("treat_whitespace_as_null"))
        if treat_empty and treat_ws:
            return "NOT_WHITESPACE_COMBINED"
        if rule_json.get("condition") in {None, "not_null"}:
            return "NOT_NULL"
        return None

    if rule_type == "uniqueness":
        columns = rule_json.get("columns")
        if columns and len(columns) > 1:
            return "COMPOSITE_UNIQUE"
        return "UNIQUE"

    if rule_type == "validity":
        check = rule_json.get("check")
        return {
            "email_syntax": "REGEX",
            "numeric_type": "NUMERIC_TYPE",
            "date_parseable": "DATE_FORMAT",
        }.get(check)

    return None


# Legacy email check compiled pattern lives in rule_execution; this exposes
# the fixed pattern name used by the legacy adapter.
LEGACY_EMAIL_PATTERN_NAME = "email_syntax"

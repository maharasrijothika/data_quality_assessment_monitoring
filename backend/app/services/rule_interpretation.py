"""Business-rule text interpretation (Stage 06+).

Layer 1 of the layered interpretation approach: controlled keyword/template
matching over a FIXED pattern list. Deterministic and explainable.

Hard rules:
  - The user text is NEVER executed. It is only mapped to
    (metric, template, parameters) and shown back for approval.
  - If no pattern matches with high enough confidence, the result is
    UNRESOLVED with metric_options - the user must choose. The system never
    silently guesses a metric.
  - Timeliness SLAs ("within 24 hours", "daily by 18:00") are normalized
    HERE, at interpretation time, into structured parameters - not re-parsed
    at every execution.
"""

from __future__ import annotations

import re
from datetime import datetime

from app.services.rule_templates import METRICS, METRIC_LABELS

UNRESOLVED = "UNRESOLVED"
RESOLVED = "RESOLVED"

# Metrics a user may pick when interpretation fails.
METRIC_OPTIONS = list(METRICS)


class InterpretationError(ValueError):
    pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NUMBER = r"-?\d+(?:\.\d+)?"

INTERVAL_UNITS = {
    "minute": "minutes",
    "minutes": "minutes",
    "min": "minutes",
    "mins": "minutes",
    "hour": "hours",
    "hours": "hours",
    "hr": "hours",
    "hrs": "hours",
    "day": "days",
    "days": "days",
    "week": "weeks",
    "weeks": "weeks",
}


def _parse_number(text: str) -> float:
    return float(text.replace(",", ""))


def _normalize_interval(text: str) -> dict | None:
    """'24 hours' -> {'amount': 24, 'unit': 'hours'}; None if no match."""
    match = re.search(
        rf"({_NUMBER})\s*({('|'.join(INTERVAL_UNITS))})\b",
        text,
        re.IGNORECASE,
    )
    if not match:
        return None
    unit = INTERVAL_UNITS[match.group(2).lower()]
    return {"amount": _parse_number(match.group(1)), "unit": unit}


def _normalize_deadline(text: str) -> dict | None:
    """'by 2026-09-30 18:00' / 'before 2026-09-30' -> normalized timestamp."""
    patterns = [
        (r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}(?::\d{2})?", "%Y-%m-%d %H:%M"),
        (r"\d{4}-\d{2}-\d{2}", "%Y-%m-%d"),
        (r"\d{2}/\d{2}/\d{4} \d{2}:\d{2}", "%d/%m/%Y %H:%M"),
        (r"\d{2}/\d{2}/\d{4}", "%d/%m/%Y"),
    ]
    for pattern, fmt in patterns:
        match = re.search(pattern, text)
        if match:
            try:
                parsed = datetime.strptime(match.group(0).replace("T", " "), fmt)
            except ValueError:
                continue
            iso = parsed.isoformat()
            # A date-only deadline means end-of-day.
            if fmt == "%Y-%m-%d":
                iso = parsed.replace(hour=23, minute=59).isoformat()
            elif fmt == "%d/%m/%Y":
                iso = parsed.replace(hour=23, minute=59).isoformat()
            return {"deadline": iso, "raw": match.group(0)}
    return None


# ---------------------------------------------------------------------------
# Layer 1 pattern table: (regex, metric, template, parameter extractor)
# Each entry documents what it matches; no pattern may claim a metric it
# cannot parameterize. Order matters: more specific patterns first.
# ---------------------------------------------------------------------------

def _p_numeric_range(text: str) -> dict | None:
    m = re.search(
        rf"between\s+({_NUMBER})\s+and\s+({_NUMBER})",
        text,
        re.IGNORECASE,
    )
    if not m:
        return None
    lo, hi = _parse_number(m.group(1)), _parse_number(m.group(2))
    if lo > hi:
        lo, hi = hi, lo
    return {"min": lo, "max": hi}


def _p_email(text: str) -> dict | None:
    if re.search(r"e-?mail\s+(format|address|syntax)|valid\s+e-?mail", text, re.IGNORECASE):
        return {"pattern_name": "email_syntax"}
    return None


def _p_allowed_values(text: str) -> dict | None:
    m = re.search(
        r"(?:must be|should be|can only be|only)\s+(?:one of\s+)?[\(\[\{]?([^\)\]\}]+)[\)\]\}]?\s*$",
        text,
        re.IGNORECASE,
    )
    if not m:
        return None
    raw = m.group(1)
    if not any(sep in raw for sep in (",", "|", "/")):
        return None
    values = [v.strip().strip("'\"") for v in re.split(r"[|/,]", raw) if v.strip()]
    if len(values) < 2:
        return None
    return {"allowed_values": values}


def _p_regex_literal(text: str) -> dict | None:
    m = re.search(r"(?:must\s+)?match(?:es)?\s+(?:the\s+)?(?:pattern|regex)\s+/?([^/]+?)/?\s*$", text, re.IGNORECASE)
    if not m:
        return None
    return {"pattern": m.group(1).strip()}


def _p_date_order(text: str) -> dict | None:
    m = re.search(
        r"([a-z0-9_ ]+?)\s+must\s+(?:be\s+)?(?:before|earlier than|precede)\s+([a-z0-9_ ]+)",
        text,
        re.IGNORECASE,
    )
    if not m:
        return None
    left = m.group(1).strip().removeprefix("the ").strip()
    right = m.group(2).strip().removeprefix("the ").strip().removesuffix(".").strip()
    if not left or not right:
        return None
    allow_equal = bool(re.search(r"on or before|no later than", text, re.IGNORECASE))
    return {"left_column": left, "right_column": right, "allow_equal": allow_equal}


def _p_column_comparison(text: str) -> dict | None:
    m = re.search(
        r"([a-z0-9_ ]+?)\s+(?:must|should)\s+(?:be\s+)?(<=|>=|<|>|==|!=|less than or equal to|greater than or equal to|less than|greater than|equal to|at most|at least)\s+([a-z0-9_ ]+)",
        text,
        re.IGNORECASE,
    )
    if not m:
        return None
    ops = {
        "less than or equal to": "<=",
        "at most": "<=",
        "greater than or equal to": ">=",
        "at least": ">=",
        "less than": "<",
        "greater than": ">",
        "equal to": "==",
    }
    raw_op = m.group(2).lower()
    operator = raw_op if raw_op in {"<=", ">=", "<", ">", "==", "!="} else ops[raw_op]
    left = m.group(1).strip().removeprefix("the ").strip()
    right = m.group(3).strip().removeprefix("the ").strip().removesuffix(".").strip()
    # A comparison against a literal number is a NUMERIC_RANGE-style bound on
    # one column, not a cross-column rule; the caller checks numeric-ness.
    right_is_number = bool(re.fullmatch(_NUMBER, right))
    return {
        "left_column": left,
        "operator": operator,
        "right_column": right,
        "right_is_number": right_is_number,
        "right_number": _parse_number(right) if right_is_number else None,
    }


def _p_arithmetic(text: str) -> dict | None:
    m = re.search(
        r"([a-z0-9_ ]+?)\s*(\*|x|times|\+)\s*([a-z0-9_ ]+?)\s+(?:must\s+)?(?:be\s+)?(?:equal to|approximately|≈|~)\s+([a-z0-9_ ]+)",
        text,
        re.IGNORECASE,
    )
    if not m:
        return None
    op = "+" if m.group(2) == "+" else "*"
    left = m.group(1).strip().removeprefix("the ").strip()
    right = m.group(3).strip()
    expected = m.group(4).strip().removeprefix("the ").strip().removesuffix(".").strip()
    tol = re.search(r"within\s+({_NUMBER})\s*%".replace("{_NUMBER}", _NUMBER), text, re.IGNORECASE)
    return {
        "left_column": left,
        "operator": op,
        "right_column": right,
        "expected_column": expected,
        "tolerance": _parse_number(tol.group(1)) if tol else 0.0,
    }


def _p_fk(text: str) -> dict | None:
    m = re.search(
        r"([a-z0-9_. ]+?)\s+must\s+(?:exist|be present)\s+in\s+([a-z0-9_. ]+)",
        text,
        re.IGNORECASE,
    )
    if not m:
        return None

    def split_qualified(token: str) -> tuple[str, str | None]:
        token = token.strip().removeprefix("the ").strip()
        if "." in token:
            table, column = token.split(".", 1)
            return table.strip(), column.strip()
        return token, None

    child_raw, parent_raw = m.group(1), m.group(2)
    child_table, child_column = split_qualified(child_raw)
    parent_table, parent_column = split_qualified(parent_raw)
    return {
        "child_table": child_table,
        "child_column": child_column,
        "parent_table": parent_table,
        "parent_column": parent_column,
        "needs_target_confirmation": parent_column is None,
    }


def _p_functional_dependency(text: str) -> dict | None:
    m = re.search(
        r"([a-z0-9_ ]+?)\s+(?:determines|uniquely determines|implies)\s+([a-z0-9_ ]+)",
        text,
        re.IGNORECASE,
    )
    if not m:
        return None
    return {
        "determinant_column": m.group(1).strip().removeprefix("the ").strip(),
        "dependent_column": m.group(2).strip().removeprefix("the ").strip().removesuffix(".").strip(),
    }


def _p_freshness(text: str) -> dict | None:
    if not re.search(r"(updated|refreshed|loaded|received)\s+within|within\s+\d+\s*(hour|day|week|minute)", text, re.IGNORECASE):
        return None
    interval = _normalize_interval(text)
    if not interval:
        return None
    return {"max_age": interval}


def _p_deadline(text: str) -> dict | None:
    if not re.search(r"by\s+\d{4}-\d{2}-\d{2}|by\s+\d{2}/\d{2}/\d{4}|daily\s+by\s+\d{1,2}:\d{2}|by\s+\d{1,2}:\d{2}$", text, re.IGNORECASE):
        return None

    # "Daily by 18:00" - a recurring time-of-day SLA.
    daily = re.search(r"daily\s+by\s+(\d{1,2}:\d{2})", text, re.IGNORECASE)
    if daily:
        return {"time_of_day": daily.group(1), "recurring": True}

    normalized = _normalize_deadline(text)
    if not normalized:
        # "by 18:00" without a date: recurring time-of-day without 'daily'.
        tod = re.search(r"by\s+(\d{1,2}:\d{2})\s*$", text, re.IGNORECASE)
        if tod:
            return {"time_of_day": tod.group(1), "recurring": True}
        return None
    return {"deadline": normalized["deadline"], "recurring": False}


def _p_unique(text: str) -> dict | None:
    m = re.search(r"([a-z0-9_,\s]+?)\s+must\s+be\s+unique", text, re.IGNORECASE)
    if not m:
        return None
    columns = [c.strip() for c in m.group(1).replace("the ", "").split(",") if c.strip()]
    return {"columns": columns}


def _p_date_format(text: str) -> dict | None:
    if not re.search(r"(?:date|timestamp).*(?:format)|format\s+\w[\w%-]*", text, re.IGNORECASE):
        return None
    if re.search(r"valid date|be a (?:valid )?date", text, re.IGNORECASE):
        return {"date_format": None}
    m = re.search(r"format\s+([\w%/:-]+)", text, re.IGNORECASE)
    if m:
        return {"date_format": m.group(1)}
    return None


def _p_numeric_type(text: str) -> dict | None:
    if re.search(r"must\s+be\s+(?:a\s+)?number|numeric(?:\s+value)?s?\s*$", text, re.IGNORECASE):
        return {}
    return None


def _p_not_null(text: str) -> dict | None:
    if re.search(r"must\s+not\s+be\s+null|cannot\s+be\s+null|not\s+null|non-null", text, re.IGNORECASE):
        return {}
    return None


def _p_not_empty(text: str) -> dict | None:
    if re.search(r"must\s+not\s+be\s+empty|cannot\s+be\s+empty|non-empty|not\s+be\s+blank", text, re.IGNORECASE):
        return {}
    return None


def _p_not_whitespace(text: str) -> dict | None:
    if re.search(r"(?:must\s+not\s+be\s+)?(?:whitespace|spaces)\s*-?\s*only|only\s+whitespace", text, re.IGNORECASE):
        return {}
    return None


# (regex, metric, template, extractor) — specific first.
PATTERN_TABLE: list[tuple[str, str, str, object]] = [
    (r".", "completeness", "NOT_WHITESPACE", _p_not_whitespace),
    (r".", "completeness", "NOT_EMPTY", _p_not_empty),
    (r".", "completeness", "NOT_NULL", _p_not_null),
    (r".", "uniqueness", "UNIQUE", _p_unique),
    (r".", "validity", "NUMERIC_RANGE", _p_numeric_range),
    (r".", "validity", "REGEX", _p_email),
    (r".", "validity", "REGEX", _p_regex_literal),
    (r".", "validity", "ALLOWED_VALUES", _p_allowed_values),
    (r".", "validity", "DATE_FORMAT", _p_date_format),
    (r".", "validity", "NUMERIC_TYPE", _p_numeric_type),
    (r".", "consistency", "ARITHMETIC_RELATION", _p_arithmetic),
    (r".", "consistency", "DATE_ORDER", _p_date_order),
    (r".", "consistency", "COLUMN_COMPARISON", _p_column_comparison),
    (r".", "consistency", "FUNCTIONAL_DEPENDENCY", _p_functional_dependency),
    (r".", "referential_integrity", "FOREIGN_KEY_EXISTS", _p_fk),
    (r".", "timeliness", "FRESHNESS_THRESHOLD", _p_freshness),
    (r".", "timeliness", "DEADLINE", _p_deadline),
]


def interpret_business_rule(
    text: str,
    *,
    default_table: str | None = None,
    default_column: str | None = None,
) -> dict:
    """Interpret one business-rule sentence into a structured candidate.

    Returns:
        {
          "status": "RESOLVED" | "UNRESOLVED",
          "metric", "rule_template", "parameters",
          "target": {"table", "columns"},
          "notes": [...],
          "metric_options": [...]      # only when UNRESOLVED
          "original_text": text
        }

    Never raises for unmatched text - ambiguity is a normal, expected outcome
    and MUST be shown to the user as a question, not guessed.
    """
    cleaned = (text or "").strip()
    if not cleaned:
        return {
            "status": UNRESOLVED,
            "metric": None,
            "rule_template": None,
            "parameters": {},
            "target": {"table": default_table, "columns": [default_column] if default_column else []},
            "notes": ["Empty rule text."],
            "metric_options": METRIC_OPTIONS,
            "original_text": text,
        }

    notes: list[str] = []

    for _regex, metric, template, extractor in PATTERN_TABLE:
        extracted = extractor(cleaned)
        if extracted is None:
            continue

        # Cross-column consistency: comparing against a literal number is
        # actually a one-sided numeric bound on a single column.
        if template in {"COLUMN_COMPARISON", "DATE_ORDER"}:
            right = extracted.get("right_column", "")
            if template == "COLUMN_COMPARISON" and extracted.get("right_is_number"):
                template = "NUMERIC_RANGE"
                metric = "validity"
                operator = extracted["operator"]
                extracted = {
                    "min": extracted["right_number"] if operator in {">", ">="} else None,
                    "max": extracted["right_number"] if operator in {"<", "<="} else None,
                }
            elif template == "DATE_ORDER" and not _looks_like_column(right):
                notes.append(
                    f"'{right}' does not look like a column name; pass the target columns explicitly."
                )
                extracted = dict(extracted)
                extracted["right_column"] = None

        # FK interpretation against a master-data phrase.
        if template == "FOREIGN_KEY_EXISTS" and extracted.get("needs_target_confirmation"):
            notes.append(
                "Parent column not fully specified; confirm the parent table.column in the review form."
            )

        columns = _target_columns(extracted, template, default_column)
        table = default_table

        return {
            "status": RESOLVED,
            "metric": metric,
            "rule_template": template,
            "parameters": {k: v for k, v in extracted.items() if k in _allowed_params(template)},
            "target": {"table": table, "columns": columns},
            "notes": notes,
            "original_text": cleaned,
            "metric_label": METRIC_LABELS.get(metric, metric),
        }

    # Nothing matched safely: ask the user. NEVER guess.
    return {
        "status": UNRESOLVED,
        "metric": None,
        "rule_template": None,
        "parameters": {},
        "target": {"table": default_table, "columns": [default_column] if default_column else []},
        "notes": [
            "Unable to determine the metric safely from this text. Choose a metric and fill the rule form explicitly - nothing was guessed."
        ],
        "metric_options": METRIC_OPTIONS,
        "original_text": cleaned,
    }


def _looks_like_column(token: str) -> bool:
    return bool(re.fullmatch(r"[a-z][a-z0-9_]*", token.strip().replace(" ", "_"), re.IGNORECASE))


def _target_columns(extracted: dict, template: str, default_column: str | None) -> list[str]:
    if template == "UNIQUE":
        return extracted.get("columns", [default_column] if default_column else [])
    if template in {"FOREIGN_KEY_EXISTS"}:
        return [c for c in [extracted.get("child_column")] if c]
    if template in {"DATE_ORDER", "COLUMN_COMPARISON", "ARITHMETIC_RELATION"}:
        cols = [
            extracted.get("left_column"),
            extracted.get("right_column") if template != "ARITHMETIC_RELATION" else extracted.get("expected_column"),
        ]
        if template == "ARITHMETIC_RELATION":
            cols = [extracted.get("left_column"), extracted.get("right_column"), extracted.get("expected_column")]
        return [c for c in cols if c]
    if template == "FUNCTIONAL_DEPENDENCY":
        return [c for c in [extracted.get("determinant_column"), extracted.get("dependent_column")] if c]
    return [default_column] if default_column else []


# Parameters each template accepts in interpreted rules (validation re-checks
# against the registry; this trims interpreter-internal keys like flags).
_ALLOWED_PARAM_NAMES = {
    "NOT_NULL": set(),
    "NOT_EMPTY": set(),
    "NOT_WHITESPACE": set(),
    "UNIQUE": {"ignore_nulls", "normalize_whitespace"},
    "COMPOSITE_UNIQUE": {"ignore_nulls", "normalize_whitespace"},
    "NUMERIC_RANGE": {"min", "max"},
    "REGEX": {"pattern", "pattern_name"},
    "ALLOWED_VALUES": {"allowed_values"},
    "DATE_FORMAT": {"date_format"},
    "NUMERIC_TYPE": set(),
    "COLUMN_COMPARISON": {"left_column", "operator", "right_column"},
    "DATE_ORDER": {"left_column", "right_column", "allow_equal"},
    "ARITHMETIC_RELATION": {"left_column", "operator", "right_column", "expected_column", "tolerance"},
    "FUNCTIONAL_DEPENDENCY": {"determinant_column", "dependent_column"},
    "FOREIGN_KEY_EXISTS": {"parent_table", "parent_column", "child_table", "child_column"},
    "REFERENCE_HIERARCHY": {"parent_column", "child_column", "hierarchy"},
    "FRESHNESS_THRESHOLD": {"max_age"},
    "DEADLINE": {"deadline", "time_of_day", "recurring"},
}


def _allowed_params(template: str) -> set[str]:
    return _ALLOWED_PARAM_NAMES.get(template, set())

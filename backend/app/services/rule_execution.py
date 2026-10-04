"""Stage 08: safe rule execution.

Rules are expressed in a constrained structured form and interpreted by one
fixed handler per template - never eval'd or executed as code.

Only approved rules can run (enforced in execute_rule).

Result semantics (spec section 16):
  N = total rows, A = applicable rows, P = passed, F = failed (A = P + F)
  Coverage = A / N, Pass rate = P / A, Violation rate = F / A
  A == 0  ->  pass_rate and violation_rate are None (N/A), never 100%.
  Execution errors raise RuleExecutionError and are persisted as status=
  "error" by the caller - never treated as passes.
"""

from __future__ import annotations

import re
from datetime import date, datetime, timedelta, timezone

import pandas as pd
from sqlalchemy.orm import Session

from app.models import Rule, RuleExecution
from app.services import storage as storage_module
from app.services.storage import RAW_DATA_DIR

EMAIL_PATTERN = re.compile(
    r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$"
)

MAX_EVIDENCE_EXAMPLES = 20


class RuleExecutionError(Exception):
    pass


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _load_table(
    dataset_id: int,
    version_number: int,
    stored_filename: str,
) -> pd.DataFrame:
    path = (
        storage_module.RAW_DATA_DIR
        / str(dataset_id)
        / f"v{version_number}"
        / stored_filename
    )

    if not path.exists():
        raise RuleExecutionError(f"Raw file missing: {path}")

    if path.suffix.lower() in {".xls", ".xlsx"}:
        return pd.read_excel(path)

    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)

    try:
        return pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        try:
            return pd.read_csv(path, encoding="cp1252")
        except UnicodeDecodeError:
            return pd.read_csv(path, encoding="latin-1")


def _is_missing(value, treat_empty_string_as_null: bool, treat_whitespace_as_null: bool) -> bool:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return True

    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass

    if not isinstance(value, str):
        return False

    if value == "" and treat_empty_string_as_null:
        return True

    if value.strip() == "" and value != "" and treat_whitespace_as_null:
        return True

    return False


# ---------------------------------------------------------------------------
# Completeness handlers (NULL vs EMPTY vs WHITESPACE stay distinct)
# ---------------------------------------------------------------------------


def _require_column(dataframe: pd.DataFrame, column: str) -> None:
    if column not in dataframe.columns:
        raise RuleExecutionError(f"Column not found: {column}")


def _execute_NOT_NULL(dataframe, rule):
    column = rule["column"]
    _require_column(dataframe, column)

    series = dataframe[column]
    applicable = pd.Series(True, index=dataframe.index)
    missing = series.map(lambda v: _is_missing(v, False, False))
    return applicable, applicable & missing


def _execute_NOT_EMPTY(dataframe, rule):
    column = rule["column"]
    _require_column(dataframe, column)

    series = dataframe[column]
    applicable = pd.Series(True, index=dataframe.index)
    # NOT_EMPTY: only the empty string fails. NULL and whitespace-only are
    # NOT empty-string failures (they are separate rules).
    failed = series.map(lambda v: isinstance(v, str) and v == "")
    return applicable, applicable & failed


def _execute_NOT_WHITESPACE(dataframe, rule):
    column = rule["column"]
    _require_column(dataframe, column)

    series = dataframe[column]
    applicable = pd.Series(True, index=dataframe.index)
    failed = series.map(lambda v: isinstance(v, str) and v != "" and v.strip() == "")
    return applicable, applicable & failed


def _execute_legacy_completeness(dataframe, rule):
    """Legacy combined completeness (treat_empty/whitespace flags)."""
    column = rule["column"]
    _require_column(dataframe, column)

    series = dataframe[column]
    applicable = pd.Series(True, index=dataframe.index)
    missing = series.map(
        lambda value: _is_missing(
            value,
            bool(rule.get("treat_empty_string_as_null")),
            bool(rule.get("treat_whitespace_as_null")),
        )
    )
    return applicable, applicable & missing


# ---------------------------------------------------------------------------
# Uniqueness handlers
# ---------------------------------------------------------------------------


def _execute_UNIQUE(dataframe, rule):
    return _execute_uniqueness(dataframe, rule)


def _execute_COMPOSITE_UNIQUE(dataframe, rule):
    return _execute_uniqueness(dataframe, rule)


def _execute_uniqueness(dataframe, rule):
    columns = rule.get("columns") or ([rule["column"]] if "column" in rule else [])

    for column in columns:
        _require_column(dataframe, column)

    params = rule.get("parameters", {}) or {}
    ignore_nulls = params.get("ignore_nulls", rule.get("ignore_nulls", True))
    normalize_ws = params.get("normalize_whitespace", rule.get("normalize_whitespace", True))

    subset = dataframe[columns]

    if ignore_nulls:
        null_mask = subset.isna().any(axis=1)
        for column in columns:
            empty_mask = subset[column].map(
                lambda value: isinstance(value, str) and value.strip() == ""
            )
            null_mask = null_mask | empty_mask
        applicable = ~null_mask
    else:
        applicable = pd.Series(True, index=dataframe.index)

    working = dataframe.loc[applicable, columns].copy()

    if working.empty:
        return applicable, pd.Series(False, index=dataframe.index)

    normalized = working.apply(
        lambda col: col.map(
            lambda value: (
                str(value).strip().lower()
                if isinstance(value, str) and normalize_ws
                else (str(value).strip() if isinstance(value, str) and normalize_ws else str(value))
            )
        )
    )

    duplicated = normalized.duplicated(keep=False)

    failed = pd.Series(False, index=dataframe.index)
    failed.loc[applicable] = duplicated.values

    return applicable, failed


# ---------------------------------------------------------------------------
# Validity handlers
# ---------------------------------------------------------------------------


def _validity_not_missing(dataframe, column):
    """Null/whitespace-only values are missing, not validity failures."""
    series = dataframe[column]
    return ~series.map(lambda value: _is_missing(value, True, True))


def _execute_REGEX(dataframe, rule):
    column = rule["column"]
    _require_column(dataframe, column)

    params = rule.get("parameters", {}) or {}
    pattern_name = params.get("pattern_name")
    pattern_str = params.get("pattern")

    if pattern_name == "email_syntax" or (not pattern_str and rule.get("check") == "email_syntax"):
        compiled = EMAIL_PATTERN
    else:
        if not pattern_str:
            raise RuleExecutionError("REGEX rule requires parameters.pattern or pattern_name=email_syntax")
        try:
            compiled = re.compile(pattern_str)
        except re.error as exc:
            raise RuleExecutionError(f"Invalid REGEX pattern: {exc}") from exc

    series = dataframe[column]
    not_missing = _validity_not_missing(dataframe, column)

    def is_valid(value) -> bool:
        if not isinstance(value, str):
            return False
        return compiled.match(value.strip()) is not None

    valid_mask = series.map(is_valid)
    return pd.Series(True, index=dataframe.index), (~valid_mask) & not_missing


def _execute_NUMERIC_TYPE(dataframe, rule):
    column = rule["column"]
    _require_column(dataframe, column)

    series = dataframe[column]
    not_missing = _validity_not_missing(dataframe, column)

    def is_numeric(value) -> bool:
        if isinstance(value, bool):
            return False
        if isinstance(value, (int, float)):
            return True
        if isinstance(value, str):
            try:
                float(value.strip())
                return True
            except ValueError:
                return False
        return False

    valid_mask = series.map(is_numeric)
    return pd.Series(True, index=dataframe.index), (~valid_mask) & not_missing


def _execute_NUMERIC_RANGE(dataframe, rule):
    column = rule["column"]
    _require_column(dataframe, column)

    params = rule.get("parameters", {}) or {}
    lo = params.get("min")
    hi = params.get("max")
    if lo is None and hi is None:
        raise RuleExecutionError("NUMERIC_RANGE requires min and/or max")

    series = dataframe[column]
    not_missing = _validity_not_missing(dataframe, column)

    def in_range(value) -> bool:
        if isinstance(value, bool):
            return False
        if isinstance(value, (int, float)):
            number = value
        elif isinstance(value, str):
            try:
                number = float(value.strip())
            except ValueError:
                return False
        else:
            return False
        if lo is not None and number < lo:
            return False
        if hi is not None and number > hi:
            return False
        return True

    valid_mask = series.map(in_range)
    return pd.Series(True, index=dataframe.index), (~valid_mask) & not_missing


def _execute_ALLOWED_VALUES(dataframe, rule):
    column = rule["column"]
    _require_column(dataframe, column)

    params = rule.get("parameters", {}) or {}
    allowed = params.get("allowed_values")
    if not isinstance(allowed, list) or not allowed:
        raise RuleExecutionError("ALLOWED_VALUES requires parameters.allowed_values")

    allowed_set = {str(v) for v in allowed}
    case_sensitive = bool(params.get("case_sensitive", False))

    series = dataframe[column]
    not_missing = _validity_not_missing(dataframe, column)

    def is_allowed(value) -> bool:
        if not isinstance(value, str):
            return False
        candidate = value if case_sensitive else value.strip().lower()
        targets = allowed_set if case_sensitive else {v.lower() for v in allowed_set}
        return candidate in targets

    valid_mask = series.map(is_allowed)
    return pd.Series(True, index=dataframe.index), (~valid_mask) & not_missing


def _execute_DATE_FORMAT(dataframe, rule):
    column = rule["column"]
    _require_column(dataframe, column)

    params = rule.get("parameters", {}) or {}
    date_format = params.get("date_format") or rule.get("format")

    series = dataframe[column]
    not_missing = _validity_not_missing(dataframe, column)

    if date_format:
        def is_valid(value) -> bool:
            if isinstance(value, (datetime, date)):
                return True
            if not isinstance(value, str):
                return False
            try:
                datetime.strptime(value.strip(), date_format)
                return True
            except ValueError:
                return False
    else:
        def is_valid(value) -> bool:
            if isinstance(value, (datetime, date)):
                return True
            if not isinstance(value, str):
                return False
            parsed = pd.to_datetime(value.strip(), errors="coerce", format="mixed")
            return parsed is not pd.NaT and not pd.isna(parsed)

    valid_mask = series.map(is_valid)
    return pd.Series(True, index=dataframe.index), (~valid_mask) & not_missing


def _execute_DATATYPE_COMPATIBILITY(dataframe, rule):
    column = rule["column"]
    _require_column(dataframe, column)

    params = rule.get("parameters", {}) or {}
    expected = str(params.get("expected_datatype", "")).lower()
    if not expected:
        raise RuleExecutionError("DATATYPE_COMPATIBILITY requires parameters.expected_datatype")

    series = dataframe[column]
    not_missing = _validity_not_missing(dataframe, column)

    def is_compatible(value) -> bool:
        if expected in {"number", "numeric", "int", "integer", "float", "decimal"}:
            if isinstance(value, bool):
                return False
            if isinstance(value, (int, float)):
                return True
            if isinstance(value, str):
                try:
                    float(value.strip())
                    return True
                except ValueError:
                    return False
            return False
        if expected in {"string", "text"}:
            return isinstance(value, str)
        if expected in {"date", "datetime", "timestamp"}:
            return isinstance(value, (datetime, date)) or (
                isinstance(value, str) and pd.to_datetime(value, errors="coerce") is not pd.NaT
            )
        if expected in {"boolean", "bool"}:
            return isinstance(value, bool) or (isinstance(value, str) and value.strip().lower() in {"true", "false", "0", "1", "yes", "no"})
        return True

    valid_mask = series.map(is_compatible)
    return pd.Series(True, index=dataframe.index), (~valid_mask) & not_missing


def _execute_legacy_validity(dataframe, rule):
    column = rule["column"]
    _require_column(dataframe, column)

    series = dataframe[column]
    not_missing = _validity_not_missing(dataframe, column)

    check = rule.get("check")

    if check == "email_syntax":
        def is_valid(value) -> bool:
            if not isinstance(value, str):
                return False
            return EMAIL_PATTERN.match(value.strip()) is not None

        valid_mask = series.map(is_valid)

    elif check == "numeric_type":
        def is_valid(value) -> bool:
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return True
            if isinstance(value, str):
                try:
                    float(value.strip())
                    return True
                except ValueError:
                    return False
            return False

        valid_mask = series.map(is_valid)

    elif check == "date_parseable":
        parsed = pd.to_datetime(series, errors="coerce", format="mixed")

        def is_parseable(value) -> bool:
            return isinstance(value, (datetime, date))

        from_pandas = series.map(is_parseable)
        valid_mask = from_pandas | parsed.notna()

    else:
        raise RuleExecutionError(f"Unsupported validity check: {check}")

    return pd.Series(True, index=dataframe.index), (~valid_mask) & not_missing


# ---------------------------------------------------------------------------
# Consistency handlers
# ---------------------------------------------------------------------------


def _coerce_number(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def _coerce_datetime(value):
    if isinstance(value, (datetime, pd.Timestamp)):
        return value
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day)
    if isinstance(value, str):
        parsed = pd.to_datetime(value.strip(), errors="coerce", format="mixed")
        if parsed is not pd.NaT and not pd.isna(parsed):
            return parsed.to_pydatetime()
    return None


def _row_columns(dataframe, rule, params, *names):
    for name in names:
        column = params.get(name)
        if column:
            _require_column(dataframe, column)


def _execute_COLUMN_COMPARISON(dataframe, rule):
    params = rule.get("parameters", {}) or {}
    left = params.get("left_column")
    right = params.get("right_column")
    operator = params.get("operator")
    _row_columns(dataframe, rule, params, "left_column", "right_column")

    if operator not in {"<=", ">=", "<", ">", "==", "!="}:
        raise RuleExecutionError(f"Unsupported comparison operator: {operator}")

    ops = {
        "<": lambda a, b: a < b,
        "<=": lambda a, b: a <= b,
        ">": lambda a, b: a > b,
        ">=": lambda a, b: a >= b,
        "==": lambda a, b: a == b,
        "!=": lambda a, b: a != b,
    }

    left_series = dataframe[left]
    right_series = dataframe[right]
    # A comparison is only APPLICABLE when both operands exist; rows missing
    # an operand belong to completeness, not to this rule's pass population.
    applicable = ~(left_series.isna() | right_series.isna())

    ops_map = ops[operator]

    def fails(left_value, right_value) -> bool:
        a, b = _coerce_number(left_value), _coerce_number(right_value)
        if a is None or b is None:
            # Non-numeric operands compare as strings when both are strings.
            if isinstance(left_value, str) and isinstance(right_value, str):
                return not ops_map(left_value.strip(), right_value.strip())
            return False  # incomparable -> not a consistency failure
        return not ops_map(a, b)

    failed = dataframe.apply(lambda row: fails(row[left], row[right]), axis=1)
    return applicable, failed & applicable


def _execute_DATE_ORDER(dataframe, rule):
    params = rule.get("parameters", {}) or {}
    left = params.get("left_column")
    right = params.get("right_column")
    allow_equal = bool(params.get("allow_equal", False))
    _row_columns(dataframe, rule, params, "left_column", "right_column")

    left_series = dataframe[left]
    right_series = dataframe[right]
    applicable = ~(left_series.isna() | right_series.isna())

    def fails(left_value, right_value) -> bool:
        a, b = _coerce_datetime(left_value), _coerce_datetime(right_value)
        if a is None or b is None:
            return True  # unparseable dates are date-order violations
        return not (a <= b if allow_equal else a < b)

    failed = dataframe.apply(lambda row: fails(row[left], row[right]), axis=1)
    return applicable, failed & applicable


def _execute_ARITHMETIC_RELATION(dataframe, rule):
    params = rule.get("parameters", {}) or {}
    left = params.get("left_column")
    operator = params.get("operator")
    right = params.get("right_column")
    expected = params.get("expected_column")
    tolerance = params.get("tolerance", 0.0)
    _row_columns(dataframe, rule, params, "left_column", "right_column", "expected_column")

    if operator not in {"*", "+"}:
        raise RuleExecutionError(f"Unsupported arithmetic operator: {operator}")
    if not _is_number_tolerance(tolerance):
        raise RuleExecutionError("tolerance must be a number")

    left_series = dataframe[left]
    right_series = dataframe[right]
    expected_series = dataframe[expected]
    applicable = ~(left_series.isna() | right_series.isna() | expected_series.isna())

    def fails(row) -> bool:
        a = _coerce_number(row[left])
        b = _coerce_number(row[right])
        c = _coerce_number(row[expected])
        if a is None or b is None or c is None:
            return True  # non-numeric operand breaks the relation
        computed = a * b if operator == "*" else a + b
        if tolerance:
            allowed = abs(c) * tolerance / 100.0 if tolerance > 1 else tolerance
            return abs(computed - c) > allowed
        return not _close(computed, c)

    failed = dataframe.apply(fails, axis=1)
    return applicable, failed & applicable


def _is_number_tolerance(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _close(a, b) -> bool:
    if a == b:
        return True
    scale = max(abs(a), abs(b), 1.0)
    return abs(a - b) <= 1e-9 * scale


def _execute_FUNCTIONAL_DEPENDENCY(dataframe, rule):
    params = rule.get("parameters", {}) or {}
    determinant = params.get("determinant_column")
    dependent = params.get("dependent_column")
    _row_columns(dataframe, rule, params, "determinant_column", "dependent_column")

    subset = dataframe[[determinant, dependent]]
    complete = subset.dropna()
    if complete.empty:
        return pd.Series(True, index=dataframe.index), pd.Series(False, index=dataframe.index)

    violations = complete.groupby(determinant)[dependent].nunique()
    bad_determinants = set(violations[violations > 1].index)

    def fails(value) -> bool:
        return value in bad_determinants

    failed = dataframe[determinant].map(lambda v: (not pd.isna(v)) and fails(v))
    return pd.Series(True, index=dataframe.index), failed


def _execute_REFERENCE_HIERARCHY(dataframe, rule):
    params = rule.get("parameters", {}) or {}
    parent_column = params.get("parent_column")
    child_column = params.get("child_column")
    hierarchy = params.get("hierarchy")
    _row_columns(dataframe, rule, params, "parent_column", "child_column")

    if not isinstance(hierarchy, list) or not hierarchy:
        raise RuleExecutionError("REFERENCE_HIERARCHY requires parameters.hierarchy (list of allowed child->parent maps)")

    # hierarchy: [{"child": "Chennai", "parent": "Tamil Nadu"}, ...] or
    # {"Tamil Nadu": ["Chennai", ...]} — normalized to (parent, child) pairs.
    pairs: set[tuple[str, str]] = set()
    if isinstance(hierarchy, dict):
        for parent, children in hierarchy.items():
            for child in children or []:
                pairs.add((str(parent).strip().lower(), str(child).strip().lower()))
    else:
        for entry in hierarchy:
            pairs.add(
                (
                    str(entry.get("parent", "")).strip().lower(),
                    str(entry.get("child", "")).strip().lower(),
                )
            )

    parent_series = dataframe[parent_column]
    child_series = dataframe[child_column]
    applicable = ~(parent_series.isna() | child_series.isna())

    def fails(p_value, c_value) -> bool:
        key = (str(p_value).strip().lower(), str(c_value).strip().lower())
        return key not in pairs

    failed = dataframe.apply(lambda row: fails(row[parent_column], row[child_column]), axis=1)
    return applicable, failed & applicable


# ---------------------------------------------------------------------------
# Referential integrity
# ---------------------------------------------------------------------------


def _execute_FOREIGN_KEY_EXISTS(dataframe, rule):
    """RI handler. Executes the CHILD side; parent table loaded separately.

    The router passes rule_json.parameters.parent_* plus the pre-loaded
    parent value set via execution context (rule["__parent_values__"]).
    """
    params = rule.get("parameters", {}) or {}
    child_column = params.get("child_column")
    _require_column(dataframe, child_column)

    parent_values = rule.get("__parent_values__")
    if parent_values is None:
        raise RuleExecutionError(
            "FOREIGN_KEY_EXISTS execution requires the approved parent values (internal context missing)"
        )

    parent_set = {str(v).strip() for v in parent_values}

    child_series = dataframe[child_column]
    not_missing = ~child_series.isna()

    def fails(value) -> bool:
        if isinstance(value, str):
            return value.strip() not in parent_set
        return str(value) not in parent_set

    failed = child_series.map(fails)
    return pd.Series(True, index=dataframe.index), failed & not_missing


# ---------------------------------------------------------------------------
# Timeliness handlers
# ---------------------------------------------------------------------------


def _execute_FRESHNESS_THRESHOLD(dataframe, rule):
    column = rule["column"]
    _require_column(dataframe, column)

    params = rule.get("parameters", {}) or {}
    max_age = params.get("max_age") or {}
    amount = max_age.get("amount")
    unit = max_age.get("unit")
    if not _is_number_tolerance(amount) or amount <= 0:
        raise RuleExecutionError("FRESHNESS_THRESHOLD requires parameters.max_age {amount, unit}")

    deltas = {
        "minutes": timedelta(minutes=amount),
        "hours": timedelta(hours=amount),
        "days": timedelta(days=amount),
        "weeks": timedelta(weeks=amount),
    }
    cutoff = _now_utc() - deltas[unit]

    series = dataframe[column]
    not_missing = _validity_not_missing(dataframe, column)

    def fails(value) -> bool:
        parsed = _coerce_datetime(value)
        if parsed is None:
            return True  # unparseable timestamps violate timeliness evidence
        ts = parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
        return ts < cutoff

    failed = series.map(fails)
    return pd.Series(True, index=dataframe.index), failed & not_missing


def _execute_DEADLINE(dataframe, rule):
    column = rule["column"]
    _require_column(dataframe, column)

    params = rule.get("parameters", {}) or {}
    deadline_str = params.get("deadline")
    time_of_day = params.get("time_of_day")
    recurring = bool(params.get("recurring"))

    if deadline_str:
        try:
            deadline = datetime.fromisoformat(str(deadline_str))
        except ValueError as exc:
            raise RuleExecutionError(f"Invalid deadline timestamp: {deadline_str}") from exc
        deadline = deadline if deadline.tzinfo else deadline.replace(tzinfo=timezone.utc)

        series = dataframe[column]
        not_missing = _validity_not_missing(dataframe, column)

        def fails(value) -> bool:
            parsed = _coerce_datetime(value)
            if parsed is None:
                return True
            ts = parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
            return ts < deadline

        failed = series.map(fails)
        return pd.Series(True, index=dataframe.index), failed & not_missing

    if time_of_day and recurring:
        hour, minute = (int(p) for p in str(time_of_day).split(":")[:2])
        now = _now_utc()
        series = dataframe[column]
        not_missing = _validity_not_missing(dataframe, column)

        def fails(value) -> bool:
            parsed = _coerce_datetime(value)
            if parsed is None:
                return True
            ts = parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
            expected = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if now < expected:
                expected = expected - timedelta(days=1)
            return ts < expected

        failed = series.map(fails)
        return pd.Series(True, index=dataframe.index), failed & not_missing

    raise RuleExecutionError("DEADLINE requires parameters.deadline or recurring time_of_day")


# ---------------------------------------------------------------------------
# Accuracy handlers (require a user-configured reference source)
# ---------------------------------------------------------------------------


def _execute_REFERENCE_LOOKUP(dataframe, rule):
    params = rule.get("parameters", {}) or {}
    reference_values = rule.get("__reference_values__")
    if reference_values is None:
        raise RuleExecutionError(
            "REFERENCE_LOOKUP requires a configured reference source; none was supplied"
        )

    column = rule["column"]
    _require_column(dataframe, column)

    reference_set = {str(v).strip() for v in reference_values}
    series = dataframe[column]
    not_missing = _validity_not_missing(dataframe, column)

    match_mode = str(params.get("match_mode", "exact")).lower()

    def fails(value) -> bool:
        if not isinstance(value, str):
            return str(value) not in reference_set
        candidate = value.strip()
        if match_mode == "exact":
            return candidate not in reference_set
        # "contains": reference entry appears within the value.
        return not any(entry in candidate for entry in reference_set)

    failed = series.map(fails)
    return pd.Series(True, index=dataframe.index), failed & not_missing


def _execute_EXTERNAL_REFERENCE_VERIFICATION(dataframe, rule):
    # Deliberately NOT implemented as a live call in this iteration: an
    # external provider needs caching, deduplication, rate limiting and
    # consent. The structured rule is accepted and validated, but execution
    # reports NOT_CHECKED evidence rather than making network calls silently.
    raise RuleExecutionError(
        "EXTERNAL_REFERENCE_VERIFICATION is validated but not executable until a provider is configured "
        "(caching, deduplication and rate limiting are required first)."
    )


# ---------------------------------------------------------------------------
# Template dispatch
# ---------------------------------------------------------------------------

TEMPLATE_HANDLERS = {
    "NOT_NULL": _execute_NOT_NULL,
    "NOT_EMPTY": _execute_NOT_EMPTY,
    "NOT_WHITESPACE": _execute_NOT_WHITESPACE,
    "UNIQUE": _execute_UNIQUE,
    "COMPOSITE_UNIQUE": _execute_COMPOSITE_UNIQUE,
    "REGEX": _execute_REGEX,
    "NUMERIC_TYPE": _execute_NUMERIC_TYPE,
    "NUMERIC_RANGE": _execute_NUMERIC_RANGE,
    "ALLOWED_VALUES": _execute_ALLOWED_VALUES,
    "DATE_FORMAT": _execute_DATE_FORMAT,
    "DATATYPE_COMPATIBILITY": _execute_DATATYPE_COMPATIBILITY,
    "COLUMN_COMPARISON": _execute_COLUMN_COMPARISON,
    "DATE_ORDER": "not_a_column_rule",  # replaced below; cross-column handled
    "ARITHMETIC_RELATION": _execute_ARITHMETIC_RELATION,
    "FUNCTIONAL_DEPENDENCY": _execute_FUNCTIONAL_DEPENDENCY,
    "REFERENCE_HIERARCHY": _execute_REFERENCE_HIERARCHY,
    "FOREIGN_KEY_EXISTS": _execute_FOREIGN_KEY_EXISTS,
    "FRESHNESS_THRESHOLD": _execute_FRESHNESS_THRESHOLD,
    "DEADLINE": _execute_DEADLINE,
    "REFERENCE_LOOKUP": _execute_REFERENCE_LOOKUP,
    "EXTERNAL_REFERENCE_VERIFICATION": _execute_EXTERNAL_REFERENCE_VERIFICATION,
}

TEMPLATE_HANDLERS["COLUMN_COMPARISON"] = _execute_COLUMN_COMPARISON
TEMPLATE_HANDLERS["DATE_ORDER"] = _execute_DATE_ORDER

LEGACY_HANDLERS = {
    "completeness": _execute_legacy_completeness,
    "uniqueness": _execute_uniqueness,
    "validity": _execute_legacy_validity,
}

# Backwards-compatible private aliases (RCA re-derives failure masks).
_execute_completeness = _execute_legacy_completeness
_execute_validity = _execute_legacy_validity


def get_failed_mask(dataframe: pd.DataFrame, rule_json: dict) -> pd.Series:
    """Return only the failed mask for a rule (template or legacy).

    Used by RCA to re-derive WHICH rows failed without duplicating any
    execution logic.
    """
    template = rule_json.get("rule_template") or rule_json.get("template")

    if template and template in TEMPLATE_HANDLERS:
        _, failed = TEMPLATE_HANDLERS[template](dataframe, rule_json)
        return failed

    rule_type = rule_json.get("type")
    handler = LEGACY_HANDLERS.get(rule_type)
    if handler is None:
        raise RuleExecutionError(f"Unsupported rule type: {rule_type}")

    _, failed = handler(dataframe, rule_json)
    return failed


def execute_rule_on_dataframe(dataframe: pd.DataFrame, rule_json: dict) -> dict:
    """Execute one rule against a dataframe. Returns bounded results."""
    template = rule_json.get("rule_template") or rule_json.get("template")

    if template and template in TEMPLATE_HANDLERS:
        applicable, failed = TEMPLATE_HANDLERS[template](dataframe, rule_json)
    else:
        rule_type = rule_json.get("type")
        handler = LEGACY_HANDLERS.get(rule_type)
        if handler is None:
            raise RuleExecutionError(f"Unsupported rule type: {rule_type}")
        applicable, failed = handler(dataframe, rule_json)

    total_rows = len(dataframe)
    applicable_rows = int(applicable.sum())
    failed_rows = int(failed.sum())
    passed_rows = applicable_rows - failed_rows
    not_applicable = total_rows - applicable_rows

    # N/A semantics: zero applicability means NOT evaluated, never 100%.
    pass_rate = (passed_rows / applicable_rows * 100) if applicable_rows else None
    violation_rate = (failed_rows / applicable_rows * 100) if applicable_rows else None

    failure_examples = []

    if failed_rows > 0:
        failed_indices = dataframe.index[failed][:MAX_EVIDENCE_EXAMPLES]

        for index in failed_indices:
            row = dataframe.loc[index]
            example = {"row_index": int(index)}

            for column in dataframe.columns:
                value = row[column]

                if isinstance(value, (pd.Timestamp, datetime, date)):
                    example[column] = str(value)
                elif pd.isna(value):
                    example[column] = None
                elif isinstance(value, (int, float, str, bool)):
                    example[column] = value
                else:
                    example[column] = str(value)

                if len(example) > 12:
                    break

            failure_examples.append(example)

    return {
        "status": "passed",
        "total_rows": total_rows,
        "applicable_rows": applicable_rows,
        "passed_rows": passed_rows,
        "failed_rows": failed_rows,
        "not_applicable_rows": not_applicable,
        "pass_rate": round(pass_rate, 4) if pass_rate is not None else None,
        "violation_rate": round(violation_rate, 4) if violation_rate is not None else None,
        "failure_examples": failure_examples,
    }


def execute_rule(
    db: Session,
    rule: Rule,
    dataset_id: int,
    version_id: int,
    version_number: int,
) -> RuleExecution:
    """Execute an approved rule against its table and persist the result."""
    if rule.status != "approved":
        raise RuleExecutionError("Only approved rules can be executed.")

    stored_filename = _find_stored_file_for_table(
        db, version_id, rule.table_name
    )

    if stored_filename is None:
        raise RuleExecutionError(
            f"No raw file found for table {rule.table_name}."
        )

    dataframe = _load_table(
        dataset_id,
        version_number,
        stored_filename,
    )

    rule_json = dict(rule.rule_json or {})

    # RI rules need the approved parent values. The parent table may live in
    # another file; it is loaded through the same storage path as the child.
    if (rule_json.get("rule_template")) == "FOREIGN_KEY_EXISTS":
        rule_json["__parent_values__"] = _load_parent_values(
            db, dataset_id, version_id, version_number, rule_json
        )

    try:
        result = execute_rule_on_dataframe(dataframe, rule_json)
    except RuleExecutionError:
        raise
    except Exception as exc:
        raise RuleExecutionError(f"Rule execution failed: {exc}") from exc

    execution = RuleExecution(
        rule_id=rule.rule_id,
        version_id=version_id,
        status=result["status"],
        total_rows=result["total_rows"],
        applicable_rows=result["applicable_rows"],
        passed_rows=result["passed_rows"],
        failed_rows=result["failed_rows"],
        not_applicable_rows=result["not_applicable_rows"],
        pass_rate=result["pass_rate"] if result["pass_rate"] is not None else 0.0,
        violation_rate=result["violation_rate"] if result["violation_rate"] is not None else 0.0,
        evidence_json={
            "failure_examples": result["failure_examples"],
            "pass_rate_not_applicable": result["pass_rate"] is None,
            "violation_rate_not_applicable": result["violation_rate"] is None,
            "rule_version_number": getattr(rule, "version_number", 1),
            "rule_code": getattr(rule, "rule_code", None),
        },
    )

    db.add(execution)
    db.flush()

    return execution


def _load_parent_values(
    db: Session,
    dataset_id: int,
    version_id: int,
    version_number: int,
    rule_json: dict,
) -> list:
    params = rule_json.get("parameters", {}) or {}
    parent_table = params.get("parent_table")
    parent_column = params.get("parent_column")

    parent_frame = _load_table_for_name(db, dataset_id, version_id, version_number, parent_table)
    if parent_frame is None:
        raise RuleExecutionError(f"Parent table file not found: {parent_table}")
    if parent_column not in parent_frame.columns:
        raise RuleExecutionError(f"Parent column not found: {parent_table}.{parent_column}")

    return [v for v in parent_frame[parent_column].dropna().tolist()]


def _load_table_for_name(
    db: Session,
    dataset_id: int,
    version_id: int,
    version_number: int,
    table_name: str,
) -> pd.DataFrame | None:
    stored_filename = _find_stored_file_for_table(db, version_id, table_name)
    if stored_filename is None:
        return None
    return _load_table(dataset_id, version_number, stored_filename)


def _find_stored_file_for_table(
    db: Session,
    version_id: int,
    table_name: str,
) -> str | None:
    from app.models import FileMetadata, TableMetadata

    table = (
        db.query(TableMetadata)
        .filter(
            TableMetadata.version_id == version_id,
            TableMetadata.table_name == table_name,
        )
        .first()
    )

    if table is None:
        return None

    file_metadata = (
        db.query(FileMetadata)
        .filter(FileMetadata.version_id == version_id)
        .all()
    )

    for metadata in file_metadata:
        original = metadata.original_filename or ""

        if original == table.source_file or original.replace("\\", "/").endswith(
            "/" + table.source_file
        ):
            return metadata.stored_filename

    return None

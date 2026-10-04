from __future__ import annotations

import datetime
import math
import re
from collections import Counter
from decimal import Decimal
from functools import lru_cache
from itertools import combinations
from typing import Any, Sequence

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Profiling limits. Profiling is deterministic evidence gathering. Heavy
# cross-column analyses are bounded; any sampled analysis reports
# "sampled": true + "sample_rows": N. Basic counts are never sampled.
# ---------------------------------------------------------------------------

PROFILING_TOP_CATEGORIES = 10
PROFILING_MAX_COMBINATIONS = 3
PROFILING_MAX_COMBO_COLUMNS = 3
PROFILING_NEAR_CONSTANT_THRESHOLD = 0.98
PROFILING_COMPOSITE_PIGEONHOLE_SHARE = 0.5
PROFILING_COMPOSITE_UNIQUENESS_THRESHOLD = 95.0
PROFILING_COMPOSITE_MIN_ROW_COVERAGE = 0.5
PROFILING_MAX_COMPOSITE_EVALUATIONS = 80
PROFILING_COMPOSITE_SAMPLE_ROWS = 500_000
PROFILING_COMPOSITE_NEAR_EXACT_THRESHOLD = 99.9
PROFILING_DATE_SAMPLE_DISTINCT = 5000
PROFILING_DATE_PREFILTER_DISTINCT = 500
PROFILING_DATE_LIKE_THRESHOLD = 95.0
PROFILING_SHAPE_SAMPLE_ROWS = 200_000
PROFILING_SHAPE_REGEX_MAX_VALUES = 5000
PROFILING_SHAPE_DISPLAY_MAX_VALUES = 200
PROFILING_SHAPE_MAX_SHAPES = 5
PROFILING_SHAPE_REGULAR_THRESHOLD = 95.0
PROFILING_SHAPE_MAX_MEAN_LENGTH = 64
PROFILING_CASE_VARIANT_MAX_DISTINCT = 50_000
PROFILING_CASE_VARIANT_EXAMPLES = 3
PROFILING_FD_MIN_COVERAGE = 98.0
PROFILING_FD_MAX_COLUMNS = 40
PROFILING_FD_MAX_PAIRS = 1200
PROFILING_FD_SAMPLE_ROWS = 200_000
PROFILING_FD_MAX_REPORTED = 30
PROFILING_FD_UNIQUE_RATIO = 0.95
PROFILING_CO_MISSING_MAX_COLUMNS = 30
PROFILING_CO_MISSING_MAX_PATTERNS = 3
PROFILING_PACKED_MASK_BITS = 62
PROFILING_MAX_OBSERVATIONS = 60
PROFILING_MAX_VIOLATION_EXAMPLES = 3
PROFILING_IDENTIFIER_MAX_LENGTH_RANGE = 4
PROFILING_LOW_CARDINALITY_MAX = 50

DISGUISED_MISSING_TOKENS = (
    "n/a", "na", "nan", "null", "none", "nil", "missing", "unknown",
    "undefined", "tbd", "not available", "not applicable", "-", "--", "?",
    "??", "..",
)
DISGUISED_MISSING_MAX_SHARE = 0.5
DISGUISED_MISSING_TOP_TOKENS = 5

NUMERIC_SENTINEL_VALUES = (-1, 0, 99, 999, 9999, -999, -9999, 99999)
NUMERIC_SENTINEL_MIN_SHARE = 0.01
NUMERIC_MAX_DECIMAL_PLACES = 6

# Code-like NAME tokens: columns whose values are labels, not quantities.
CODE_LIKE_NAME_TOKENS = (
    "zip", "zipcode", "postal", "postcode", "pin", "pincode", "phone",
    "mobile", "tel", "telephone", "fax", "id", "code", "sku", "serial",
    "account", "acct", "ssn", "ref", "reference", "key", "keys", "fk", "pk",
)

# postal_like: pure digits (3-10) or specific postcode shapes.
_POSTAL_PATTERNS = (
    r"\d{3,10}",
    r"[A-Za-z]{1,2}\d[A-Za-z\d]?\s?\d[A-Za-z]{2}",   # UK
    r"[A-Za-z]\d[A-Za-z][ -]?\d[A-Za-z]\d",           # Canada
    r"\d{5}-\d{4}",                                   # US ZIP+4
    r"\d{4}\s?[A-Za-z]{2}",                           # Netherlands
)
_COMBINED_POSTAL_REGEX = "|".join(f"(?:{p})" for p in _POSTAL_PATTERNS)

SUPPORTED_DATE_FORMATS: tuple[tuple[str, str], ...] = (
    ("%Y-%m-%d", r"^\d{4}-\d{1,2}-\d{1,2}$"),
    ("%Y/%m/%d", r"^\d{4}/\d{1,2}/\d{1,2}$"),
    ("%d/%m/%Y", r"^\d{1,2}/\d{1,2}/\d{4}$"),
    ("%m/%d/%Y", r"^\d{1,2}/\d{1,2}/\d{4}$"),
    ("%d-%m-%Y", r"^\d{1,2}-\d{1,2}-\d{4}$"),
    ("%m-%d-%Y", r"^\d{1,2}-\d{1,2}-\d{4}$"),
    ("%d.%m.%Y", r"^\d{1,2}\.\d{1,2}\.\d{4}$"),
    ("%d-%b-%Y", r"^\d{1,2}-[A-Za-z]{3}-\d{4}$"),
    ("%d %b %Y", r"^\d{1,2} [A-Za-z]{3} \d{4}$"),
    ("%d %B %Y", r"^\d{1,2} [A-Za-z]{3,9} \d{4}$"),
    ("%B %d, %Y", r"^[A-Za-z]{3,9} \d{1,2}, \d{4}$"),
)
_TIME_SUFFIX_RE = r"(?:[ T]\d{1,2}:\d{2}(?::\d{2})?(?:\.\d+)?)?"
COMPACT_DATE_NAME_TOKENS = (
    "date", "dt", "day", "time", "timestamp", "created", "updated", "dob",
    "birth",
)
_TOKEN_SPLIT_RE = re.compile(r"[^A-Za-z0-9]+")


@lru_cache(maxsize=1)
def _combined_date_shape_regex() -> str:
    regexes: list[str] = []
    for _, shape_regex in SUPPORTED_DATE_FORMATS:
        regexes.append(shape_regex)
        regexes.append(shape_regex[:-1] + _TIME_SUFFIX_RE + "$")
    regexes.append(r"^\d{8}$")
    return "|".join(f"(?:{r})" for r in regexes)


IDENTIFIER_NAME_TOKENS = (
    "id", "identifier", "uuid", "guid", "key", "keys", "fk", "pk", "code",
    "number", "no", "ref", "reference", "sku", "serial",
)


def _column_name_tokens(name: Any) -> list[str]:
    """Lowercase word tokens of a column name (snake, kebab, camelCase)."""
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", str(name))
    return [t.lower() for t in _TOKEN_SPLIT_RE.split(text) if t]


def identifier_name_signal_from_name(name: Any) -> bool:
    """Token-aware identifier name evidence (deterministic, no ML)."""
    return any(t in IDENTIFIER_NAME_TOKENS for t in _column_name_tokens(name))


def _column_name_has_token(name: Any, tokens: Sequence[str]) -> bool:
    return any(t in tokens for t in _column_name_tokens(name))


def _clean_value(value: Any) -> Any:
    """Convert pandas/numpy values into JSON-safe Python values."""
    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.datetime64):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, (datetime.datetime, datetime.date, datetime.time)):
        return value.isoformat()
    if isinstance(value, (datetime.timedelta, pd.Timedelta)):
        return str(value)
    if isinstance(value, np.bool_):
        return bool(value)
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, Decimal):
        if value.is_nan() or value.is_infinite():
            return None
        return float(value)
    return value


def _error_message(error: BaseException) -> str:
    """Exception class + message, truncated to 200 characters."""
    return f"{type(error).__name__}: {error}"[:200]


def _as_string_series(series: pd.Series) -> pd.Series:
    """Non-null string image of a column (non-string cells via str())."""
    try:
        return series.astype("string").dropna()
    except (TypeError, ValueError):
        return series.dropna().map(str).astype("string")


def _is_text_like(series: pd.Series) -> bool:
    return bool(pd.api.types.is_string_dtype(series) or series.dtype == object)


def _vectorised_string_counts(series: pd.Series) -> tuple[int, int]:
    """(empty strings, whitespace-only strings); non-strings never counted."""
    if len(series) == 0 or not _is_text_like(series):
        return 0, 0
    if series.dtype == object:
        mask = series.map(lambda v: isinstance(v, str))
        strings = series[mask].astype("string")
    else:
        strings = series.dropna().astype("string")
    if strings.empty:
        return 0, 0
    empty = int((strings == "").sum())
    blank = int(((strings.str.strip() == "") & (strings != "")).sum())
    return empty, blank


def _value_counts_safe(series: pd.Series) -> tuple[pd.Series | None, bool]:
    """value_counts(dropna=True) with an unhashable-cell fallback."""
    try:
        return series.value_counts(dropna=True), False
    except TypeError:
        try:
            return series.astype(str).value_counts(dropna=True), True
        except Exception:  # noqa: BLE001
            return None, False


def _top_value_share(value_counts: pd.Series | None, non_null: int) -> float:
    if value_counts is None or value_counts.empty or non_null == 0:
        return 0.0
    return float(value_counts.iloc[0]) / non_null


def _pattern_summary(
    series: pd.Series, value_counts: pd.Series | None = None
) -> dict[str, int]:
    """Structural text-pattern counts (vectorised, distinct-weighted if given)."""
    patterns = {
        "email_like": 0, "phone_like": 0, "postal_like": 0, "currency_like": 0,
        "numeric_like": 0, "date_like": 0, "alphanumeric_like": 0,
        "contains_whitespace": 0, "contains_special_character": 0,
    }
    if len(series) == 0 or not _is_text_like(series):
        return patterns

    if value_counts is not None and not value_counts.empty:
        values = pd.Series(value_counts.index.astype(str), dtype="string")
        weights = value_counts.to_numpy(dtype="int64")
    else:
        values = _as_string_series(series).reset_index(drop=True)
        weights = None
    if values.empty:
        return patterns

    def total(mask: pd.Series) -> int:
        arr = mask.fillna(False).to_numpy(dtype=bool)
        return int(weights[arr].sum()) if weights is not None else int(arr.sum())

    stripped = values.str.strip()
    patterns["email_like"] = total(
        stripped.str.fullmatch(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", na=False)
    )
    digits = stripped.str.replace(r"\D", "", regex=True)
    phone_shape = stripped.str.fullmatch(r"\+?[0-9][0-9\s().-]{5,}", na=False)
    patterns["phone_like"] = total(digits.str.len().between(7, 15) & phone_shape.fillna(False))
    patterns["postal_like"] = total(stripped.str.fullmatch(_COMBINED_POSTAL_REGEX, na=False))
    patterns["currency_like"] = total(stripped.str.fullmatch(r"[A-Z]{3}", na=False))
    patterns["numeric_like"] = total(stripped.str.fullmatch(r"[-+]?\d+(\.\d+)?", na=False))
    patterns["date_like"] = total(stripped.str.fullmatch(_combined_date_shape_regex(), na=False))
    patterns["alphanumeric_like"] = total(stripped.str.fullmatch(r"[A-Za-z0-9]+", na=False))
    patterns["contains_whitespace"] = total(values.str.contains(r"\s", regex=True, na=False))
    patterns["contains_special_character"] = total(
        values.str.contains(r"[^A-Za-z0-9\s]", regex=True, na=False)
    )
    return patterns


def _separator_characters(
    series: pd.Series, top_n: int = 5, value_counts: pd.Series | None = None
) -> dict[str, int]:
    """Most frequent non-alphanumeric, non-space characters (row weighted)."""
    if value_counts is not None and not value_counts.empty:
        values = pd.Series(value_counts.index.astype(str))
        weights = value_counts.to_numpy(dtype="int64")
    else:
        if len(series) == 0 or not _is_text_like(series):
            return {}
        values = _as_string_series(series).astype(str).reset_index(drop=True)
        weights = np.ones(len(values), dtype="int64")
    if values.empty:
        return {}
    if len(values) > 50_000:  # bounded: deterministic head of distinct values
        values, weights = values.iloc[:50_000], weights[:50_000]
    counter: Counter = Counter()
    for chars, weight in zip(values.str.findall(r"[^A-Za-z0-9\s]"), weights):
        for char in chars:
            counter[char] += int(weight)
    ordered = sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    return {char: count for char, count in ordered[:top_n]}


def _collapse_shape(shape: str) -> str:
    """Run-length collapse: AA-9999 -> A{2}-9{4}."""
    if not shape:
        return shape
    parts: list[str] = []
    i = 0
    while i < len(shape):
        j = i
        while j < len(shape) and shape[j] == shape[i]:
            j += 1
        run = j - i
        parts.append(shape[i] if run == 1 else f"{shape[i]}{{{run}}}")
        i = j
    return "".join(parts)


def _display_shape(shape: str, values: Sequence[str]) -> str:
    """Readable class-and-size form: [A-Z]{3}-[A-Z]{2}-[0-9]{8}.

    Letter case is read from the values of that shape (upper -> A-Z,
    lower -> a-z, mixed -> A-Za-z); digits -> 0-9; other characters stay.
    """
    sample = [v for v in list(values)[:PROFILING_SHAPE_DISPLAY_MAX_VALUES] if len(v) == len(shape)]
    tokens: list[str] = []
    for position, char in enumerate(shape):
        if char == "9":
            tokens.append("[0-9]")
        elif char == "A":
            chars = [v[position] for v in sample] or ["A"]
            if all(c.isupper() for c in chars):
                tokens.append("[A-Z]")
            elif all(c.islower() for c in chars):
                tokens.append("[a-z]")
            else:
                tokens.append("[A-Za-z]")
        else:
            tokens.append(char)
    out: list[str] = []
    i = 0
    while i < len(tokens):
        j = i
        while j < len(tokens) and tokens[j] == tokens[i]:
            j += 1
        run = j - i
        out.append(tokens[i] if run == 1 or not tokens[i].startswith("[") else f"{tokens[i]}{{{run}}}")
        if run > 1 and not tokens[i].startswith("["):
            out[-1] = tokens[i] * run
        i = j
    return "".join(out)


def _suggested_regex(values: pd.Series) -> str | None:
    """Anchored regex from the values of ONE shape; matches all inputs."""
    items = [str(v) for v in values]
    if not items:
        return None
    lengths = {len(v) for v in items}
    if len(lengths) != 1:
        return None
    length = lengths.pop()
    if length == 0:
        return None
    parts: list[str] = []
    for pos in range(length):
        chars = {v[pos] for v in items}
        digits = any(c.isdigit() for c in chars)
        upper = any(c.isupper() for c in chars)
        lower = any(c.islower() for c in chars)
        other = any(not c.isalnum() for c in chars)
        if not other and digits and upper and lower:
            parts.append("[A-Za-z0-9]")
        elif not other and digits and upper:
            parts.append("(?:\\d|[A-Z])")
        elif not other and digits and lower:
            parts.append("(?:\\d|[a-z])")
        elif not other and upper and lower:
            parts.append("[A-Za-z]")
        elif not other and digits:
            parts.append("\\d")
        elif not other and upper:
            parts.append("[A-Z]")
        elif not other and lower:
            parts.append("[a-z]")
        elif len(chars) == 1:
            parts.append(re.escape(next(iter(chars))))
        else:
            parts.append("[" + "".join(re.escape(c) for c in sorted(chars)) + "]")
    compressed: list[str] = []
    i = 0
    while i < len(parts):
        j = i
        while j < len(parts) and parts[j] == parts[i]:
            j += 1
        run = j - i
        if run == 1:
            compressed.append(parts[i])
        elif parts[i].startswith("\\"):
            compressed.append(f"(?:{parts[i]}){{{run}}}")
        else:
            compressed.append(f"{parts[i]}{{{run}}}")
        i = j
    return "^" + "".join(compressed) + "$"


def _shape_transform(values: pd.Series) -> pd.Series:
    return values.str.replace(r"[A-Za-z]", "A", regex=True).str.replace(r"\d", "9", regex=True)


def _shape_profile(
    series: pd.Series, value_counts: pd.Series | None = None
) -> dict[str, Any]:
    """Shape evidence: letters->A, digits->9, other characters kept.

    Counting is weighted by distinct-value counts when available. Adds
    display_shape (readable [A-Z]{n}-[0-9]{n} form) next to the raw shape.
    Skipped for free text (mean length above the cap).
    """
    empty: dict[str, Any] = {
        "dominant_shape": None, "dominant_display_shape": None,
        "dominant_coverage_percentage": 0.0, "distinct_shapes": 0,
        "top_shapes": [], "is_regular": False, "constant_length": False,
        "length_min": None, "length_max": None, "collapsed_shape": None,
        "suggested_regex": None,
    }
    if value_counts is not None and not value_counts.empty:
        distinct = pd.Series(value_counts.index.astype(str))
        weights = value_counts.to_numpy(dtype="int64")
        lengths = distinct.str.len()
        mean_length = float((lengths.to_numpy() * weights).sum() / weights.sum())
    else:
        if len(series) == 0 or not _is_text_like(series):
            return empty
        strings = _as_string_series(series).astype(str).reset_index(drop=True)
        if strings.empty:
            return empty
        distinct, weights = strings, np.ones(len(strings), dtype="int64")
        lengths = strings.str.len()
        mean_length = float(lengths.mean())

    profile: dict[str, Any] = {
        "length_min": int(lengths.min()), "length_max": int(lengths.max()),
        "constant_length": bool(lengths.min() == lengths.max()),
    }
    if mean_length > PROFILING_SHAPE_MAX_MEAN_LENGTH:
        return {**empty, **profile, "skipped_reason": "long_text"}

    sampled = False
    total_rows = int(weights.sum())
    if total_rows > PROFILING_SHAPE_SAMPLE_ROWS and value_counts is not None:
        rows = np.repeat(distinct.to_numpy(), weights)
        picked = np.random.RandomState(0).choice(len(rows), PROFILING_SHAPE_SAMPLE_ROWS, replace=False)
        distinct = pd.Series(rows[np.sort(picked)])
        weights = np.ones(len(distinct), dtype="int64")
        sampled = True
    elif total_rows > PROFILING_SHAPE_SAMPLE_ROWS:
        distinct = distinct.sample(n=PROFILING_SHAPE_SAMPLE_ROWS, random_state=0).reset_index(drop=True)
        weights = np.ones(len(distinct), dtype="int64")
        sampled = True

    shapes = _shape_transform(distinct).to_numpy()
    originals = distinct.to_numpy()
    unique_shapes, inverse = np.unique(shapes, return_inverse=True)
    counts = np.bincount(inverse.ravel(), weights=weights).astype("int64")
    order = sorted(range(len(unique_shapes)), key=lambda p: (-int(counts[p]), str(unique_shapes[p])))
    total = int(counts.sum())
    dominant = str(unique_shapes[order[0]])
    coverage = counts[order[0]] / total * 100 if total else 0.0

    top: list[dict[str, Any]] = []
    for position in order[:PROFILING_SHAPE_MAX_SHAPES]:
        shape = str(unique_shapes[position])
        members = originals[shapes == shape]
        top.append({
            "shape": shape,
            "display_shape": _display_shape(shape, [str(m) for m in members]),
            "percentage": _clean_value(counts[position] / total * 100 if total else 0.0),
            "count": int(counts[position]),
            "example": str(members[0]) if len(members) else None,
        })

    dominant_members = [str(m) for m in originals[shapes == dominant]]
    profile.update({
        "dominant_shape": dominant,
        "dominant_display_shape": _display_shape(dominant, dominant_members),
        "dominant_coverage_percentage": _clean_value(coverage),
        "distinct_shapes": int(len(unique_shapes)),
        "is_regular": bool(coverage >= PROFILING_SHAPE_REGULAR_THRESHOLD),
        "top_shapes": top,
        "collapsed_shape": _collapse_shape(dominant),
        "suggested_regex": None,
    })
    if profile["is_regular"]:
        profile["suggested_regex"] = _suggested_regex(
            pd.Series(dominant_members[:PROFILING_SHAPE_REGEX_MAX_VALUES])
        )
    profile["sampled"] = sampled
    if sampled:
        profile["sample_rows"] = int(min(PROFILING_SHAPE_SAMPLE_ROWS, total))
    return profile


# ---------------------------------------------------------------------------
# Dates
# ---------------------------------------------------------------------------


def _date_like_mask(text: pd.Series) -> pd.Series:
    """Mask of values matching ANY supported date shape."""
    return text.str.fullmatch(_combined_date_shape_regex(), na=False).fillna(False)


def _parse_with_format(values: pd.Series, fmt: str) -> pd.Series:
    """Explicit-format parse; an optional time suffix never breaks date-only formats."""
    parsed = pd.to_datetime(values, format=fmt, errors="coerce")
    failed = parsed.isna() & values.notna()
    if bool(failed.any()):
        stripped = values[failed].str.replace(
            r"[ T]\d{1,2}:\d{2}(?::\d{2})?(?:\.\d+)?$", "", regex=True
        )
        parsed.loc[failed] = pd.to_datetime(stripped, format=fmt, errors="coerce")
    return parsed


def _choose_date_format(text: pd.Series, parse_rates: dict[str, float]) -> tuple[str, float]:
    """Pick the date format; day/month ambiguity is decided only on evidence."""
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y%m%d"):
        if fmt in parse_rates:
            return fmt, 100.0
    for first_fmt, second_fmt in (
        ("%d/%m/%Y", "%m/%d/%Y"), ("%d-%m-%Y", "%m-%d-%Y"), ("%d.%m.%Y", "%m.%d.%Y"),
    ):
        first_rate = parse_rates.get(first_fmt, 0.0)
        second_rate = parse_rates.get(second_fmt, 0.0)
        if first_rate == 0.0 and second_rate == 0.0:
            continue
        parts = text.str.extract(r"^(\d{1,2})[-/.](\d{1,2})[-/.]", expand=True)
        first_part = pd.to_numeric(parts[0], errors="coerce")
        second_part = pd.to_numeric(parts[1], errors="coerce")
        if bool((first_part > 12).any()):
            return first_fmt, 100.0
        if bool((second_part > 12).any()):
            return second_fmt, 100.0
        if first_rate > 0.0 and second_rate > 0.0:
            return second_fmt, 50.0
        return (first_fmt, 100.0) if first_rate > 0.0 else (second_fmt, 100.0)
    best = max(parse_rates.items(), key=lambda item: (item[1], item[0]))
    return best[0], min(100.0, float(best[1]))


def _datetime_like_profile(series: pd.Series) -> dict[str, Any]:
    """Date-like string detection and parse-format inference.

    A column is date-like when >= 95% of non-null values parse under ONE
    explicit supported format. Day/month ambiguity is flagged, never
    silently guessed.
    """
    non_null = series.dropna()
    if non_null.empty:
        return {}
    text = non_null.astype(str).str.strip()

    # Cheap prefilter: most string columns are not dates.
    head = text.drop_duplicates().head(PROFILING_DATE_PREFILTER_DISTINCT)
    if float(_date_like_mask(head).mean()) < 0.5:
        return {}

    date_like_count = int(_date_like_mask(text).sum())
    if date_like_count == 0:
        return {}
    date_like_percentage = date_like_count / len(non_null) * 100
    if date_like_percentage < PROFILING_DATE_LIKE_THRESHOLD:
        return {}

    candidates = [fmt for fmt, _ in SUPPORTED_DATE_FORMATS]
    if _column_name_has_token(str(series.name or ""), COMPACT_DATE_NAME_TOKENS):
        if bool(text.str.fullmatch(r"\d{8}").all()):
            candidates.append("%Y%m%d")

    sample = text.drop_duplicates().head(PROFILING_DATE_SAMPLE_DISTINCT)
    rates: dict[str, float] = {}
    for fmt in candidates:
        rate = float(_parse_with_format(sample, fmt).notna().mean()) * 100.0
        if rate >= PROFILING_DATE_LIKE_THRESHOLD:
            rates[fmt] = rate
    if not rates:
        return {}

    chosen, confidence = _choose_date_format(text, rates)
    parsed = _parse_with_format(text, chosen)
    valid = parsed.dropna()
    if valid.empty:
        return {}

    has_time = bool(text.str.contains(r"[ T]\d{1,2}:\d{2}", regex=True, na=False)[parsed.notna()].any())
    now = pd.Timestamp.now(tz=valid.dt.tz) if valid.dt.tz is not None else pd.Timestamp.now()
    ambiguous_family = chosen in {"%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y", "%m-%d-%Y", "%d.%m.%Y"}
    valid_pct = len(valid) / len(non_null) * 100
    return {
        "count": int(len(valid)),
        "date_like_count": date_like_count,
        "date_like_percentage": _clean_value(date_like_percentage),
        "format_valid_percentage": _clean_value(valid_pct),
        "min": valid.min().isoformat(),
        "max": valid.max().isoformat(),
        "monotonic_increasing": bool(valid.is_monotonic_increasing),
        "monotonic_decreasing": bool(valid.is_monotonic_decreasing),
        "detected_format": chosen,
        "format_ambiguous": bool(ambiguous_family and confidence < 100.0),
        "format_confidence_percentage": _clean_value(confidence),
        "parse_success_percentage": _clean_value(valid_pct),
        "has_time_component": has_time,
        "future_date_percentage": _clean_value(float((valid > now).mean()) * 100.0),
        "span_days": _clean_value(float((valid.max() - valid.min()) / pd.Timedelta(days=1))),
        "distinct_dates": int(valid.nunique()),
        "null_or_unparseable_count": int(len(non_null) - len(valid)),
    }


def _datetime_profile(series: pd.Series) -> dict[str, Any]:
    """Profile for true datetime dtype columns."""
    valid = pd.to_datetime(series, errors="coerce").dropna()
    if valid.empty:
        return {}
    now = pd.Timestamp.now(tz=valid.dt.tz) if valid.dt.tz is not None else pd.Timestamp.now()
    return {
        "count": int(len(valid)),
        "min": valid.min().isoformat(),
        "max": valid.max().isoformat(),
        "format_valid_percentage": _clean_value(len(valid) / len(series) * 100),
        "monotonic_increasing": bool(valid.is_monotonic_increasing),
        "monotonic_decreasing": bool(valid.is_monotonic_decreasing),
        "detected_format": None,
        "has_time_component": bool((valid != valid.dt.normalize()).any()),
        "future_date_percentage": _clean_value(float((valid > now).mean()) * 100.0),
        "span_days": _clean_value(float((valid.max() - valid.min()) / pd.Timedelta(days=1))),
        "distinct_dates": int(valid.nunique()),
    }


# ---------------------------------------------------------------------------
# Numeric
# ---------------------------------------------------------------------------


def _max_decimal_places(numeric: pd.Series) -> int:
    finite = numeric[np.isfinite(numeric)]
    if finite.empty or bool((finite == finite.round()).all()):
        return 0
    for places in range(1, NUMERIC_MAX_DECIMAL_PLACES + 1):
        scaled = np.round(finite.to_numpy(dtype="float64") * 10.0**places, 6)
        if bool(np.all(np.abs(scaled - np.round(scaled)) < 1e-6)):
            return places
    return NUMERIC_MAX_DECIMAL_PLACES


def _numeric_profile(
    numeric: pd.Series, value_counts: pd.Series | None = None
) -> dict[str, Any]:
    """Statistical profile of a cleaned (non-null numeric) series."""
    if numeric.empty:
        return {}
    distinct = int(len(value_counts)) if value_counts is not None else int(numeric.nunique())
    is_constant = distinct <= 1
    top_share = 0.0
    if not is_constant:
        top_share = (
            float(value_counts.iloc[0] / len(numeric))
            if value_counts is not None and not value_counts.empty
            else float(numeric.value_counts().iloc[0] / len(numeric))
        )
    near_constant = not is_constant and top_share >= PROFILING_NEAR_CONSTANT_THRESHOLD
    q25, q50, q75 = (numeric.quantile(q) for q in (0.25, 0.5, 0.75))
    iqr = q75 - q25
    mad = (numeric - numeric.median()).abs().median()
    outliers = int(((numeric < q25 - 1.5 * iqr) | (numeric > q75 + 1.5 * iqr)).sum())
    finite = numeric[np.isfinite(numeric)]
    integer_valued = bool(len(finite) == len(numeric) and (finite == finite.round()).all())
    try:
        skewness = float(numeric.skew())
    except (ValueError, TypeError):
        skewness = None
    return {
        "count": int(len(numeric)),
        "min": _clean_value(numeric.min()), "max": _clean_value(numeric.max()),
        "mean": _clean_value(numeric.mean()), "median": _clean_value(numeric.median()),
        "std": _clean_value(numeric.std()),
        "q25": _clean_value(q25), "q50": _clean_value(q50), "q75": _clean_value(q75),
        "iqr": _clean_value(iqr), "mad": _clean_value(mad),
        "outlier_count_iqr": outliers,
        "outlier_percentage_iqr": _clean_value(outliers / len(numeric) * 100),
        "constant": bool(is_constant), "near_constant": bool(near_constant),
        "top_value_share_percentage": _clean_value(top_share * 100.0),
        "distinct_numeric_values": distinct,
        "zero_count": int((numeric == 0).sum()),
        "negative_count": int((numeric < 0).sum()),
        "positive_count": int((numeric > 0).sum()),
        "integer_valued": integer_valued,
        "max_decimal_places": _max_decimal_places(numeric),
        "p01": _clean_value(numeric.quantile(0.01)), "p05": _clean_value(numeric.quantile(0.05)),
        "p95": _clean_value(numeric.quantile(0.95)), "p99": _clean_value(numeric.quantile(0.99)),
        "skewness": _clean_value(skewness),
        "low_cardinality": bool(distinct <= PROFILING_LOW_CARDINALITY_MAX),
    }


def _integer_sequence_profile(numeric: pd.Series) -> dict[str, Any]:
    """Counter evidence: counter_like needs UNIQUE values in constant steps."""
    result = {"dense": False, "step_one": False, "counter_like": False}
    finite = numeric[np.isfinite(numeric)]
    distinct_count = int(finite.nunique()) if not finite.empty else 0
    result["low_cardinality"] = bool(distinct_count <= PROFILING_LOW_CARDINALITY_MAX)
    if finite.empty or not bool((finite == finite.round()).all()):
        return result
    values = np.unique(finite.to_numpy(dtype="float64"))
    if len(values) < 2:
        return result
    steps = np.diff(values)
    constant_step = bool(np.all(steps == steps[0]))
    unique_all = bool(len(values) == len(finite))
    result["step_one"] = bool(constant_step and steps[0] == 1)
    result["dense"] = bool(unique_all and result["step_one"])
    result["counter_like"] = bool(unique_all and constant_step and values[0] >= 0)
    return result


def _numeric_code_like(name: Any, block: dict[str, Any], sequence: dict[str, Any],
                       identifier_signal: bool) -> bool:
    """code_like = code name token OR integer counter/identifier evidence."""
    if _column_name_has_token(name, CODE_LIKE_NAME_TOKENS):
        return True
    return bool(block.get("integer_valued") and (sequence.get("counter_like") or identifier_signal))


def _sentinel_profile(value_counts: pd.Series | None, non_null: int) -> dict[str, Any]:
    if value_counts is None or value_counts.empty or non_null == 0:
        return {"suspicious_sentinel": None}
    try:
        top = float(value_counts.index[0])
    except (TypeError, ValueError):
        return {"suspicious_sentinel": None}
    share = int(value_counts.iloc[0]) / non_null
    if top.is_integer() and int(top) in NUMERIC_SENTINEL_VALUES and share >= NUMERIC_SENTINEL_MIN_SHARE:
        return {"suspicious_sentinel": {"value": int(top), "percentage": _clean_value(share * 100.0)}}
    return {"suspicious_sentinel": None}


def _leading_zero_profile(numeric: pd.Series, code_like: bool) -> dict[str, Any]:
    """Suspect lost leading zeros: modal digit length >= 80%, shorter values exist."""
    if not code_like:
        return {}
    finite = numeric[np.isfinite(numeric)]
    if finite.empty or not bool((finite == finite.round()).all()):
        return {}
    lengths = np.char.str_len(np.abs(finite.to_numpy(dtype="int64")).astype(str))
    unique_lengths, counts = np.unique(lengths, return_counts=True)
    order = np.argsort(-counts, kind="stable")
    modal_length, modal_count = int(unique_lengths[order[0]]), int(counts[order[0]])
    shorter = int((lengths < modal_length).sum())
    suspected = bool(modal_count / len(lengths) >= 0.8 and shorter > 0 and modal_length > 1)
    return {
        "leading_zero_loss_suspected": suspected,
        "leading_zero_loss_count": shorter if suspected else 0,
        "digit_length_distribution": {str(int(unique_lengths[i])): int(counts[i]) for i in order[:5]},
    }


def _numeric_shape_profile(numeric: pd.Series) -> dict[str, Any]:
    finite = numeric[np.isfinite(numeric)]
    if finite.empty or not bool((finite == finite.round()).all()):
        return {}
    return _shape_profile(finite.abs().astype("int64").astype(str))


# ---------------------------------------------------------------------------
# Text hygiene
# ---------------------------------------------------------------------------


def _disguised_missing_profile(value_counts: pd.Series, non_null: int, row_count: int) -> dict[str, Any]:
    """Disguised-missing tokens, counted by rows; tokens always reported."""
    base = {"disguised_missing_count": 0, "disguised_missing_percentage": 0.0,
            "disguised_missing_values": []}
    if value_counts is None or value_counts.empty:
        return base
    lowered = value_counts.index.to_series().astype(str).str.strip().str.lower().to_numpy()
    counts = value_counts.to_numpy(dtype="int64")
    found: list[tuple[str, int]] = []
    for token in DISGUISED_MISSING_TOKENS:
        hit = int(counts[lowered == token].sum())
        if hit > 0 and (row_count == 0 or hit / row_count < DISGUISED_MISSING_MAX_SHARE):
            found.append((token, hit))
    if not found:
        return base
    total = sum(c for _, c in found)
    ordered = sorted(found, key=lambda item: (-item[1], item[0]))[:DISGUISED_MISSING_TOP_TOKENS]
    return {
        "disguised_missing_count": int(total),
        "disguised_missing_percentage": _clean_value(total / non_null * 100 if non_null else 0.0),
        "disguised_missing_values": [{"value": t, "count": c} for t, c in ordered],
    }


def _case_variant_profile(value_counts: pd.Series, distinct_count: int) -> dict[str, Any]:
    """Case/whitespace variants of the same normalised value."""
    if distinct_count > PROFILING_CASE_VARIANT_MAX_DISTINCT:
        return {"normalized_distinct_count": None, "case_variant_groups": None,
                "case_variant_examples": [], "skipped_reason": "too_many_distinct_values"}
    raw = value_counts.index.to_series().astype(str).reset_index(drop=True)
    stripped = raw.str.strip().str.replace(r"\s+", " ", regex=True)
    norm = stripped.str.casefold()
    frame = pd.DataFrame({"raw": stripped, "norm": norm}).drop_duplicates()
    sizes = frame.groupby("norm", sort=True)["raw"].nunique()
    variant_keys = sizes[sizes > 1].index
    groups = [frame.loc[frame["norm"] == key, "raw"].tolist() for key in variant_keys]
    groups.sort(key=lambda g: (-len(g), g[0]))
    return {
        "normalized_distinct_count": int(norm.nunique()),
        "case_variant_groups": len(groups),
        "case_variant_examples": [g[:PROFILING_CASE_VARIANT_EXAMPLES]
                                  for g in groups[:PROFILING_CASE_VARIANT_EXAMPLES]],
    }


def _categorical_profile(non_null_count: int, distinct_count: int, value_counts: pd.Series) -> dict[str, Any]:
    if value_counts is None or value_counts.empty:
        return {}
    top = [
        {"value": _clean_value(value), "count": int(count),
         "percentage": _clean_value(int(count) / non_null_count * 100 if non_null_count else 0.0)}
        for value, count in value_counts.head(PROFILING_TOP_CATEGORIES).items()
    ]
    return {"category_count": distinct_count, "top_values": top, "top_values_returned": len(top)}


def _identifier_like_text(text_block: dict[str, Any], shape: dict[str, Any],
                          name_signal: bool, unique_signal: bool) -> tuple[bool, list[str]]:
    """Identifier-likeness for text WITHOUT requiring uniqueness."""
    mean_length = text_block.get("mean_length")
    if isinstance(mean_length, (int, float)) and mean_length > PROFILING_SHAPE_MAX_MEAN_LENGTH:
        return False, []
    reasons: list[str] = []
    if unique_signal:
        reasons.append("unique_with_name_signal")
    if name_signal:
        reasons.append("name_signal")
    if not reasons and shape and not shape.get("skipped_reason") and shape.get("is_regular"):
        dominant = shape.get("dominant_shape") or ""
        spread = 0
        if shape.get("length_min") is not None and shape.get("length_max") is not None:
            spread = int(shape["length_max"]) - int(shape["length_min"])
        if "9" in dominant and (shape.get("constant_length") or spread <= PROFILING_IDENTIFIER_MAX_LENGTH_RANGE):
            reasons.append("regular_shape_with_digit")
    elif reasons and shape and shape.get("is_regular") and not shape.get("skipped_reason"):
        reasons.append("regular_shape")
    return bool(reasons), reasons


# ---------------------------------------------------------------------------
# Composite uniqueness (factorised int64 codes)
# ---------------------------------------------------------------------------


def _factorize(series: pd.Series) -> tuple[np.ndarray, int]:
    """int64 codes (-1 for null) and the number of categories.

    Hashable values use normal pandas factorization. If cells contain
    unhashable values such as lists or dictionaries, convert them to a
    deterministic hashable representation before factorization.
    """
    try:
        codes, uniques = pd.factorize(series, sort=False)
        return np.asarray(codes, dtype=np.int64), int(len(uniques))
    except TypeError:
        def is_null(value: Any) -> bool:
            if value is None:
                return True
            try:
                result = pd.isna(value)
                return bool(result) if not isinstance(result, (list, tuple, np.ndarray)) else False
            except (TypeError, ValueError):
                return False

        def canonicalize(value: Any) -> Any:
            if isinstance(value, dict):
                return (
                    "dict",
                    tuple(
                        sorted(
                            (
                                canonicalize(key),
                                canonicalize(item),
                            )
                            for key, item in value.items()
                        )
                    ),
                )
            if isinstance(value, (list, tuple)):
                return ("sequence", tuple(canonicalize(item) for item in value))
            if isinstance(value, set):
                return (
                    "set",
                    tuple(sorted(canonicalize(item) for item in value)),
                )
            return value

        normalized = series.map(
            lambda value: None if is_null(value) else canonicalize(value)
        )

        codes, uniques = pd.factorize(normalized, sort=False)
        return np.asarray(codes, dtype=np.int64), int(len(uniques))

def _composite_uniqueness_candidates(
    dataframe: pd.DataFrame,
    column_distinct: dict[str, int],
    column_non_null: dict[str, int],
    row_count: int,
    column_meta: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Composite uniqueness candidates (observations, never business keys).

    1. Eligible members: >= 2 distinct, not unique alone, not constant, not
       a float measure, not free text.
    2. Pigeonhole pruning on the product of distinct counts.
    3. Key-like combinations are evaluated first (the evaluation cap can
       never hide a true key behind random pairs).
    4. Minimality: a combination is skipped when a strict subset is already
       a candidate.
    Ranking: near-exact first, fewer columns, key-likeness, uniqueness.
    Rows with any NULL member are excluded; evaluated_row_count is reported.
    Deterministic sample above PROFILING_COMPOSITE_SAMPLE_ROWS.
    """
    meta = column_meta or {}
    if row_count == 0:
        return []
    work = dataframe
    sampled = False
    if row_count > PROFILING_COMPOSITE_SAMPLE_ROWS:
        work = dataframe.sample(n=PROFILING_COMPOSITE_SAMPLE_ROWS, random_state=0)
        sampled, row_count = True, len(work)

    bar = PROFILING_COMPOSITE_PIGEONHOLE_SHARE * row_count
    unique_bar = PROFILING_COMPOSITE_UNIQUENESS_THRESHOLD / 100.0
    eligible: dict[str, int] = {}
    for column, distinct in column_distinct.items():
        info = meta.get(column, {})
        if info.get("is_constant") or info.get("is_float_measure") or info.get("is_free_text"):
            continue
        non_null = column_non_null.get(column, 0)
        if distinct < 2 or non_null == 0 or distinct / non_null >= unique_bar:
            continue
        eligible[column] = distinct

    codes: dict[str, np.ndarray] = {}
    sizes: dict[str, int] = {}
    for column in eligible:
        codes[column], sizes[column] = _factorize(work[column])

    def priority(combo: tuple[str, ...]) -> tuple[int, int]:
        like = sum(1 for c in combo if meta.get(c, {}).get("is_key_like"))
        return (-like, math.prod(eligible[c] for c in combo))

    candidates: list[dict[str, Any]] = []
    unique_subsets: set[frozenset[str]] = set()
    evaluated = 0
    for size in range(2, min(PROFILING_MAX_COMBO_COLUMNS, len(eligible)) + 1):
        combos = [c for c in combinations(eligible.keys(), size)
                  if math.prod(eligible[x] for x in c) >= bar]
        combos.sort(key=priority)
        for combo in combos:
            if evaluated >= PROFILING_MAX_COMPOSITE_EVALUATIONS:
                return _rank_composite_candidates(candidates)
            if any(frozenset(sub) in unique_subsets for sub in combinations(combo, size - 1)):
                continue
            valid = np.ones(len(work), dtype=bool)
            for column in combo:
                valid &= codes[column] >= 0
            non_null = int(valid.sum())
            if non_null == 0 or non_null < PROFILING_COMPOSITE_MIN_ROW_COVERAGE * row_count:
                continue
            evaluated += 1
            key = codes[combo[0]][valid]
            for column in combo[1:]:
                key = pd.factorize(key * sizes[column] + codes[column][valid])[0].astype(np.int64)
            counts = np.bincount(pd.factorize(key)[0])
            distinct_count = int(len(counts))
            pct = distinct_count / non_null * 100
            if pct < PROFILING_COMPOSITE_UNIQUENESS_THRESHOLD:
                continue
            entry = {
                "columns": [meta.get(c, {}).get("report_name", c) for c in combo],
                "evaluated_row_count": non_null,
                "composite_distinct_count": distinct_count,
                "composite_uniqueness_percentage": _clean_value(pct),
                "duplicate_composite_rows": int(counts[counts > 1].sum()),
                "duplicate_composite_excess_count": int(non_null - distinct_count),
                "key_like_members": [meta.get(c, {}).get("report_name", c) for c in combo
                                     if meta.get(c, {}).get("is_key_like")],
                "contains_measure": False,
                "near_exact": bool(pct >= PROFILING_COMPOSITE_NEAR_EXACT_THRESHOLD),
            }
            if sampled:
                entry.update({"sampled": True, "sample_rows": int(row_count), "claim": "sample"})
            candidates.append(entry)
            unique_subsets.add(frozenset(combo))
    return _rank_composite_candidates(candidates)


def _rank_composite_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Near-exact first, then fewer columns, more key-like members, higher uniqueness."""
    ranked = sorted(
        candidates,
        key=lambda e: (
            not e["near_exact"], len(e["columns"]), -len(e.get("key_like_members", [])),
            -e["composite_uniqueness_percentage"], -e["composite_distinct_count"],
        ),
    )
    return ranked[:PROFILING_MAX_COMBINATIONS]


# ---------------------------------------------------------------------------
# Row completeness
# ---------------------------------------------------------------------------


def _row_completeness(dataframe: pd.DataFrame) -> dict[str, Any]:
    """Row-level completeness and co-missing patterns (nulls = pd.isna)."""
    row_count, column_count = len(dataframe), len(dataframe.columns)
    if row_count == 0 or column_count == 0:
        return {"row_count": row_count, "fully_complete_rows": 0, "fully_complete_percentage": 0.0,
                "rows_with_any_null": 0, "average_filled_percentage_per_row": 0.0,
                "emptiest_row_filled_percentage": 0.0, "co_missing_patterns": []}
    null_mask = dataframe.isna()
    filled = (~null_mask).sum(axis=1)
    complete = int((filled == column_count).sum())
    return {
        "row_count": row_count,
        "fully_complete_rows": complete,
        "fully_complete_percentage": _clean_value(complete / row_count * 100),
        "rows_with_any_null": int(row_count - complete),
        "average_filled_percentage_per_row": _clean_value(float(filled.mean() / column_count) * 100),
        "emptiest_row_filled_percentage": _clean_value(int(filled.min()) / column_count * 100),
        "co_missing_patterns": _co_missing_patterns(null_mask),
    }


def _co_missing_patterns(null_mask: pd.DataFrame) -> list[dict[str, Any]]:
    """Top groups of >= 2 columns that are null together (bounded)."""
    null_counts = null_mask.sum()
    names = [c for c in null_counts.index if null_counts[c] > 0]
    if len(names) < 2:
        return []
    names.sort(key=lambda c: (-int(null_counts[c]), str(c)))
    selected = names[:PROFILING_CO_MISSING_MAX_COLUMNS]
    sub = null_mask[selected].to_numpy(dtype=bool)
    bits = min(len(selected), PROFILING_PACKED_MASK_BITS)
    packed = np.zeros(len(sub), dtype=np.uint64)
    for bit in range(bits):
        packed |= sub[:, bit].astype(np.uint64) << np.uint64(bit)
    if len(selected) > bits:  # very wide: fold the remainder into a hash
        packed ^= pd.util.hash_array(np.ascontiguousarray(sub[:, bits:]).view(np.uint8).reshape(len(sub), -1).sum(axis=1)).astype(np.uint64)
    unique_masks, first_index, counts = np.unique(packed, return_index=True, return_counts=True)
    patterns: list[dict[str, Any]] = []
    for position, mask in enumerate(unique_masks):
        row = sub[first_index[position]]
        if int(row.sum()) < 2:
            continue
        patterns.append({"columns": [str(selected[i]) for i in np.flatnonzero(row)],
                         "row_count": int(counts[position])})
    patterns.sort(key=lambda p: (-p["row_count"], p["columns"]))
    return patterns[:PROFILING_CO_MISSING_MAX_PATTERNS]


# ---------------------------------------------------------------------------
# Functional dependencies (row-weighted coverage, O(n) per pair)
# ---------------------------------------------------------------------------


def _fd_coverage(ca: np.ndarray, cb: np.ndarray, n_a: int, n_b: int) -> tuple[float, int, int, int]:
    """(coverage %, violating groups, violating rows, evaluated rows) for A -> B."""
    valid = (ca >= 0) & (cb >= 0)
    evaluated = int(valid.sum())
    if evaluated == 0:
        return 0.0, 0, 0, 0
    a, b = ca[valid], cb[valid]
    rows_per_a = np.bincount(a, minlength=n_a)
    pairs = np.unique(a * n_b + b)
    distinct_b_per_a = np.bincount(pairs // n_b, minlength=n_a)
    violating = np.flatnonzero(distinct_b_per_a > 1)
    violating_rows = int(rows_per_a[violating].sum())
    return 100.0 * (1.0 - violating_rows / evaluated), int(len(violating)), violating_rows, evaluated


def _functional_dependencies(
    dataframe: pd.DataFrame,
    column_meta: dict[str, dict[str, Any]],
    column_distinct: dict[str, int],
    column_non_null: dict[str, int],
) -> dict[str, Any]:
    """Within-table dependencies A -> B (OBSERVATIONS on this data, never rules).

    Coverage = 100 * (1 - rows_in_violating_groups / evaluated_rows); a
    violating group is an A value mapped to more than one B value.
    """
    row_count = len(dataframe)
    empty = {"evaluated_rows": 0, "sampled": False, "dependencies": [], "skipped_reason": None}
    if row_count == 0:
        return empty
    work, sampled = dataframe, False
    if row_count > PROFILING_FD_SAMPLE_ROWS:
        work = dataframe.sample(n=PROFILING_FD_SAMPLE_ROWS, random_state=0)
        sampled, row_count = True, len(work)

    ratio: dict[str, float] = {}
    for column, info in column_meta.items():
        distinct, non_null = column_distinct.get(column, 0), column_non_null.get(column, 0)
        if info.get("is_constant") or info.get("is_free_text") or info.get("is_float_measure"):
            continue
        if non_null == 0 or distinct < 2:
            continue
        ratio[column] = distinct / non_null
    determinants = sorted((c for c, r in ratio.items() if r <= PROFILING_FD_UNIQUE_RATIO),
                          key=lambda c: (ratio[c], c))[:PROFILING_FD_MAX_COLUMNS]
    codes: dict[str, np.ndarray] = {}
    sizes: dict[str, int] = {}
    for column in determinants:
        codes[column], sizes[column] = _factorize(work[column])

    budget = PROFILING_FD_MAX_PAIRS
    found: dict[tuple[str, str], tuple[float, int, int, int]] = {}
    for a_col in determinants:
        for b_col in determinants:
            if a_col == b_col:
                continue
            if budget <= 0:
                break
            budget -= 1
            result = _fd_coverage(codes[a_col], codes[b_col], sizes[a_col], sizes[b_col])
            if result[3] and result[0] >= PROFILING_FD_MIN_COVERAGE:
                found[(a_col, b_col)] = result

    entries: list[dict[str, Any]] = []
    for (a_col, b_col), (coverage, groups, rows, evaluated) in found.items():
        reverse = found.get((b_col, a_col))
        entries.append({
            "determinant": a_col, "dependent": b_col,
            "coverage_percentage": _clean_value(coverage),
            "violating_groups": groups, "violating_rows": rows,
            "bidirectional": bool(reverse is not None and reverse[0] >= PROFILING_FD_MIN_COVERAGE),
            "violation_examples": [], "evaluated_rows": evaluated, "sampled": sampled,
        })
    entries = _reduce_fd_entries(entries)
    entries.sort(key=lambda e: (-e["coverage_percentage"], -ratio.get(e["determinant"], 0.0),
                                e["determinant"], e["dependent"]))
    entries = entries[:PROFILING_FD_MAX_REPORTED]
    for entry in entries:
        if entry["coverage_percentage"] < 100.0:
            entry["violation_examples"] = _fd_violation_examples(work, entry["determinant"], entry["dependent"])
        entry["determinant"] = column_meta[entry["determinant"]].get("report_name", entry["determinant"])
        entry["dependent"] = column_meta[entry["dependent"]].get("report_name", entry["dependent"])
    return {"evaluated_rows": int(row_count), "sampled": sampled, "dependencies": entries,
            "skipped_reason": None}


def _fd_violation_examples(frame: pd.DataFrame, determinant: str, dependent: str) -> list[dict[str, Any]]:
    """First violating groups for A -> B (max 3, deterministic)."""
    subset = frame[[determinant, dependent]].dropna()
    if subset.empty:
        return []
    distinct_b = subset.groupby(determinant, sort=True)[dependent].nunique()
    examples: list[dict[str, Any]] = []
    for value in distinct_b[distinct_b > 1].index[:PROFILING_MAX_VIOLATION_EXAMPLES]:
        dependents = subset.loc[subset[determinant] == value, dependent].unique()
        examples.append({"determinant_value": _clean_value(value),
                         "dependent_values": [_clean_value(v) for v in dependents][:PROFILING_MAX_VIOLATION_EXAMPLES]})
    return examples


def _reduce_fd_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop A->C when exact A->B and B->C exist (and C->B does not)."""
    exact = {(e["determinant"], e["dependent"]) for e in entries if e["coverage_percentage"] >= 100.0}
    nodes = {n for pair in exact for n in pair}
    kept = []
    for entry in entries:
        if entry["coverage_percentage"] >= 100.0:
            a, c = entry["determinant"], entry["dependent"]
            if any((a, b) in exact and (b, c) in exact and (c, b) not in exact
                   for b in nodes if b not in (a, c)):
                continue
        kept.append(entry)
    return kept


# ---------------------------------------------------------------------------
# Observations
# ---------------------------------------------------------------------------


def _observations(columns: list[dict[str, Any]], table: dict[str, Any]) -> list[dict[str, Any]]:
    """Deterministic, evidence-carrying observations from computed numbers."""
    out: list[dict[str, Any]] = []

    def add(severity: str, code: str, column: str | None, message: str, evidence: dict[str, Any]) -> None:
        out.append({"severity": severity, "code": code, "column": column,
                    "message": message, "evidence": evidence})

    dependencies = table.get("functional_dependencies", {}).get("dependencies", [])
    violated = sum(1 for d in dependencies if d.get("coverage_percentage", 100.0) < 100.0)
    if violated:
        add("warning", "dependency_violations", None,
            f"{violated} functional dependency pair(s) have violating rows on this data",
            {"pairs_with_violations": violated})
    composites = table.get("composite_uniqueness_candidates", [])
    if composites:
        best = composites[0]
        add("info", "best_composite_key", None,
            f"{' + '.join(best['columns'])}: {best['composite_uniqueness_percentage']:.3f}% unique composite",
            {"columns": best["columns"], "composite_uniqueness_percentage": best["composite_uniqueness_percentage"],
             "near_exact": best.get("near_exact", False)})
    if table.get("complete_duplicate_rows"):
        add("warning", "duplicate_rows", None, f"{table['complete_duplicate_rows']} complete duplicate rows",
            {"complete_duplicate_rows": table["complete_duplicate_rows"],
             "complete_duplicate_excess_count": table.get("complete_duplicate_excess_count", 0)})
    pct = table.get("row_completeness", {}).get("fully_complete_percentage")
    if pct is not None and pct < 90.0:
        add("warning", "row_completeness_low", None, f"only {pct:.1f}% of rows are fully complete",
            {"fully_complete_percentage": pct,
             "rows_with_any_null": table["row_completeness"].get("rows_with_any_null")})

    for col in columns:
        name = col["column_name"]
        numeric, text = col.get("numeric") or {}, col.get("text") or {}
        dt = col.get("datetime") or {}
        shape = col.get("shape") or text.get("shape") or {}
        if dt.get("detected_format"):
            add("info", "date_format_detected", name,
                f"{name}: {dt.get('parse_success_percentage', 0):.1f}% parse as {dt['detected_format']}",
                {"detected_format": dt["detected_format"], "parse_success_percentage": dt.get("parse_success_percentage"),
                 "format_ambiguous": dt.get("format_ambiguous", False)})
        if dt.get("format_ambiguous"):
            add("warning", "date_format_ambiguous", name,
                f"{name}: day/month order is ambiguous ({dt.get('detected_format')}) - confirm the real format",
                {"detected_format": dt.get("detected_format"),
                 "format_confidence_percentage": dt.get("format_confidence_percentage")})
        if numeric.get("constant") or text.get("constant"):
            values = (col.get("categorical") or {}).get("top_values") or []
            add("info", "constant_column", name, f"{name}: every non-null value is identical",
                {"value": values[0].get("value") if values else numeric.get("min")})
        near = numeric.get("near_constant") or text.get("near_constant")
        if near:
            share = numeric.get("top_value_share_percentage") if numeric.get("near_constant") \
                else text.get("top_value_share_percentage")
            add("warning", "near_constant_column", name, f"{name}: one value covers {share:.1f}% of non-null rows",
                {"top_value_share_percentage": share})
        if numeric.get("leading_zero_loss_suspected"):
            add("warning", "leading_zero_loss_suspected", name,
                f"{name}: {numeric.get('leading_zero_loss_count', 0)} value(s) shorter than the modal digit length - possible lost leading zeros",
                {"leading_zero_loss_count": numeric.get("leading_zero_loss_count"),
                 "digit_length_distribution": numeric.get("digit_length_distribution")})
        elif col.get("leading_zero_loss_suspected"):
            add("warning", "leading_zero_loss_suspected", name,
                f"{name}: values with leading zeros sit next to shorter values of the same family - keep it as text",
                {"evidence": "leading-zero strings observed"})
        if numeric.get("code_like"):
            add("info", "code_like_numeric", name,
                f"{name}: numeric column is code-like - statistics are not meaningful",
                {"meaningful_statistics": False, "integer_valued": numeric.get("integer_valued")})
        hygiene = text.get("disguised_missing") or {}
        if hygiene.get("disguised_missing_count"):
            tokens = [t["value"] for t in hygiene.get("disguised_missing_values", [])]
            add("warning", "disguised_missing_values", name,
                f"{name}: {hygiene['disguised_missing_count']} disguised missing value(s) ({', '.join(tokens)})",
                {"disguised_missing_count": hygiene["disguised_missing_count"],
                 "disguised_missing_values": hygiene.get("disguised_missing_values")})
        if text.get("case_variant_groups"):
            add("warning", "case_variants", name, f"{name}: {text['case_variant_groups']} case-variant group(s)",
                {"case_variant_groups": text["case_variant_groups"],
                 "case_variant_examples": text.get("case_variant_examples")})
        if col.get("leading_trailing_whitespace_count"):
            add("warning", "whitespace_padding", name,
                f"{name}: {col['leading_trailing_whitespace_count']} value(s) with leading/trailing whitespace",
                {"leading_trailing_whitespace_count": col["leading_trailing_whitespace_count"]})
        sequence = numeric.get("integer_sequence") or {}
        if sequence.get("counter_like"):
            add("info", "dense_integer_counter", name,
                f"{name}: unique integers in a constant-step sequence (surrogate-counter evidence)",
                {"dense": sequence.get("dense"), "step_one": sequence.get("step_one")})
        if col.get("identifier_repeats"):
            add("info", "identifier_repeats", name,
                f"{name}: identifier-like column with repeated values ({col.get('distinct_percentage', 0):.1f}% distinct)",
                {"distinct_percentage": col.get("distinct_percentage"),
                 "identifier_like_reasons": col.get("identifier_like_reasons")})
        if shape and shape.get("dominant_shape") and not dt.get("detected_format"):
            if shape.get("is_regular"):
                add("info", "regular_pattern", name,
                    f"{name}: {shape.get('dominant_coverage_percentage', 0):.1f}% of values match {shape.get('dominant_display_shape') or shape.get('collapsed_shape')}",
                    {"dominant_shape": shape.get("dominant_shape"),
                     "dominant_display_shape": shape.get("dominant_display_shape"),
                     "dominant_coverage_percentage": shape.get("dominant_coverage_percentage"),
                     "suggested_regex": shape.get("suggested_regex")})
            elif col.get("identifier_like"):
                add("warning", "irregular_pattern", name,
                    f"{name}: identifier-like column is irregular (dominant shape covers {shape.get('dominant_coverage_percentage', 0):.1f}%)",
                    {"dominant_shape": shape.get("dominant_shape"),
                     "dominant_coverage_percentage": shape.get("dominant_coverage_percentage")})

    sampled_blocks = [
        {"column": c["column_name"], "analysis": block, "sample_rows": (c.get(block) or {}).get("sample_rows")}
        for c in columns for block in ("shape",) if (c.get(block) or {}).get("sampled")
    ] + [
        {"column": c["column_name"], "analysis": "text.shape", "sample_rows": c["text"]["shape"].get("sample_rows")}
        for c in columns if (c.get("text") or {}).get("shape", {}).get("sampled")
    ]
    composite_sampled = any(c.get("sampled") for c in composites)
    fd_sampled = bool(table.get("functional_dependencies", {}).get("sampled"))
    if sampled_blocks or composite_sampled or fd_sampled:
        add("info", "sampled_analysis", None, "part of this profile was computed on deterministic samples",
            {"columns": sampled_blocks, "composite_sampled": composite_sampled,
             "functional_dependencies_sampled": fd_sampled})
    return out[:PROFILING_MAX_OBSERVATIONS]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def profile_dataframe(dataframe: pd.DataFrame, table_name: str | None = None) -> dict[str, Any]:
    """Reusable profile artifact for a dataframe (profiling only: no rules)."""
    row_count, column_count = len(dataframe), len(dataframe.columns)
    columns: list[dict[str, Any]] = []
    column_distinct: dict[str, int] = {}
    column_non_null: dict[str, int] = {}
    column_meta: dict[str, dict[str, Any]] = {}
    unique_keys: list[str] = []
    seen: Counter = Counter()

    for position, column in enumerate(dataframe.columns):
        series = dataframe.iloc[:, position]
        name_key = str(column)
        seen[name_key] += 1
        unique_key = name_key if seen[name_key] == 1 else f"{name_key}__pos{position}"
        unique_keys.append(unique_key)

        null_count = int(series.isna().sum())
        empty_count, blank_count = _vectorised_string_counts(series)
        value_counts, unhashable = _value_counts_safe(series)
        distinct_count = int(len(value_counts)) if value_counts is not None else int(series.nunique(dropna=True))
        non_null_count = int(series.notna().sum())
        duplicate_count = int(value_counts[value_counts > 1].sum()) if value_counts is not None and not value_counts.empty else 0
        distinct_pct = distinct_count / non_null_count * 100 if non_null_count else 0.0
        name_signal = identifier_name_signal_from_name(column)
        identifier_signal = bool(non_null_count > 0 and distinct_count == non_null_count and name_signal)

        profile: dict[str, Any] = {
            "column_name": name_key,
            "data_type": str(series.dtype),
            "row_count": row_count,
            "null_count": null_count,
            "null_percentage": null_count / row_count * 100 if row_count else 0.0,
            "empty_string_count": empty_count,
            "empty_string_percentage": empty_count / row_count * 100 if row_count else 0.0,
            "whitespace_only_count": blank_count,
            "whitespace_only_percentage": blank_count / row_count * 100 if row_count else 0.0,
            "distinct_count": distinct_count,
            "distinct_percentage": _clean_value(distinct_pct),
            "duplicate_count": duplicate_count,
            "duplicate_excess_count": int(non_null_count - distinct_count) if duplicate_count else 0,
            "identifier_name_signal": name_signal,
            "identifier_completeness_percentage": _clean_value(non_null_count / row_count * 100 if row_count else 0.0),
            "identifier_uniqueness_percentage": _clean_value(distinct_pct),
            "identifier_signal": identifier_signal,
        }
        if unhashable:
            profile["unhashable_values"] = True

        numeric_block: dict[str, Any] = {}
        if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            clean = pd.to_numeric(series, errors="coerce").dropna()
            numeric_block = _numeric_profile(clean, value_counts)
            if numeric_block:
                sequence = _integer_sequence_profile(clean)
                numeric_block["integer_sequence"] = sequence
                code_like = _numeric_code_like(column, numeric_block, sequence, identifier_signal)
                numeric_block["code_like"] = code_like
                numeric_block["meaningful_statistics"] = not code_like
                numeric_block.update(_sentinel_profile(value_counts, non_null_count))
                numeric_block.update(_leading_zero_profile(clean, code_like))
                is_integer = bool(numeric_block.get("integer_valued"))
                is_const = bool(numeric_block.get("constant"))
                id_like = bool(is_integer and not is_const
                               and (name_signal or sequence.get("counter_like")))
                reasons = ([] if not id_like else
                           (["name_signal"] if name_signal else []) + (["counter_like"] if sequence.get("counter_like") else []))
                profile["identifier_like"] = id_like
                profile["identifier_like_reasons"] = reasons
                profile["identifier_repeats"] = bool(id_like and distinct_pct < 100.0)
                if code_like and not is_const:
                    try:
                        shape = _numeric_shape_profile(clean)
                        if shape:
                            profile["shape"] = shape
                    except Exception as error:  # noqa: BLE001
                        profile["shape_error"] = _error_message(error)
                if numeric_block.get("leading_zero_loss_suspected"):
                    profile["leading_zero_loss_suspected"] = True
                profile["numeric"] = numeric_block
            profile["stored_as"] = "numeric"
            profile["semantic_storage_mismatch"] = bool(numeric_block.get("code_like", False))

        if _is_text_like(series):
            strings = _as_string_series(series).astype(str)
            text_block: dict[str, Any] = {}
            if not strings.empty:
                lengths = strings.str.len()
                mean_length = float(lengths.mean())
                text_block = {
                    "min_length": int(lengths.min()), "max_length": int(lengths.max()),
                    "mean_length": _clean_value(mean_length), "median_length": _clean_value(lengths.median()),
                    "patterns": _pattern_summary(series, value_counts if (value_counts is not None and not unhashable and len(value_counts) <= 0.5 * max(non_null_count, 1)) else None),
                    "separator_characters": _separator_characters(series, value_counts=value_counts),
                }
                try:
                    shape = _shape_profile(series, value_counts=value_counts)
                except Exception as error:  # noqa: BLE001
                    shape = {}
                    profile["shape_error"] = _error_message(error)
                text_block["shape"] = shape
                if value_counts is not None and not value_counts.empty:
                    hygiene = _disguised_missing_profile(value_counts, non_null_count, row_count)
                    variants = _case_variant_profile(value_counts, distinct_count)
                    hygiene.update(variants)
                    text_block["disguised_missing"] = hygiene
                    for key in ("normalized_distinct_count", "case_variant_groups", "case_variant_examples"):
                        text_block[key] = variants.get(key)
                    if variants.get("skipped_reason"):
                        text_block["skipped_reason"] = variants["skipped_reason"]
                    top_share = _top_value_share(value_counts, non_null_count)
                    text_block["top_value_share_percentage"] = _clean_value(top_share * 100.0)
                    text_block["constant"] = bool(distinct_count <= 1)
                    text_block["near_constant"] = bool(distinct_count > 1 and top_share >= PROFILING_NEAR_CONSTANT_THRESHOLD)
                    if _column_name_has_token(column, CODE_LIKE_NAME_TOKENS):
                        raw = pd.Series(value_counts.index.astype(str)).str.strip()
                        if bool(raw.str.fullmatch(r"0\d+", na=False).any()):
                            profile["leading_zero_loss_suspected"] = True
                padded = (strings != strings.str.strip()) & (strings.str.strip() != "")
                profile["leading_trailing_whitespace_count"] = int(padded.sum())
                like, reasons = _identifier_like_text(text_block, shape, name_signal, identifier_signal)
                profile["identifier_like"] = like
                profile["identifier_like_reasons"] = reasons
                profile["identifier_repeats"] = bool(like and distinct_pct < 100.0)
                profile["stored_as"] = "string"
                numeric_like = text_block["patterns"].get("numeric_like", 0)
                profile["semantic_storage_mismatch"] = bool(len(strings) > 0 and numeric_like == len(strings))
                profile["text"] = text_block
            try:
                datetime_like = _datetime_like_profile(series)
            except Exception as error:  # noqa: BLE001
                datetime_like = {}
                profile["datetime_error"] = _error_message(error)
            if datetime_like:
                profile["datetime"] = datetime_like

        if pd.api.types.is_datetime64_any_dtype(series):
            try:
                profile["datetime"] = _datetime_profile(series)
            except Exception as error:  # noqa: BLE001
                profile["datetime_error"] = _error_message(error)
            profile["stored_as"] = "datetime"
            profile["semantic_storage_mismatch"] = False
        if pd.api.types.is_bool_dtype(series):
            profile["stored_as"], profile["semantic_storage_mismatch"] = "boolean", False
        if profile.get("stored_as") is None:
            profile["stored_as"], profile["semantic_storage_mismatch"] = "other", False

        if (non_null_count > 0 and distinct_count <= 50 and distinct_count < non_null_count
                and not pd.api.types.is_datetime64_any_dtype(series) and value_counts is not None):
            profile["categorical"] = _categorical_profile(non_null_count, distinct_count, value_counts)

        is_constant = distinct_count <= 1
        is_float_measure = bool(numeric_block and not numeric_block.get("integer_valued", True) and not numeric_block.get("constant"))
        text_mean = (profile.get("text") or {}).get("mean_length")
        is_free_text = bool(isinstance(text_mean, (int, float)) and text_mean > PROFILING_SHAPE_MAX_MEAN_LENGTH)
        is_key_like = bool(profile.get("identifier_like") or profile.get("datetime") is not None
                           or numeric_block.get("code_like")
                           or (0 < distinct_count <= PROFILING_LOW_CARDINALITY_MAX and not is_constant))
        column_meta[unique_key] = {"report_name": name_key, "is_constant": is_constant,
                                   "is_float_measure": is_float_measure, "is_free_text": is_free_text,
                                   "is_key_like": is_key_like}
        column_distinct[unique_key] = distinct_count
        column_non_null[unique_key] = non_null_count
        columns.append(profile)

    complete_duplicate_rows = int(dataframe.duplicated(keep=False).sum()) if row_count and column_count else 0
    complete_duplicate_excess = int(dataframe.duplicated(keep="first").sum()) if row_count and column_count else 0

    # Analyses run on a positional copy with unique string labels so integer
    # or duplicate column names can never break indexing.
    frame = dataframe.copy(deep=False)
    frame.columns = unique_keys
    errors: dict[str, str] = {}

    def guarded(key: str, fn, default):
        try:
            return fn()
        except Exception as error:  # noqa: BLE001
            errors[key] = _error_message(error)
            return default

    composite = guarded("composite_uniqueness_error", lambda: _composite_uniqueness_candidates(
        frame, column_distinct, column_non_null, row_count, column_meta=column_meta), [])
    def _completeness_with_report_names() -> dict[str, Any]:
        result = _row_completeness(frame)
        for pattern in result.get("co_missing_patterns", []):
            pattern["columns"] = [column_meta.get(c, {}).get("report_name", c) for c in pattern["columns"]]
        return result

    completeness = guarded("row_completeness_error", _completeness_with_report_names, {})
    dependencies = guarded("functional_dependencies_error", lambda: _functional_dependencies(
        frame, column_meta, column_distinct, column_non_null),
        {"dependencies": [], "sampled": False})

    table_profile: dict[str, Any] = {
        "table_name": table_name,
        "row_count": row_count,
        "column_count": column_count,
        "complete_duplicate_rows": complete_duplicate_rows,
        "complete_duplicate_excess_count": complete_duplicate_excess,
        "composite_uniqueness_candidates": composite,
        "row_completeness": completeness,
        "functional_dependencies": dependencies,
        "columns": columns,
    }
    table_profile["observations"] = guarded("observations_error", lambda: _observations(columns, table_profile), [])
    table_profile.update(errors)
    return table_profile
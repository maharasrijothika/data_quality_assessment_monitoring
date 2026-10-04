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
# Lightweight profiling limits. Profiling is deterministic evidence
# gathering; these bounds keep it O(n) per column and O(bounded) cross-column
# without heavy analytics. Every sampled/bounded analysis reports
# "sampled": true + "sample_rows": N when a cap or a fixed-seed sample was
# actually applied.
# ---------------------------------------------------------------------------

PROFILING_TOP_CATEGORIES = 10
PROFILING_MAX_COMBINATIONS = 3
PROFILING_MAX_COMBO_COLUMNS = 3

# Dominant-value threshold used by the near-constant heuristic. A column is
# "near-constant" when one value covers at least this share of non-null rows
# AND more than one distinct value exists. One value covering everything is
# reported separately as "constant".
PROFILING_NEAR_CONSTANT_THRESHOLD = 0.98

# A composite combination is only worth evaluating when the product of its
# member distinct counts can reach at least this share of the row count
# (pigeonhole bound). Combinations that mathematically cannot approach
# uniqueness are skipped without touching the data.
PROFILING_COMPOSITE_PIGEONHOLE_SHARE = 0.5

# Composite combinations are reported as uniqueness candidates at or above
# this percentage of distinct combinations. Reported candidates remain
# OBSERVATIONS; uniqueness alone does not prove a business key.
PROFILING_COMPOSITE_UNIQUENESS_THRESHOLD = 95.0

# A composite uniqueness claim must be evaluated on at least this share of
# the table's rows. Combinations that only survive on a small non-null
# subset (e.g. because one member is mostly NULL) say nothing about the
# table grain and are not reported.
PROFILING_COMPOSITE_MIN_ROW_COVERAGE = 0.5

# Hard cap on composite combinations actually evaluated (drop_duplicates
# passes). Keeps profiling bounded on wide tables.
PROFILING_MAX_COMPOSITE_EVALUATIONS = 80

# Composite evaluation on very wide tables works on a deterministic sample
# above this row count; every candidate then carries "sampled": true and
# a "claim": "sample" marker.
PROFILING_COMPOSITE_SAMPLE_ROWS = 500_000

# Composite candidates at or above this uniqueness percentage are marked
# near_exact (strong key evidence, still observations only).
PROFILING_COMPOSITE_NEAR_EXACT_THRESHOLD = 99.9

# Date-format inference samples up to this many DISTINCT non-null values
# (deterministic slice) before choosing the column's parse format.
PROFILING_DATE_SAMPLE_DISTINCT = 5000

# Datetime-like pre-check: sample up to this many MOST FREQUENT distinct
# values (deterministic; value_counts order) and skip the full-row
# date-shape scan when fewer than this share of the sampled ROWS match a
# date shape. Real date columns always pass the pre-check (their values
# are the most frequent ones), so detected results are unchanged for them.
PROFILING_DATETIME_PRECHECK_DISTINCT = 500
PROFILING_DATETIME_PRECHECK_SKIP_SHARE = 0.5

# A string column is date-like when at least this share of its non-null
# values parse under the inferred format (existing threshold semantics).
PROFILING_DATE_LIKE_THRESHOLD = 95.0

# Shape/regex extraction samples up to this many non-null values above the
# cap (fixed seed), and builds the suggested regex from at most this many
# values of the dominant shape.
PROFILING_SHAPE_SAMPLE_ROWS = 200_000
PROFILING_SHAPE_REGEX_MAX_VALUES = 5000
PROFILING_SHAPE_MAX_SHAPES = 5
PROFILING_SHAPE_REGULAR_THRESHOLD = 95.0
PROFILING_SHAPE_MAX_MEAN_LENGTH = 64

# Case-variant analysis is bounded: only run when the column has at most
# this many distinct values.
PROFILING_CASE_VARIANT_MAX_DISTINCT = 50_000
PROFILING_CASE_VARIANT_EXAMPLES = 3

# Functional-dependency analysis bounds.
PROFILING_FD_MIN_COVERAGE = 98.0
PROFILING_FD_MAX_COLUMNS = 40
PROFILING_FD_MAX_PAIRS = 1200
PROFILING_FD_SAMPLE_ROWS = 200_000
PROFILING_FD_MAX_REPORTED = 30

# A determinant with distinct/non_null above this share determines every
# other column trivially (it is unique); it is excluded from FD analysis.
PROFILING_FD_UNIQUE_RATIO = 0.95

# Row-completeness co-missing analysis is bounded to the columns with the
# most nulls.
PROFILING_CO_MISSING_MAX_COLUMNS = 30
PROFILING_CO_MISSING_MAX_PATTERNS = 3
PROFILING_PACKED_MASK_BITS = 62

# Output caps.
PROFILING_MAX_OBSERVATIONS = 60
PROFILING_MAX_VIOLATION_EXAMPLES = 3

# Identifier-like text columns: shape evidence requirements.
PROFILING_IDENTIFIER_MAX_LENGTH_RANGE = 4

# Disguised-missing tokens (case-insensitive, stripped). Counted only when
# the token appears in < 50% of rows; the tokens are always reported so a
# human can judge.
DISGUISED_MISSING_TOKENS = (
    "n/a",
    "na",
    "nan",
    "null",
    "none",
    "nil",
    "missing",
    "unknown",
    "undefined",
    "tbd",
    "not available",
    "not applicable",
    "-",
    "--",
    "?",
    "??",
    "..",
)
DISGUISED_MISSING_MAX_SHARE = 0.5
DISGUISED_MISSING_TOP_TOKENS = 5

# Numeric sentinel values commonly used to encode "missing" in exports.
NUMERIC_SENTINEL_VALUES = (-1, 0, 99, 999, 9999, -999, -9999, 99999)
NUMERIC_SENTINEL_MIN_SHARE = 0.01
NUMERIC_MAX_DECIMAL_PLACES = 6

# Code-like NAME tokens for numeric columns (leading-zero loss, code-like).
CODE_LIKE_NAME_TOKENS = (
    "zip",
    "zipcode",
    "postal",
    "postcode",
    "pin",
    "pincode",
    "phone",
    "mobile",
    "tel",
    "telephone",
    "fax",
    "id",
    "code",
    "sku",
    "serial",
    "account",
    "acct",
    "ssn",
    "ref",
    "reference",
    "key",
    "keys",
    "fk",
    "pk",
)

# Postal-like patterns (Part C). The generic two-segment alnum branch was
# replaced by specific postcode shapes; the pure-digit branch is unchanged.
_POSTAL_PATTERNS = (
    re.compile(r"^\d{3,10}$"),
    # UK postcode
    re.compile(r"^[A-Za-z]{1,2}\d[A-Za-z\d]?\s?\d[A-Za-z]{2}$"),
    # Canada postcode
    re.compile(r"^[A-Za-z]\d[A-Za-z][ -]?\d[A-Za-z]\d$"),
    # US ZIP+4
    re.compile(r"^\d{5}-\d{4}$"),
    # Netherlands postcode
    re.compile(r"^\d{4}\s?[A-Za-z]{2}$"),
)

@lru_cache(maxsize=1)
def _combined_postal_regex() -> str:
    """One alternation regex for all postal shapes (single vectorised pass)."""
    return "|".join(
        f"(?:{postal_regex.pattern})" for postal_regex in _POSTAL_PATTERNS
    )

# Supported date formats (Part B). Numeric-with-separator and month-name
# forms; time part is optional. %Y%m%d is handled separately (name-gated).
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

# Time-of-day suffix, optionally attached to any date format.
_TIME_SUFFIX_RE = r"(?:[ T]\d{1,2}:\d{2}(?::\d{2})?(?:\.\d+)?)?"

# Compact 8-digit dates are only considered when the column name carries a
# date token AND every value has exactly 8 digits.
COMPACT_DATE_NAME_TOKENS = (
    "date",
    "dt",
    "day",
    "time",
    "timestamp",
    "created",
    "updated",
    "dob",
    "birth",
)

# Tokens that make a column name "date-like" for the %Y%m%d gate.
_DATE_COLUMN_NAME_TOKENS = COMPACT_DATE_NAME_TOKENS

_TOKEN_SPLIT_RE = re.compile(r"[^A-Za-z0-9]+")


@lru_cache(maxsize=1)
def _date_shape_regexes() -> tuple[str, ...]:
    """All date-shape regexes (with and without optional time suffix)."""
    regexes: list[str] = []
    for _, shape_regex in SUPPORTED_DATE_FORMATS:
        regexes.append(shape_regex)
        regexes.append(shape_regex[:-1] + _TIME_SUFFIX_RE + "$")
    regexes.append(r"^\d{8}$")
    return tuple(regexes)


@lru_cache(maxsize=1)
def _combined_date_shape_regex() -> str:
    """One alternation regex for ALL date shapes (single vectorised pass)."""
    return "|".join(
        f"(?:{regex})" for regex in _date_shape_regexes()
    )

# Identifier NAME evidence: token-aware matching on the column name. Tokens
# are matched against word boundaries (snake_case, kebab-case, spaces, and
# camelCase transitions), so "no" only matches a standalone word — not the
# "no" inside "note", "memory", or "annotation" — and "id" no longer matches
# inside "video" or "humidity".
IDENTIFIER_NAME_TOKENS = (
    "id",
    "identifier",
    "uuid",
    "guid",
    "key",
    "code",
    "number",
    "no",
    "ref",
    "reference",
    "sku",
    "serial",
)


def _column_name_tokens(name: str) -> list[str]:
    """Split a column name into lowercase word tokens.

    snake_case / kebab-case / spaces split on separators; camelCase splits at
    lower-to-upper transitions ("LoyaltyNumber" -> ["loyalty", "number"],
    "CustomerID" -> ["customer", "id"]).
    """
    text = str(name)
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", text)
    tokens = [t.lower() for t in _TOKEN_SPLIT_RE.split(text) if t]
    return tokens


def identifier_name_signal_from_name(name: str) -> bool:
    """Token-aware identifier name evidence (deterministic, no ML)."""
    tokens = _column_name_tokens(name)
    return any(token in IDENTIFIER_NAME_TOKENS for token in tokens)


def _column_name_has_token(name: str, tokens: Sequence[str]) -> bool:
    """True when any name token matches one of `tokens` exactly."""
    name_tokens = _column_name_tokens(name)
    return any(token in tokens for token in name_tokens)


def _clean_value(value: Any) -> Any:
    """Convert pandas/numpy values into JSON-safe Python values."""
    if value is None:
        return None

    # Timestamps and timezone-aware datetimes can appear as categorical
    # top values (e.g. a datetime64 column with few distinct values) and
    # are not JSON serializable. Store their ISO representation.
    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if isinstance(value, (np.datetime64,)):
        return pd.Timestamp(value).isoformat()

    # Excel workbooks can surface datetime.date / datetime.time /
    # datetime.timedelta objects inside object-dtype columns.
    if isinstance(value, (datetime.datetime, datetime.date)):
        return value.isoformat()

    if isinstance(value, datetime.time):
        return value.isoformat()

    if isinstance(value, (datetime.timedelta, pd.Timedelta)):
        return str(value)

    if isinstance(value, (np.bool_,)):
        return bool(value)

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        # pd.isna raises on sequences (list/dict cells from Excel/JSON).
        pass

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        value = float(value)

        if math.isnan(value) or math.isinf(value):
            return None

        return value

    if isinstance(value, Decimal):
        if value.is_nan() or value.is_infinite():
            return None
        return float(value)

    return value


def _is_empty_string(value: Any) -> bool:
    """Return True only for an actual empty string."""
    return isinstance(value, str) and value == ""


def _is_whitespace_only(value: Any) -> bool:
    """Return True only for strings containing whitespace and no other characters."""
    return isinstance(value, str) and value.strip() == "" and value != ""


def _vectorised_string_counts(
    series: pd.Series,
    value_counts: pd.Series | None = None,
) -> tuple[int, int]:
    """Count (empty strings, whitespace-only strings) without per-value loops.

    Non-string cells are never counted, matching the historical
    _is_empty_string / _is_whitespace_only semantics exactly. When the
    shared value_counts pass is available the counts are derived from the
    distinct values weighted by occurrence (identical results).
    """
    if len(series) == 0:
        return 0, 0

    if not (pd.api.types.is_string_dtype(series) or series.dtype == object):
        return 0, 0

    if value_counts is not None and not value_counts.empty:
        distinct_pack = _distinct_strings_and_weights(value_counts)
        if distinct_pack is not None:
            strings, weights = distinct_pack
            empty_mask = (strings == "").to_numpy(dtype="bool")
            stripped = strings.str.strip()
            whitespace_mask = (
                (stripped == "") & (strings != "")
            ).to_numpy(dtype="bool")
            return (
                int(weights[empty_mask].sum()),
                int(weights[whitespace_mask].sum()),
            )

    try:
        str_series = series.astype("string")
    except (TypeError, ValueError):
        str_series = series.map(lambda value: value if isinstance(value, str) else None).astype("string")

    is_string_mask = str_series.notna()
    if not bool(is_string_mask.any()):
        return 0, 0

    strings = str_series[is_string_mask]

    empty_count = int((strings == "").sum())
    stripped = strings.str.strip()
    whitespace_count = int(((stripped == "") & (strings != "")).sum())

    return empty_count, whitespace_count


def _value_counts_safe(series: pd.Series) -> tuple[pd.Series | None, bool]:
    """value_counts(dropna=True) with an unhashable-cell fallback.

    Returns (counts, unhashable). When value_counts raises (list/dict cells
    from Excel/JSON) the series is counted on its astype(str) image and the
    caller records "unhashable_values": true.
    """
    try:
        return series.value_counts(dropna=True), False
    except TypeError:
        try:
            return series.astype(str).value_counts(dropna=True), True
        except Exception:
            return None, False


def _duplicate_counts_from_value_counts(
    value_counts: pd.Series,
    non_null_count: int,
    distinct_count: int,
) -> tuple[int, int]:
    """derive (duplicate_count, duplicate_excess_count) from value counts.

    duplicate_count: rows in groups of size > 1 (keep=False semantics).
    duplicate_excess_count: occurrences beyond the first per group
    (keep="first" semantics) = non_null - distinct.
    """
    if value_counts is None or value_counts.empty:
        return 0, 0

    group_sizes = value_counts[value_counts > 1]
    duplicate_count = int(group_sizes.sum())
    duplicate_excess_count = int(non_null_count - distinct_count)

    return duplicate_count, duplicate_excess_count


def _json_safe_float(value: Any) -> float | None:
    """JSON-safe float (no NaN/inf/numpy types)."""
    cleaned = _clean_value(value)
    if cleaned is None:
        return None
    try:
        return float(cleaned)
    except (TypeError, ValueError):
        return None


def _top_value_share(value_counts: pd.Series, non_null_count: int) -> float:
    """Share of the most frequent non-null value (0.0 when empty)."""
    if value_counts is None or value_counts.empty or non_null_count == 0:
        return 0.0
    return float(value_counts.iloc[0]) / non_null_count


def _error_detail(error: BaseException) -> str:
    """Exception class + message, truncated to 200 characters.

    Recorded in *_error profile keys so a swallowed analysis failure is
    diagnosable instead of hiding behind a bare class name.
    """
    message = str(error)
    detail = f"{type(error).__name__}: {message}" if message else type(error).__name__
    return detail[:200]


def _distinct_strings_and_weights(
    value_counts: pd.Series,
) -> tuple[pd.Series, np.ndarray] | None:
    """Distinct values as strings plus per-value row weights.

    The shared building block for distinct-weighted counting: regex and
    string operations run once per DISTINCT value, then counts are summed
    weighted by occurrence - identical results to a row-level scan at a
    fraction of the cost.
    """
    if value_counts is None or len(value_counts) == 0:
        return None
    try:
        strings = value_counts.index.to_series().astype(str)
        strings = strings.reset_index(drop=True)
        weights = value_counts.to_numpy(dtype="int64")
    except (TypeError, ValueError):
        return None
    if len(strings) != len(weights):
        return None
    return strings, weights


def _pattern_summary_weighted(
    patterns: dict[str, int],
    distinct_strings: pd.Series,
    weights: np.ndarray,
) -> dict[str, int]:
    """_pattern_summary on distinct values weighted by occurrence counts.

    Every regex runs once per DISTINCT value; each mask is summed weighted
    by row counts. Bit-identical to the row-level scan because a pattern
    matches a row iff it matches that row's value.
    """
    stripped = distinct_strings.str.strip()

    def weighted_count(mask: pd.Series) -> int:
        mask_np = mask.fillna(False).to_numpy(dtype="bool")
        return int(weights[mask_np].sum())

    patterns["email_like"] = weighted_count(
        stripped.str.fullmatch(
            r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", na=False
        )
    )

    digits = stripped.str.replace(r"\D", "", regex=True)
    digit_len_ok = digits.str.len().between(7, 15)
    phone_shape = stripped.str.fullmatch(r"\+?[0-9][0-9\s().-]{5,}", na=False)
    patterns["phone_like"] = weighted_count(digit_len_ok & phone_shape.fillna(False))

    patterns["postal_like"] = weighted_count(
        stripped.str.fullmatch(_combined_postal_regex(), na=False)
    )

    patterns["currency_like"] = weighted_count(
        stripped.str.fullmatch(r"[A-Z]{3}", na=False)
    )
    patterns["numeric_like"] = weighted_count(
        stripped.str.fullmatch(r"[-+]?\d+(\.\d+)?", na=False)
    )

    patterns["date_like"] = weighted_count(
        stripped.str.fullmatch(_combined_date_shape_regex(), na=False)
    )

    patterns["alphanumeric_like"] = weighted_count(
        stripped.str.fullmatch(r"[A-Za-z0-9]+", na=False)
    )
    patterns["contains_whitespace"] = weighted_count(
        distinct_strings.str.contains(r"\s", regex=True, na=False)
    )
    patterns["contains_special_character"] = weighted_count(
        distinct_strings.str.contains(r"[^A-Za-z0-9\s]", regex=True, na=False)
    )

    return patterns


def _pattern_summary(
    series: pd.Series,
    value_counts: pd.Series | None = None,
) -> dict[str, int]:
    """Return simple structural text-pattern counts (vectorised).

    Counts apply to non-null values. Multi-value cells (list/dict) are
    excluded exactly like the historical per-value loop. postal_like and
    date_like intentionally use the new Part B/C semantics; every other
    counter keeps the historical regex behaviour.

    When the shared value_counts pass is available the regexes run on the
    DISTINCT values weighted by occurrence counts (identical results,
    one pass over distinct values instead of all rows).
    """
    patterns = {
        "email_like": 0,
        "phone_like": 0,
        "postal_like": 0,
        "currency_like": 0,
        "numeric_like": 0,
        "date_like": 0,
        "alphanumeric_like": 0,
        "contains_whitespace": 0,
        "contains_special_character": 0,
    }

    if len(series) == 0:
        return patterns

    if not (pd.api.types.is_string_dtype(series) or series.dtype == object):
        return patterns

    if value_counts is not None and not value_counts.empty:
        distinct_pack = _distinct_strings_and_weights(value_counts)
        if distinct_pack is not None:
            return _pattern_summary_weighted(patterns, *distinct_pack)

    try:
        as_string = series.astype("string")
    except (TypeError, ValueError):
        as_string = series.map(lambda value: value if isinstance(value, str) else None).astype("string")

    non_null = as_string.dropna()
    if non_null.empty:
        return patterns

    # The historical per-value implementation matched all structural
    # patterns against the STRIPPED value; whitespace/special-char evidence
    # applies to the RAW value. Keep both semantics exactly.
    stripped = non_null.str.strip()

    def fullmatch_count(regex: str, values: pd.Series) -> int:
        return int(values.str.fullmatch(regex, na=False).sum())

    def search_count(regex: str, values: pd.Series) -> int:
        return int(values.str.contains(regex, regex=True, na=False).sum())

    patterns["email_like"] = fullmatch_count(
        r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", stripped
    )

    # Phone-like: digits with common separators, optionally a leading +,
    # 7-15 digits total (deterministic, same semantics as the loop).
    digits = stripped.str.replace(r"\D", "", regex=True)
    digit_len_ok = digits.str.len().between(7, 15)
    phone_shape = stripped.str.fullmatch(r"\+?[0-9][0-9\s().-]{5,}", na=False)
    patterns["phone_like"] = int((digit_len_ok & phone_shape.fillna(False)).sum())

    # Postal-like: pure digits (3-10) or specific postcode shapes (Part C).
    patterns["postal_like"] = int(
        stripped.str.fullmatch(_combined_postal_regex(), na=False)
        .fillna(False)
        .sum()
    )

    patterns["currency_like"] = fullmatch_count(r"[A-Z]{3}", stripped)
    patterns["numeric_like"] = fullmatch_count(r"[-+]?\d+(\.\d+)?", stripped)

    # Date-like: ANY supported date shape, optional time suffix (Part B).
    # One combined alternation keeps this a single vectorised pass.
    date_shape = stripped.str.fullmatch(
        _combined_date_shape_regex(), na=False
    ).fillna(False)
    patterns["date_like"] = int(date_shape.sum())

    patterns["alphanumeric_like"] = fullmatch_count(r"[A-Za-z0-9]+", stripped)
    patterns["contains_whitespace"] = search_count(r"\s", non_null)
    patterns["contains_special_character"] = search_count(r"[^A-Za-z0-9\s]", non_null)

    return patterns


def _separator_characters(
    series: pd.Series,
    top_n: int = 5,
    value_counts: pd.Series | None = None,
) -> dict[str, int]:
    """Most frequent non-alphanumeric characters (top N), ROW weighted.

    Counting runs on the distinct values (weighted by occurrence counts)
    so the result equals the row-level scan at a fraction of the cost.
    Whitespace is not a separator. Empty dict when nothing observed.
    """
    if value_counts is not None and not value_counts.empty:
        values = value_counts.index.to_series().astype(str)
        weights = value_counts.astype("int64")
    else:
        if len(series) == 0:
            return {}
        if not (pd.api.types.is_string_dtype(series) or series.dtype == object):
            return {}
        try:
            as_string = series.astype("string")
        except (TypeError, ValueError):
            as_string = series.map(
                lambda value: value if isinstance(value, str) else None
            ).astype("string")
        values = as_string.dropna().astype(str)
        if values.empty:
            return {}
        weights = pd.Series(1, index=values.index, dtype="int64")

    try:
        separators = values.str.findall(r"[^A-Za-z0-9\s]")
    except (TypeError, ValueError):
        return {}

    try:
        lengths = separators.str.len().fillna(0).astype("int64").to_numpy()
    except (TypeError, ValueError):
        return {}

    if int(lengths.sum()) == 0:
        return {}

    # Vectorised weighted count: explode the per-value separator lists and
    # group-sum the row weights per character (replaces the per-value
    # Python loop; identical counts and ordering).
    exploded = separators.explode()
    exploded = exploded[exploded.notna()]
    weights_np = np.asarray(weights, dtype="int64")
    element_weights = np.repeat(weights_np, lengths)

    weighted_counts = pd.Series(element_weights).groupby(
        exploded.to_numpy(dtype=object)
    ).sum()

    if weighted_counts.empty:
        return {}

    # Deterministic order: count desc, then character code.
    ordered = sorted(
        weighted_counts.items(),
        key=lambda item: (-int(item[1]), str(item[0])),
    )
    return {str(char): int(count) for char, count in ordered[:top_n]}


def _collapse_shape(shape: str) -> str:
    """Run-length collapse a shape string: AA-9999 -> A{2}-9{4}."""
    if not shape:
        return shape

    parts: list[str] = []
    i = 0
    while i < len(shape):
        char = shape[i]
        j = i
        while j < len(shape) and shape[j] == char:
            j += 1
        run = j - i
        parts.append(char if run == 1 else f"{char}{{{run}}}")
        i = j

    return "".join(parts)


def _suggested_regex(values: pd.Series) -> str | None:
    """Build an anchored regex from the values of one shape (deterministic).

    Per position: [A-Z], [a-z], [A-Za-z] or \\d by observed char class;
    literal characters are escaped; runs are compressed with {n}. Returns
    None when nothing can be built. The result must match every input value
    (tested).
    """
    values = [str(value) for value in values]
    if not values:
        return None

    lengths = {len(value) for value in values}
    if len(lengths) != 1:
        # The caller only feeds one shape, so all lengths match; keep the
        # guard anyway.
        return None

    length = lengths.pop()
    if length == 0:
        return None

    pattern_parts: list[str] = []
    for position in range(length):
        chars = {value[position] for value in values}
        digits = any(char.isdigit() for char in chars)
        upper = any(char.isupper() for char in chars)
        lower = any(char.islower() for char in chars)
        other = any(not char.isalnum() for char in chars)

        if not other and digits and upper and lower:
            pattern_parts.append("[A-Za-z0-9]")
        elif not other and digits and upper:
            pattern_parts.append("(?:\\d|[A-Z])")
        elif not other and digits and lower:
            pattern_parts.append("(?:\\d|[a-z])")
        elif not other and upper and lower:
            pattern_parts.append("[A-Za-z]")
        elif not other and digits:
            pattern_parts.append("\\d")
        elif not other and upper:
            pattern_parts.append("[A-Z]")
        elif not other and lower:
            pattern_parts.append("[a-z]")
        elif len(chars) == 1:
            pattern_parts.append(re.escape(next(iter(chars))))
        else:
            pattern_parts.append(
                "[" + "".join(re.escape(char) for char in sorted(chars)) + "]"
            )

    compressed: list[str] = []
    i = 0
    while i < len(pattern_parts):
        part = pattern_parts[i]
        j = i
        while j < len(pattern_parts) and pattern_parts[j] == part:
            j += 1
        run = j - i
        if run == 1:
            compressed.append(part)
        elif part.startswith("\\"):
            compressed.append(f"(?:{part}){{{run}}}")
        else:
            compressed.append(f"{part}{{{run}}}")
        i = j

    return "^" + "".join(compressed) + "$"


def _shape_profile(
    series: pd.Series,
    value_counts: pd.Series | None = None,
) -> dict[str, Any]:
    """Value-shape evidence: letters->A, digits->9, others kept as-is.

    Structural observation for format rules. Counting is weighted by the
    shared distinct-value counts when provided (exactly the row-level
    result, one value_counts per column overall); row sampling applies
    above PROFILING_SHAPE_SAMPLE_ROWS. Skipped for free text
    (mean_length > PROFILING_SHAPE_MAX_MEAN_LENGTH) with skipped_reason.
    """
    empty: dict[str, Any] = {
        "dominant_shape": None,
        "dominant_coverage_percentage": 0.0,
        "distinct_shapes": 0,
        "top_shapes": [],
        "is_regular": False,
        "constant_length": False,
        "length_min": None,
        "length_max": None,
        "collapsed_shape": None,
        "suggested_regex": None,
    }

    def _transform(values: pd.Series) -> pd.Series:
        shapes = values.str.replace(r"[A-Za-z]", "A", regex=True)
        return shapes.str.replace(r"\d", "9", regex=True)

    # ------------------------------------------------------------------
    # Length stats + free-text skip (on distinct values when available).
    # ------------------------------------------------------------------
    if value_counts is not None and not value_counts.empty:
        distinct_values = value_counts.index.to_series().astype(str)
        lengths = distinct_values.str.len()
        min_length = int(lengths.min())
        max_length = int(lengths.max())
        try:
            weights = value_counts.astype("int64")
            mean_length = float(
                (lengths * weights).sum() / weights.sum()
            )
        except (TypeError, ValueError, ZeroDivisionError):
            mean_length = 0.0
    else:
        if len(series) == 0:
            return empty
        if not (pd.api.types.is_string_dtype(series) or series.dtype == object):
            return empty
        try:
            as_string = series.astype("string")
        except (TypeError, ValueError):
            as_string = series.map(
                lambda value: value if isinstance(value, str) else None
            ).astype("string")
        non_null = as_string.dropna().astype(str)
        if non_null.empty:
            return empty
        lengths = non_null.str.len()
        min_length = int(lengths.min())
        max_length = int(lengths.max())
        try:
            mean_length = float(lengths.mean())
        except (TypeError, ValueError):
            mean_length = 0.0

    profile: dict[str, Any] = {
        "length_min": min_length,
        "length_max": max_length,
        "constant_length": bool(min_length == max_length),
    }

    if mean_length > PROFILING_SHAPE_MAX_MEAN_LENGTH:
        profile["skipped_reason"] = "long_text"
        profile["suggested_regex"] = None
        return {**empty, **profile}

    sampled = False
    example_values: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Shape counting via np.unique (+ bincount weighting). Deliberately
    # NOT a second .value_counts() per column; order is count desc, then
    # shape asc (deterministic).
    # ------------------------------------------------------------------
    if value_counts is not None and not value_counts.empty:
        weights_np = value_counts.to_numpy(dtype="int64")
        total = int(weights_np.sum())

        if total > PROFILING_SHAPE_SAMPLE_ROWS:
            # Row-level deterministic sample above the cap.
            sampled_frame = series.sample(
                n=PROFILING_SHAPE_SAMPLE_ROWS, random_state=0
            ).astype(str)
            shapes_np = _transform(sampled_frame).to_numpy()
            weights_np = np.ones(len(shapes_np), dtype="int64")
            example_values = sampled_frame.to_numpy()
            sampled = True
        else:
            distinct_values = value_counts.index.to_series().astype(str)
            example_values = distinct_values.to_numpy()
            shapes_np = _transform(distinct_values).to_numpy()
    else:
        try:
            as_string = series.astype("string")
        except (TypeError, ValueError):
            as_string = series.map(
                lambda value: value if isinstance(value, str) else None
            ).astype("string")
        strings = as_string.dropna().astype(str)
        if len(strings) > PROFILING_SHAPE_SAMPLE_ROWS:
            strings = strings.sample(n=PROFILING_SHAPE_SAMPLE_ROWS, random_state=0)
            sampled = True
        shapes_np = _transform(strings).to_numpy()
        weights_np = np.ones(len(shapes_np), dtype="int64")

    if len(shapes_np) == 0:
        return {**empty, **profile}

    unique_shapes, inverse = np.unique(shapes_np, return_inverse=True)
    shape_counts_np = np.bincount(inverse.ravel(), weights=weights_np).astype("int64")
    order = sorted(
        range(len(unique_shapes)),
        key=lambda position: (-int(shape_counts_np[position]), str(unique_shapes[position])),
    )

    dominant_shape = str(unique_shapes[order[0]])
    dominant_count = int(shape_counts_np[order[0]])
    total_count = int(shape_counts_np.sum())
    dominant_coverage = (dominant_count / total_count) * 100 if total_count else 0.0

    profile["dominant_shape"] = dominant_shape
    profile["dominant_coverage_percentage"] = _clean_value(dominant_coverage)
    profile["distinct_shapes"] = int(len(unique_shapes))
    profile["is_regular"] = bool(
        dominant_coverage >= PROFILING_SHAPE_REGULAR_THRESHOLD
    )

    top_shapes: list[dict[str, Any]] = []
    for position in order[:PROFILING_SHAPE_MAX_SHAPES]:
        shape = str(unique_shapes[position])
        count = int(shape_counts_np[position])

        example: str | None = None
        mask = shapes_np == shape
        if bool(mask.any()):
            if example_values is not None:
                example = str(example_values[mask][0])
            else:
                first_index = int(np.flatnonzero(mask)[0])
                example = str(strings.astype(str).iloc[first_index])

        top_shapes.append(
            {
                "shape": shape,
                "percentage": _clean_value(
                    (count / total_count) * 100 if total_count else 0.0
                ),
                "count": count,
                "example": example,
            }
        )

    profile["top_shapes"] = top_shapes
    profile["collapsed_shape"] = _collapse_shape(dominant_shape)
    profile.setdefault("suggested_regex", None)

    if profile["is_regular"]:
        mask = shapes_np == dominant_shape
        if example_values is not None:
            regex_source = pd.Series(example_values[mask])
        else:
            regex_source = pd.Series(shapes_np[mask])
        if len(regex_source) > PROFILING_SHAPE_REGEX_MAX_VALUES:
            regex_source = regex_source.iloc[:PROFILING_SHAPE_REGEX_MAX_VALUES]
        profile["suggested_regex"] = _suggested_regex(regex_source)

    profile["sampled"] = sampled
    if sampled:
        profile["sample_rows"] = int(
            min(PROFILING_SHAPE_SAMPLE_ROWS, total_count)
        )

    return profile


def _date_like_precheck_share(value_counts: pd.Series) -> float | None:
    """Row-weighted date-shape share of the most frequent distinct values.

    Deterministic sample: the up-to-500 most frequent distinct values
    (value_counts order). Used only to SKIP impossible columns; a passing
    pre-check always runs the exact full-row scan.
    """
    sample = value_counts.head(PROFILING_DATETIME_PRECHECK_DISTINCT)
    distinct_pack = _distinct_strings_and_weights(sample)
    if distinct_pack is None:
        return None
    distinct_strings, weights = distinct_pack
    if len(distinct_strings) == 0:
        return None
    stripped = distinct_strings.str.strip()
    mask = _date_like_mask(stripped).to_numpy(dtype="bool")
    sampled_rows = int(weights.sum())
    if sampled_rows == 0:
        return None
    return float(weights[mask].sum()) / float(sampled_rows)


def _datetime_like_profile(
    series: pd.Series,
    value_counts: pd.Series | None = None,
) -> dict[str, Any]:
    """
    Detect strongly date-like string columns and infer the parse format.

    Evidence semantics: the column is date-like when >= 95% of non-null
    values parse under ONE explicit supported format (>= 95% parse rate).
    Ambiguous day/month orders (dd/mm vs mm/dd) are reported with
    "format_ambiguous": true and decided only on positive evidence (any
    first/second part > 12). This is profiling evidence only; the original
    series is never modified.
    """

    non_null = series.dropna()

    if non_null.empty:
        return {}

    # Bounded pre-check on the most frequent distinct values: when far
    # fewer than half of the sampled ROWS match a date shape, the column
    # cannot plausibly reach the date-like threshold and the full-row
    # scan is skipped (identical outcome: no date detection).
    if value_counts is not None and not value_counts.empty:
        precheck_share = _date_like_precheck_share(value_counts)
        if (
            precheck_share is not None
            and precheck_share < PROFILING_DATETIME_PRECHECK_SKIP_SHARE
        ):
            return {}

    text = non_null.astype(str).str.strip()

    date_like_mask = _date_like_mask(text)
    date_like_count = int(date_like_mask.sum())

    if date_like_count == 0:
        return {}

    date_like_percentage = (date_like_count / len(non_null)) * 100

    if date_like_percentage < PROFILING_DATE_LIKE_THRESHOLD:
        return {}

    # Candidate formats: full-parse candidates (incl. month names and the
    # name-gated %Y%m%d) plus shapes without year (month-name-only forms
    # use the current year, never claimed as real dates).
    candidate_formats: list[str] = [fmt for fmt, _ in SUPPORTED_DATE_FORMATS]
    name_gates_compact = _column_name_has_token(
        str(series.name or ""), COMPACT_DATE_NAME_TOKENS
    )
    if name_gates_compact:
        candidate_formats.append("%Y%m%d")

    sample_values = text.drop_duplicates().head(PROFILING_DATE_SAMPLE_DISTINCT)

    parse_rates: dict[str, float] = {}
    for fmt in candidate_formats:
        parsed_sample = _parse_with_format(sample_values, fmt)
        rate = float(parsed_sample.notna().mean()) * 100.0
        if rate >= PROFILING_DATE_LIKE_THRESHOLD:
            parse_rates[fmt] = rate

    if not parse_rates:
        return {}

    chosen_format, confidence = _choose_date_format(text, parse_rates)

    # Parse the FULL column with the chosen explicit format.
    parsed = _parse_with_format(text, chosen_format)
    valid = parsed.dropna()

    if valid.empty:
        return {}

    time_mask = text.str.contains(r"[ T]\d{1,2}:\d{2}", regex=True, na=False)
    has_time_component = bool(time_mask[parsed.notna()].any())
    now = pd.Timestamp.now(tz=valid.dt.tz) if valid.dt.tz is not None else pd.Timestamp.now()
    future_percentage = float((valid > now).mean()) * 100.0
    span_days = float((valid.max() - valid.min()) / pd.Timedelta(days=1))

    ambiguous = chosen_format in {"%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y", "%m-%d-%Y", "%d.%m.%Y"}

    return {
        "count": int(len(valid)),
        "date_like_count": date_like_count,
        "date_like_percentage": _clean_value(date_like_percentage),
        "format_valid_percentage": _clean_value(
            (len(valid) / len(non_null)) * 100
        ),
        "min": valid.min().isoformat(),
        "max": valid.max().isoformat(),
        "monotonic_increasing": bool(valid.is_monotonic_increasing),
        "monotonic_decreasing": bool(valid.is_monotonic_decreasing),
        "detected_format": chosen_format,
        "format_ambiguous": bool(ambiguous and confidence < 100.0),
        "format_confidence_percentage": _clean_value(confidence),
        "parse_success_percentage": _clean_value(
            (len(valid) / len(non_null)) * 100
        ),
        "has_time_component": has_time_component,
        "future_date_percentage": _clean_value(future_percentage),
        "span_days": _clean_value(span_days),
        "distinct_dates": int(valid.nunique()),
        "null_or_unparseable_count": int(len(non_null) - len(valid)),
    }


def _date_like_mask(text: pd.Series) -> pd.Series:
    """Boolean mask: values matching ANY supported date shape.

    Includes the compact ^\\d{8}$ shape; the name gate for actually PARSING
    such values lives in _datetime_like_profile.
    """
    return text.str.fullmatch(
        _combined_date_shape_regex(), na=False
    ).fillna(False)


def _parse_with_format(values: pd.Series, fmt: str) -> pd.Series:
    """Parse with an explicit format; time suffixes never break date-only formats.

    When a date-only format fails because values carry an optional time
    suffix, the suffix is stripped and the date part re-parsed. Returns a
    datetime series with NaT for unparseable values.
    """
    parsed = pd.to_datetime(values, format=fmt, errors="coerce")
    failed = parsed.isna() & values.notna()
    if bool(failed.any()):
        stripped = values[failed].str.replace(
            r"[ T]\d{1,2}:\d{2}(?::\d{2})?(?:\.\d+)?$", "", regex=True
        )
        reparsed = pd.to_datetime(stripped, format=fmt, errors="coerce")
        parsed.loc[failed] = reparsed
    return parsed


def _choose_date_format(
    text: pd.Series,
    parse_rates: dict[str, float],
) -> tuple[str, float]:
    """Pick the date format deterministically and report confidence.

    Returns (format, confidence_percentage). Day/month ambiguity is decided
    on positive evidence (first/second part > 12); when both orders parse
    and no value gives evidence the column is reported ambiguous with
    confidence 50% (month-first assumed, flagged).
    """
    rates = {fmt: parse_rates[fmt] for fmt in parse_rates}

    if "%Y-%m-%d" in rates:
        return "%Y-%m-%d", 100.0
    if "%Y/%m/%d" in rates:
        return "%Y/%m/%d", 100.0
    if "%Y%m%d" in rates:
        return "%Y%m%d", 100.0

    # Day/month ambiguous families: decide by evidence.
    for sep_fmt_first, sep_fmt_second in (
        ("%d/%m/%Y", "%m/%d/%Y"),
        ("%d-%m-%Y", "%m-%d-%Y"),
        ("%d.%m.%Y", "%m.%d.%Y"),
    ):
        first_rate = rates.get(sep_fmt_first, 0.0)
        second_rate = rates.get(sep_fmt_second, 0.0)

        if first_rate == 0.0 and second_rate == 0.0:
            continue

        first_parts = text.str.extract(r"^(\d{1,2})([-/.])(\d{1,2})[-/.]", expand=False)
        first_component = pd.to_numeric(first_parts[0], errors="coerce")
        second_component = pd.to_numeric(first_parts[2], errors="coerce")

        if bool((first_component > 12).any()):
            return sep_fmt_first, 100.0
        if bool((second_component > 12).any()):
            return sep_fmt_second, 100.0

        if first_rate > 0.0 and second_rate > 0.0:
            # No evidence either way: month-first, but flagged ambiguous.
            return sep_fmt_second, 50.0
        if first_rate > 0.0:
            return sep_fmt_first, 100.0
        return sep_fmt_second, 100.0

    # Month-name and single-family formats are unambiguous.
    best = max(rates.items(), key=lambda item: (item[1], item[0]))
    return best[0], min(100.0, float(best[1]))


def _numeric_profile(
    series: pd.Series,
    value_counts: pd.Series | None = None,
    numeric: pd.Series | None = None,
) -> dict[str, Any]:
    """Generate statistical profiling information for numeric columns.

    value_counts may carry the caller's single shared count pass (numeric
    dtype only) so each column calls value_counts exactly once. `numeric`
    may carry the caller's shared pd.to_numeric coercion so the column is
    converted exactly once.
    """

    if numeric is None:
        numeric = pd.to_numeric(series, errors="coerce")
    numeric = numeric.dropna()

    if numeric.empty:
        return {}

    distinct_numeric_count = int(numeric.nunique(dropna=True))
    is_constant = distinct_numeric_count <= 1

    # Near-constant: one dominant value covers (almost) all non-null rows but
    # other values also occur. One value covering everything is "constant".
    top_value_share = 0.0
    if not is_constant:
        if value_counts is not None and not value_counts.empty:
            top_value_share = float(value_counts.iloc[0] / len(numeric))
        else:
            top_value_share = float(numeric.value_counts().iloc[0] / len(numeric))

    is_near_constant = not is_constant and top_value_share >= PROFILING_NEAR_CONSTANT_THRESHOLD

    q25 = numeric.quantile(0.25)
    q50 = numeric.quantile(0.50)
    q75 = numeric.quantile(0.75)

    iqr = q75 - q25

    mad = (numeric - numeric.median()).abs().median()

    lower_bound = q25 - 1.5 * iqr
    upper_bound = q75 + 1.5 * iqr

    outlier_count = int(((numeric < lower_bound) | (numeric > upper_bound)).sum())

    zero_count = int((numeric == 0).sum())
    negative_count = int((numeric < 0).sum())
    positive_count = int((numeric > 0).sum())

    finite = numeric[np.isfinite(numeric)]
    integer_valued = bool(
        len(finite) == len(numeric)
        and len(numeric) > 0
        and (finite == finite.round()).all()
    )

    decimal_places = _max_decimal_places(numeric)

    p01 = numeric.quantile(0.01)
    p05 = numeric.quantile(0.05)
    p95 = numeric.quantile(0.95)
    p99 = numeric.quantile(0.99)

    try:
        skewness = float(numeric.skew())
    except (ValueError, TypeError):
        skewness = None

    return {
        "count": int(len(numeric)),
        "min": _clean_value(numeric.min()),
        "max": _clean_value(numeric.max()),
        "mean": _clean_value(numeric.mean()),
        "median": _clean_value(numeric.median()),
        "std": _clean_value(numeric.std()),
        "q25": _clean_value(q25),
        "q50": _clean_value(q50),
        "q75": _clean_value(q75),
        "iqr": _clean_value(iqr),
        "mad": _clean_value(mad),
        # Statistical observations, NOT DQ violations. The IQR fence is a
        # descriptive statistic; consumers must not turn these counts into
        # business validity rules automatically.
        "outlier_count_iqr": outlier_count,
        "outlier_percentage_iqr": _clean_value(
            (outlier_count / len(numeric)) * 100
        ),
        # Constant vs near-constant are distinct observations:
        # - "constant": every non-null value is the same (nunique <= 1).
        # - "near_constant": one dominant value covers >= 98% of non-null
        #   rows while other values also occur.
        "constant": bool(is_constant),
        "near_constant": bool(is_near_constant),
        "top_value_share_percentage": _clean_value(top_value_share * 100.0),
        "distinct_numeric_values": distinct_numeric_count,
        # Part D3 additions.
        "zero_count": zero_count,
        "negative_count": negative_count,
        "positive_count": positive_count,
        "integer_valued": integer_valued,
        "max_decimal_places": decimal_places,
        "p01": _clean_value(p01),
        "p05": _clean_value(p05),
        "p95": _clean_value(p95),
        "p99": _clean_value(p99),
        "skewness": _clean_value(skewness),
    }


def _max_decimal_places(numeric: pd.Series) -> int:
    """Capped decimal-place count (0..NUMERIC_MAX_DECIMAL_PLACES), vectorised."""
    finite = numeric[np.isfinite(numeric)]
    if finite.empty:
        return 0

    if bool((finite == finite.round()).all()):
        return 0

    scaled = finite * (10**NUMERIC_MAX_DECIMAL_PLACES)
    whole = scaled.round()
    decimals = (scaled - whole).abs()

    for places in range(1, NUMERIC_MAX_DECIMAL_PLACES + 1):
        factor = 10.0**places
        if bool(((finite * factor) % 1.0 == 0).all()):
            return places

    return NUMERIC_MAX_DECIMAL_PLACES


def _categorical_profile(
    series: pd.Series,
    distinct_count: int,
    non_null_count: int,
    value_counts: pd.Series | None = None,
) -> dict[str, Any]:
    """Lightweight categorical evidence for low-cardinality columns.

    Deterministic value counts only. No ML, no advanced analytics.
    """
    if value_counts is None:
        non_null = series.dropna()
        if non_null.empty:
            return {}
        value_counts = non_null.value_counts()

    if value_counts is None or value_counts.empty:
        return {}

    top_values = []
    for value, count in value_counts.head(PROFILING_TOP_CATEGORIES).items():
        top_values.append(
            {
                "value": _clean_value(value),
                "count": int(count),
                "percentage": _clean_value(
                    (int(count) / non_null_count) * 100 if non_null_count else 0.0
                ),
            }
        )

    return {
        "category_count": distinct_count,
        "top_values": top_values,
        "top_values_returned": len(top_values),
    }


def _datetime_profile(series: pd.Series) -> dict[str, Any]:
    """Generate profiling information for true datetime columns."""

    parsed = pd.to_datetime(series, errors="coerce")

    valid = parsed.dropna()

    if valid.empty:
        return {}

    has_time_component = bool((valid != valid.dt.normalize()).any())
    now = pd.Timestamp.now(tz=valid.dt.tz) if valid.dt.tz is not None else pd.Timestamp.now()
    future_percentage = float((valid > now).mean()) * 100.0
    span_days = float((valid.max() - valid.min()) / pd.Timedelta(days=1))

    return {
        "count": int(len(valid)),
        "min": valid.min().isoformat(),
        "max": valid.max().isoformat(),
        "format_valid_percentage": _clean_value(
            (len(valid) / len(series)) * 100
        ),
        "monotonic_increasing": bool(valid.is_monotonic_increasing),
        "monotonic_decreasing": bool(valid.is_monotonic_decreasing),
        "detected_format": None,
        "has_time_component": has_time_component,
        "future_date_percentage": _clean_value(future_percentage),
        "span_days": _clean_value(span_days),
        "distinct_dates": int(valid.nunique()),
    }


def _disguised_missing_profile(
    value_counts: pd.Series, non_null_count: int, row_count: int
) -> dict[str, Any]:
    """Disguised-missing token evidence for string/object columns.

    value_counts maps distinct raw values to occurrence counts. A token is
    counted (by ROW count) only when it appears in < 50% of rows (so a
    column where a token IS the norm is not blindly flagged); the tokens
    are always reported so a human can judge.
    """
    if value_counts is None or value_counts.empty:
        return {
            "disguised_missing_count": 0,
            "disguised_missing_percentage": 0.0,
            "disguised_missing_values": [],
        }

    counts_series = value_counts
    lowered = value_counts.index.to_series().astype(str).str.strip().str.lower()

    token_counts: list[tuple[str, int]] = []
    for token in DISGUISED_MISSING_TOKENS:
        mask = lowered == token
        count = int(counts_series[mask.fillna(False)].sum())
        if count > 0:
            token_counts.append((token, count))

    if not token_counts:
        return {
            "disguised_missing_count": 0,
            "disguised_missing_percentage": 0.0,
            "disguised_missing_values": [],
        }

    counted = [
        (token, count)
        for token, count in token_counts
        if row_count == 0 or (count / row_count) < DISGUISED_MISSING_MAX_SHARE
    ]

    total = sum(count for _, count in counted)
    ordered = sorted(counted, key=lambda item: (-item[1], item[0]))[
        :DISGUISED_MISSING_TOP_TOKENS
    ]

    return {
        "disguised_missing_count": int(total),
        "disguised_missing_percentage": _clean_value(
            (total / non_null_count) * 100 if non_null_count else 0.0
        ),
        "disguised_missing_values": [
            {"value": token, "count": count} for token, count in ordered
        ],
    }


def _case_variant_profile(value_counts: pd.Series, distinct_count: int) -> dict[str, Any]:
    """Case-variant evidence: strip + casefold + collapse internal whitespace.

    Operates on distinct values (each distinct raw spelling appears once in
    value_counts' index), so groups with > 1 spelling are real variants.
    Bounded: runs only when distinct_count <= PROFILING_CASE_VARIANT_MAX_DISTINCT;
    otherwise returns skipped_reason with up to 3 examples.
    """
    if distinct_count > PROFILING_CASE_VARIANT_MAX_DISTINCT:
        return {
            "normalized_distinct_count": None,
            "case_variant_groups": None,
            "case_variant_examples": [],
            "skipped_reason": "too_many_distinct_values",
        }

    strings = value_counts.index.to_series().astype(str)
    stripped = strings.str.strip().str.replace(r"\s+", " ", regex=True)
    normalized = stripped.str.casefold()

    normalized_distinct = int(normalized.nunique())

    groups: dict[str, list[str]] = {}
    for raw, norm in zip(stripped, normalized):
        variants = groups.setdefault(norm, [])
        if raw not in variants:
            variants.append(raw)

    variant_groups = [variants for variants in groups.values() if len(variants) > 1]
    variant_groups.sort(key=lambda variants: (-len(variants), variants[0]))

    return {
        "normalized_distinct_count": normalized_distinct,
        "case_variant_groups": len(variant_groups),
        "case_variant_examples": [
            variants[:PROFILING_CASE_VARIANT_EXAMPLES]
            for variants in variant_groups[:PROFILING_CASE_VARIANT_EXAMPLES]
        ],
    }


def _leading_zero_profile(
    numeric: pd.Series,
    value_counts: pd.Series,
    code_like: bool,
) -> dict[str, Any]:
    """Leading-zero-loss evidence for integer-valued code-like numerics.

    Suspected when the modal digit-length covers >= 80% of values and some
    values are shorter (e.g. 5-digit postal codes with 4-digit remainders
    after CSV float inference ate the leading zero).
    """
    if not code_like:
        return {}

    finite = numeric[np.isfinite(numeric)]
    if finite.empty or not bool((finite == finite.round()).all()):
        return {}

    digit_lengths = finite.abs().astype("int64").astype(str).str.len()

    length_counts = digit_lengths.value_counts()
    if length_counts.empty:
        return {}

    modal_length = int(length_counts.index[0])
    modal_share = float(length_counts.iloc[0]) / len(digit_lengths)

    shorter_count = int((digit_lengths < modal_length).sum())

    distribution = length_counts.head(5)
    digit_length_distribution = {
        str(int(length)): int(count) for length, count in distribution.items()
    }

    suspected = bool(
        modal_share >= 0.80
        and shorter_count > 0
        and modal_length > 1
    )

    return {
        "leading_zero_loss_suspected": suspected,
        "leading_zero_loss_count": shorter_count if suspected else 0,
        "digit_length_distribution": digit_length_distribution,
    }


def _leading_zero_evidence_from_strings(
    value_counts: pd.Series,
) -> bool:
    """Leading-zero-loss evidence for STRING code columns.

    True when some values are digits-only with a leading 0 while others of
    the same family lost it (e.g. "01234" next to "1234"). Deterministic,
    count-weighted.
    """
    if value_counts is None or value_counts.empty:
        return False

    values = value_counts.index.to_series().astype(str).str.strip()
    mask = values.str.fullmatch(r"0\d+", na=False).fillna(False)
    return bool(mask.any())


def _sentinel_profile(numeric: pd.Series, value_counts: pd.Series, non_null_count: int) -> dict[str, Any]:
    """Suspicious sentinel detection (-1/0/99/999/... covering >= 1%)."""
    if value_counts.empty or non_null_count == 0:
        return {"suspicious_sentinel": None}

    top_value = value_counts.index[0]
    top_count = int(value_counts.iloc[0])

    try:
        top_numeric = float(top_value)
    except (TypeError, ValueError):
        return {"suspicious_sentinel": None}

    if top_numeric in NUMERIC_SENTINEL_VALUES and top_numeric.is_integer():
        share = top_count / non_null_count
        if share >= NUMERIC_SENTINEL_MIN_SHARE:
            return {
                "suspicious_sentinel": {
                    "value": int(top_numeric),
                    "percentage": _clean_value(share * 100.0),
                }
            }

    return {"suspicious_sentinel": None}


def _integer_sequence_profile(numeric: pd.Series) -> dict[str, Any]:
    """Counter evidence for integer-valued numeric columns.

    counter_like: every non-null value is UNIQUE and the sorted distinct
    values increase by one constant step (step 1 = dense). A repeated
    constant-step column (TerritoryKey 1..10 over 29k rows) is a LABEL,
    not a counter: it must not be counter_like. It is reported separately
    as low_cardinality (distinct <= 50), which does NOT affect code_like.
    """
    finite = numeric[np.isfinite(numeric)]
    if finite.empty or not bool((finite == finite.round()).all()):
        return {
            "dense": False,
            "step_one": False,
            "counter_like": False,
            "low_cardinality": False,
        }

    values = np.sort(finite.to_numpy(dtype="float64"))
    distinct_values = np.unique(values)

    dense = bool(
        len(distinct_values) == len(values)
        and len(distinct_values) > 1
        and (np.diff(distinct_values) == 1).all()
    )
    step_one = bool(
        len(distinct_values) > 1 and (np.diff(distinct_values) == 1).all()
    )

    counter_like = False
    if len(distinct_values) == len(values) and len(distinct_values) > 1:
        # Every non-null value is unique AND the gaps are constant:
        # a real surrogate counter.
        gaps = np.diff(distinct_values)
        if len(np.unique(gaps)) == 1 and distinct_values[0] >= 0:
            counter_like = True

    return {
        "dense": dense,
        "step_one": step_one,
        "counter_like": counter_like,
        "low_cardinality": bool(0 < len(distinct_values) <= 50),
    }


def _composite_uniqueness_candidates(
    dataframe: pd.DataFrame,
    column_distinct: dict[str, int],
    column_non_null: dict[str, int],
    row_count: int,
    column_meta: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Evaluate generic candidate composite-uniqueness column combinations.

    The candidates are never dataset-specific: they are derived purely from
    observed column statistics, in four bounded steps:

    1. Member eligibility: a column can only contribute to a composite
       candidate when it has at least two distinct non-null values (a
       constant column adds nothing), is not already unique on its own, is
       not a float measure (fractional values), not free text
       (mean_length > 64), and not constant. Identifier-like columns, dates,
       codes/low-cardinality categoricals and integers may join.
    2. Pigeonhole pruning: a combination is only evaluated when the product
       of member distinct counts can reach
       PROFILING_COMPOSITE_PIGEONHOLE_SHARE of the row count.
    3. Key-like priority: combinations whose members are key-like
       (identifier_like / date / code / low-cardinality) are evaluated
       FIRST so the PROFILING_MAX_COMPOSITE_EVALUATIONS cap can never hide
       a true key behind random low-cardinality pairs.
    4. Minimality (Apriori-style): combinations are evaluated smallest
       first; a combination is skipped when a strict subset is already a
       uniqueness candidate.

    Ranking of the reported top PROFILING_MAX_COMBINATIONS candidates:
    near-exact first, then fewer columns, then key-likeness of members,
    then composite uniqueness percentage descending (NOT uniqueness
    alone).

    Rows containing any NULL are excluded from the uniqueness claim; the
    number of rows actually evaluated is reported as evaluated_row_count.
    On tables above PROFILING_COMPOSITE_SAMPLE_ROWS a deterministic sample
    (random_state=0) is evaluated and every candidate carries
    "sampled": true / "claim": "sample".
    """
    meta = column_meta or {}
    candidates: list[dict[str, Any]] = []
    evaluated = 0

    if row_count == 0:
        return []

    work_dataframe = dataframe
    sampled_frame = False
    if row_count > PROFILING_COMPOSITE_SAMPLE_ROWS:
        work_dataframe = dataframe.sample(
            n=PROFILING_COMPOSITE_SAMPLE_ROWS, random_state=0
        )
        sampled_frame = True
        row_count = len(work_dataframe)

    pigeonhole_bar = PROFILING_COMPOSITE_PIGEONHOLE_SHARE * row_count
    uniqueness_bar = PROFILING_COMPOSITE_UNIQUENESS_THRESHOLD / 100.0

    eligible: dict[str, int] = {}
    for column, distinct in column_distinct.items():
        column_info = meta.get(column, {})
        if column_info.get("is_constant"):
            continue
        if column_info.get("is_float_measure"):
            continue
        if column_info.get("is_free_text"):
            continue

        non_null = column_non_null.get(column, 0)
        if distinct < 2 or non_null == 0:
            continue
        if distinct / non_null >= uniqueness_bar:
            # Already unique alone; composites add nothing.
            continue
        eligible[column] = distinct

    def key_like_score(combo: tuple[str, ...]) -> tuple[int, int]:
        """(-key-like member count, cardinality product): strongest keys first.

        Combinations with MORE key-like members evaluate first (a 2-strong
        -key combo beats a 1-strong-key combo), then ascending cardinality
        product. This ordering plus the evaluation cap guarantees real keys
        are never crowded out by random low-cardinality pairs.
        """
        like = sum(1 for c in combo if meta.get(c, {}).get("is_key_like"))
        return (-like, math.prod(eligible[c] for c in combo))

    measured_unique_subsets: set[frozenset[str]] = set()

    # Factorize lazily ONCE per eligible column into plain int64 codes
    # (-1 where null). Each combination then costs pure numpy int64 work
    # (np.unique on combined codes) instead of a 3-column
    # drop_duplicates/duplicated pass over the whole frame.
    code_cache: dict[str, tuple[np.ndarray, int]] = {}

    def codes_for(column: str) -> tuple[np.ndarray, int]:
        cached = code_cache.get(column)
        if cached is None:
            column_series = work_dataframe[column]
            valid_mask = column_series.notna().to_numpy()
            codes_array = np.full(len(column_series), -1, dtype="int64")
            n_cat = 0
            if bool(valid_mask.any()):
                factor_codes, uniques = pd.factorize(
                    column_series[valid_mask], sort=False
                )
                codes_array[valid_mask] = factor_codes
                n_cat = int(len(uniques))
            cached = (codes_array, n_cat)
            code_cache[column] = cached
        return cached

    # Columns with unhashable cells (lists/dicts) cannot be factorized:
    # exclude them instead of failing the whole analysis.
    for column in list(eligible.keys()):
        try:
            codes_for(column)
        except TypeError:
            del eligible[column]

    max_size = min(PROFILING_MAX_COMBO_COLUMNS, len(eligible))

    for size in range(2, max_size + 1):
        combos = [
            combo
            for combo in combinations(eligible.keys(), size)
            if math.prod(eligible[c] for c in combo) >= pigeonhole_bar
        ]

        # Deterministic evaluation order: key-like combinations first, then
        # ascending cardinality product. Finding a true key early lets the
        # minimality rule prune all of its supersets.
        combos.sort(key=key_like_score)

        for combo in combos:
            if evaluated >= PROFILING_MAX_COMPOSITE_EVALUATIONS:
                return _rank_composite_candidates(candidates)

            combo_set = frozenset(combo)

            if any(
                frozenset(subset) in measured_unique_subsets
                for subset in combinations(combo, size - 1)
            ):
                # Minimality: a subset is already a candidate.
                continue

            combo_codes = [codes_for(column) for column in combo]

            valid_mask = np.logical_and.reduce(
                [codes_array >= 0 for codes_array, _ in combo_codes]
            )

            non_null_count = int(valid_mask.sum())
            if non_null_count == 0:
                continue

            if non_null_count < PROFILING_COMPOSITE_MIN_ROW_COVERAGE * row_count:
                # The claim would only describe a small NULL-free subset.
                continue

            evaluated += 1

            # Composite uniqueness = number of unique combined int64 codes
            # over the complete rows (identical to drop_duplicates on the
            # subset, without building it).
            combined_codes = np.zeros(non_null_count, dtype="int64")
            for codes_array, n_cat in combo_codes:
                combined_codes = (
                    combined_codes * np.int64(n_cat) + codes_array[valid_mask]
                )
            unique_counts = np.unique(combined_codes, return_counts=True)
            distinct_count = int(unique_counts[0].size)
            uniqueness_percentage = (distinct_count / non_null_count) * 100

            key_like_members = [
                column
                for column in combo
                if meta.get(column, {}).get("is_key_like")
            ]

            duplicate_group_counts = unique_counts[1][unique_counts[1] > 1]

            entry = {
                "columns": list(combo),
                "evaluated_row_count": non_null_count,
                "composite_distinct_count": distinct_count,
                "composite_uniqueness_percentage": _clean_value(uniqueness_percentage),
                "duplicate_composite_rows": int(duplicate_group_counts.sum()),
                "duplicate_composite_excess_count": int(
                    non_null_count - distinct_count
                ),
                "key_like_members": key_like_members,
                "contains_measure": False,
                "near_exact": bool(
                    uniqueness_percentage >= PROFILING_COMPOSITE_NEAR_EXACT_THRESHOLD
                ),
            }

            if sampled_frame:
                entry["sampled"] = True
                entry["sample_rows"] = int(row_count)
                entry["claim"] = "sample"

            if uniqueness_percentage >= PROFILING_COMPOSITE_UNIQUENESS_THRESHOLD:
                candidates.append(entry)
                measured_unique_subsets.add(combo_set)

    return _rank_composite_candidates(candidates)


def _rank_composite_candidates(
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Keep the strongest few candidates (deterministic order).

    Ranking: near-exact keys first, then fewer columns, then more key-like
    members, then uniqueness descending, then distinct-count descending.
    An exact (OrderNumber + ProductKey) pair must outrank a 99.1% pair
    with more key-like members, and an exact 2-column key must outrank an
    exact 3-column key.
    """
    ranked = sorted(
        candidates,
        key=lambda entry: (
            not entry.get("near_exact", False),
            len(entry["columns"]),
            -len(entry.get("key_like_members", [])),
            -entry["composite_uniqueness_percentage"],
            -entry["composite_distinct_count"],
        ),
    )

    return ranked[:PROFILING_MAX_COMBINATIONS]


def _row_completeness(dataframe: pd.DataFrame) -> dict[str, Any]:
    """Row-level completeness and co-missing patterns (table level).

    fully_complete_rows: rows without any null. co_missing_patterns: top
    groups of columns that are null together (bounded to the columns with
    the most nulls; packed int masks for <= 62 columns, hashed signatures
    beyond). Nulls here mean pd.isna, NOT empty strings (empty strings are
    reported per column).
    """
    row_count = len(dataframe)

    if row_count == 0 or len(dataframe.columns) == 0:
        return {
            "row_count": 0,
            "fully_complete_rows": 0,
            "fully_complete_percentage": 0.0,
            "rows_with_any_null": 0,
            "average_filled_percentage_per_row": 0.0,
            "emptiest_row_filled_percentage": 0.0,
            "co_missing_patterns": [],
        }

    null_mask = dataframe.isna()
    filled_per_row = (~null_mask).sum(axis=1)
    column_count = len(dataframe.columns)

    complete_mask = filled_per_row == column_count
    fully_complete_rows = int(complete_mask.sum())

    emptiest = int(filled_per_row.min()) if row_count else 0

    co_missing_patterns = _co_missing_patterns(dataframe, null_mask)

    return {
        "row_count": int(row_count),
        "fully_complete_rows": fully_complete_rows,
        "fully_complete_percentage": _clean_value(
            (fully_complete_rows / row_count) * 100
        ),
        "rows_with_any_null": int(row_count - fully_complete_rows),
        "average_filled_percentage_per_row": _clean_value(
            float(filled_per_row.mean() / column_count) * 100
        ),
        "emptiest_row_filled_percentage": _clean_value(
            (emptiest / column_count) * 100
        ),
        "co_missing_patterns": co_missing_patterns,
    }


def _co_missing_patterns(dataframe: pd.DataFrame, null_mask: pd.DataFrame) -> list[dict[str, Any]]:
    """Top co-missing column groups (columns null in the same rows).

    Bounded to the PROFILING_CO_MISSING_MAX_COLUMNS columns with the most
    nulls, and only columns that have at least one null.
    """
    null_counts = null_mask.sum()
    null_columns = [column for column in null_counts.index if null_counts[column] > 0]

    if not null_columns:
        return []

    null_columns.sort(key=lambda column: (-int(null_counts[column]), str(column)))
    selected = null_columns[:PROFILING_CO_MISSING_MAX_COLUMNS]

    sub_mask = null_mask[selected]

    if len(selected) <= PROFILING_PACKED_MASK_BITS:
        # Pack the null row mask into integers, then count identical masks.
        packed = np.zeros(len(sub_mask), dtype="uint64")
        for bit_position, column in enumerate(selected):
            packed |= sub_mask[column].to_numpy(dtype="uint64") << np.uint64(bit_position)

        unique_masks, inverse, counts = np.unique(
            packed, return_inverse=True, return_counts=True
        )
        pattern_rows = [
            (mask, int(counts[position])) for position, mask in enumerate(unique_masks)
        ]
        column_names = selected
        mask_to_columns = None
    else:
        # Hashed row signature for very wide tables.
        signatures = pd.util.hash_pandas_object(sub_mask, index=False).astype("uint64")
        unique_masks, inverse, counts = np.unique(
            signatures.to_numpy(), return_inverse=True, return_counts=True
        )
        first_row_of: dict[int, int] = {}
        inverse_list = inverse.tolist()
        for row_index, group_index in enumerate(inverse_list):
            if group_index not in first_row_of:
                first_row_of[group_index] = row_index
        pattern_rows = [
            (mask, int(counts[position])) for position, mask in enumerate(unique_masks)
        ]
        column_names = selected
        mask_to_columns = first_row_of

    patterns: list[dict[str, Any]] = []
    for position, (mask, count) in enumerate(pattern_rows):
        if mask == 0:
            continue

        # How many columns are null in this mask?
        if mask_to_columns is None:
            null_column_count = bin(mask).count("1")
        else:
            row_index = mask_to_columns[position]
            null_column_count = int(sub_mask.iloc[row_index].sum())

        # A co-missing pattern needs >= 2 columns missing together.
        if null_column_count < 2:
            continue

        if mask_to_columns is None:
            columns = [
                column_names[bit]
                for bit in range(len(column_names))
                if mask & (1 << bit)
            ]
        else:
            row_index = mask_to_columns[position]
            columns = [column for column, is_null in zip(column_names, sub_mask.iloc[row_index]) if is_null]

        patterns.append({"columns": columns, "row_count": count})

    patterns.sort(key=lambda pattern: (-pattern["row_count"], pattern["columns"]))

    return patterns[:PROFILING_CO_MISSING_MAX_PATTERNS]


def _functional_dependencies(
    dataframe: pd.DataFrame,
    column_meta: dict[str, dict[str, Any]],
    column_distinct: dict[str, int],
    column_non_null: dict[str, int],
) -> dict[str, Any]:
    """Within-table functional dependencies (O(n) per pair, factorized codes).

    IMPORTANT SEMANTICS: dependencies are OBSERVATIONS ON THIS DATA.
    A -> B with 100% coverage means "no counter-example was found in this
    table"; violations may be data errors or real business exceptions.
    Dependencies are NEVER turned into rules automatically.

    A violating group is an A value mapped to more than one B value.
    Coverage = 100 * (1 - rows_in_violating_groups / evaluated_rows).
    """
    row_count = len(dataframe)

    if row_count == 0:
        return {
            "evaluated_rows": 0,
            "sampled": False,
            "dependencies": [],
            "skipped_reason": None,
        }

    work_dataframe = dataframe
    sampled_frame = False
    if row_count > PROFILING_FD_SAMPLE_ROWS:
        work_dataframe = dataframe.sample(n=PROFILING_FD_SAMPLE_ROWS, random_state=0)
        sampled_frame = True
        row_count = len(work_dataframe)

    # Deterministic eligible-column selection: lowest distinct ratio first.
    eligible: list[tuple[str, float]] = []
    for column, info in column_meta.items():
        distinct = column_distinct.get(column, 0)
        non_null = column_non_null.get(column, 0)

        if info.get("is_constant") or info.get("is_free_text") or info.get("is_float_measure"):
            continue
        if non_null == 0 or distinct < 2:
            continue
        ratio = distinct / non_null
        if ratio > PROFILING_FD_UNIQUE_RATIO:
            # A unique column determines everything trivially.
            continue
        eligible.append((column, ratio))

    eligible.sort(key=lambda item: (item[1], item[0]))
    selected_columns = [column for column, _ in eligible[:PROFILING_FD_MAX_COLUMNS]]

    # Factorize once per eligible column into PLAIN int64 codes (-1 where
    # null). pandas Int64/pd.NA arrays are deliberately avoided: they are
    # an order of magnitude slower on wide pair loops.
    codes: dict[str, np.ndarray] = {}
    n_categories: dict[str, int] = {}
    for column in selected_columns:
        series = work_dataframe[column]
        non_null_mask = series.notna().to_numpy()
        code_array = np.full(len(series), -1, dtype="int64")
        n_cat = 0
        if bool(non_null_mask.any()):
            try:
                factor_codes, uniques = pd.factorize(series[non_null_mask], sort=True)
            except TypeError:
                # Unhashable cells (lists/dicts): the column cannot take
                # part in factorized dependency analysis - exclude it
                # instead of failing the whole analysis.
                continue
            code_array[non_null_mask] = factor_codes
            n_cat = int(len(uniques))
        codes[column] = code_array
        n_categories[column] = n_cat

    # Keep only columns that were successfully factorized.
    selected_columns = [column for column in selected_columns if column in codes]

    pair_budget = PROFILING_FD_MAX_PAIRS
    raw_pairs: list[tuple[str, str, float, int, int]] = []

    for a_column in selected_columns:
        if pair_budget <= 0:
            break
        a_codes = codes[a_column]
        for b_column in selected_columns:
            if b_column == a_column:
                continue
            if pair_budget <= 0:
                break

            b_ratio = (
                column_distinct.get(b_column, 0)
                / column_non_null.get(b_column, 1)
            )
            if b_ratio > PROFILING_FD_UNIQUE_RATIO:
                # B unique: the dependency is vacuous.
                continue

            pair_budget -= 1

            ca_codes = codes[a_column]
            cb_codes = codes[b_column]
            n_a = n_categories[a_column]
            n_b = n_categories[b_column]

            valid = (ca_codes >= 0) & (cb_codes >= 0)
            evaluated_rows = int(valid.sum())
            if evaluated_rows == 0:
                continue

            a_values = ca_codes[valid]
            b_values = cb_codes[valid]

            # ROWS per A value (not distinct pairs).
            rows_per_a = np.bincount(a_values, minlength=n_a)
            # Distinct (A, B) pairs, then distinct B values per A value.
            combined_pairs = np.unique(a_values * np.int64(n_b) + b_values)
            distinct_b_per_a = np.bincount(
                combined_pairs // np.int64(n_b), minlength=n_a
            )
            violating_a = np.flatnonzero(distinct_b_per_a > 1)
            # Violating rows = rows in ANY group whose A value maps to
            # more than one B value (the old pair-count math undercounted
            # these badly).
            violating_rows = int(rows_per_a[violating_a].sum())

            coverage = (1.0 - violating_rows / evaluated_rows) * 100.0
            if coverage >= PROFILING_FD_MIN_COVERAGE:
                raw_pairs.append(
                    (
                        a_column,
                        b_column,
                        coverage,
                        len(violating_a),
                        violating_rows,
                    )
                )

    # Bidirectionality check for reported pairs.
    pair_lookup = {
        (a, b): (coverage, groups, rows)
        for a, b, coverage, groups, rows in raw_pairs
    }

    entries: list[dict[str, Any]] = []
    for a_column, b_column, coverage, violating_groups, violating_rows in raw_pairs:
        reverse = pair_lookup.get((b_column, a_column))
        bidirectional = bool(reverse is not None and reverse[0] >= PROFILING_FD_MIN_COVERAGE)

        entry: dict[str, Any] = {
            "determinant": a_column,
            "dependent": b_column,
            "coverage_percentage": _clean_value(coverage),
            "violating_groups": int(violating_groups),
            "violating_rows": int(violating_rows),
            "bidirectional": bidirectional,
            "violation_examples": [],
            "evaluated_rows": int(row_count),
            "sampled": sampled_frame,
        }

        entries.append(entry)

    entries = _reduce_fd_entries(entries)

    # Report order: the strongest evidence first, so the report cap never
    # hides real keys behind chance 98% pairs between two 3-value columns.
    # Order: coverage desc, then HIGHER determinant distinct ratio first
    # (a 1,861-value determinant is meaningful; a 3-value one is not).
    distinct_ratio = {
        column: (
            column_distinct.get(column, 0) / max(column_non_null.get(column, 1), 1)
        )
        for column in set(
            [entry["determinant"] for entry in entries]
            + [entry["dependent"] for entry in entries]
        )
    }
    entries.sort(
        key=lambda entry: (
            -entry["coverage_percentage"],
            -distinct_ratio.get(entry["determinant"], 0.0),
            entry["determinant"],
            entry["dependent"],
        )
    )

    reported = entries[:PROFILING_FD_MAX_REPORTED]

    # Violation examples only for the pairs that are actually REPORTED
    # (bounded work; identical output - examples belong to reported
    # entries only).
    for entry in reported:
        if entry["coverage_percentage"] < 100.0:
            entry["violation_examples"] = _fd_violation_examples(
                work_dataframe, entry["determinant"], entry["dependent"]
            )

    return {
        "evaluated_rows": int(row_count),
        "sampled": sampled_frame,
        "dependencies": reported,
        "skipped_reason": None,
    }


def _fd_violation_examples(
    dataframe: pd.DataFrame,
    determinant: str,
    dependent: str,
    max_examples: int = PROFILING_MAX_VIOLATION_EXAMPLES,
) -> list[dict[str, Any]]:
    """First violating groups for A -> B (max 3, deterministic order)."""
    subset = dataframe[[determinant, dependent]].dropna()
    if subset.empty:
        return []

    grouped = subset.groupby(determinant, sort=False)[dependent].nunique()
    violating_values = grouped[grouped > 1].index.tolist()

    examples: list[dict[str, Any]] = []
    for determinant_value in violating_values[:max_examples]:
        rows = subset[subset[determinant] == determinant_value]
        dependent_values = [
            _clean_value(value) for value in rows[dependent].unique()
        ][:PROFILING_MAX_VIOLATION_EXAMPLES]
        examples.append(
            {
                "determinant_value": _clean_value(determinant_value),
                "dependent_values": dependent_values,
            }
        )

    return examples


def _reduce_fd_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Transitive reduction, exact implications only.

    A -> C is dropped when a middle column B exists with A -> B and B -> C
    both exactly 100% AND the path is not part of a 1:1 cycle
    (C -> B absent). Weaker (sub-100%) implications are never removed.
    """
    exact_pairs = {
        (entry["determinant"], entry["dependent"])
        for entry in entries
        if entry["coverage_percentage"] >= 100.0
    }
    nodes = {node for pair in exact_pairs for node in pair}

    kept: list[dict[str, Any]] = []
    for entry in entries:
        if entry["coverage_percentage"] >= 100.0:
            a = entry["determinant"]
            c = entry["dependent"]
            implied = any(
                (a, b) in exact_pairs
                and (b, c) in exact_pairs
                and (c, b) not in exact_pairs
                for b in nodes
                if b not in (a, c)
            )
            if implied:
                # A -> B -> C exists exactly; A -> C is redundant.
                continue
        kept.append(entry)

    return kept


def _observations(
    columns: list[dict[str, Any]],
    table_profile: dict[str, Any],
    row_count: int,
) -> list[dict[str, Any]]:
    """Deterministic evidence observations (severity/code/column/message).

    Generated only from the computed numbers; no dataset-specific text.
    Capped at PROFILING_MAX_OBSERVATIONS entries.
    """
    observations: list[dict[str, Any]] = []

    def add(
        severity: str,
        code: str,
        column: str | None,
        message: str,
        evidence: dict[str, Any],
    ) -> None:
        observations.append(
            {
                "severity": severity,
                "code": code,
                "column": column,
                "message": message,
                "evidence": evidence,
            }
        )

    dependencies = (
        table_profile.get("functional_dependencies", {}).get("dependencies", [])
    )
    dependency_violation_count = sum(
        1
        for dependency in dependencies
        if dependency.get("coverage_percentage", 100.0) < 100.0
    )
    if dependency_violation_count:
        add(
            "warning",
            "dependency_violations",
            None,
            f"{dependency_violation_count} functional dependency pair(s) have violating rows on this data",
            {"pairs_with_violations": dependency_violation_count},
        )

    composite_candidates = table_profile.get("composite_uniqueness_candidates", [])
    if composite_candidates:
        best = composite_candidates[0]
        add(
            "info",
            "best_composite_key",
            None,
            f"{' + '.join(best['columns'])}: {best['composite_uniqueness_percentage']:.3f}% unique composite",
            {
                "columns": best["columns"],
                "composite_uniqueness_percentage": best[
                    "composite_uniqueness_percentage"
                ],
                "near_exact": best.get("near_exact", False),
            },
        )

    complete_duplicate_rows = table_profile.get("complete_duplicate_rows", 0)
    if complete_duplicate_rows:
        add(
            "warning",
            "duplicate_rows",
            None,
            f"{complete_duplicate_rows} complete duplicate rows",
            {
                "complete_duplicate_rows": complete_duplicate_rows,
                "complete_duplicate_excess_count": table_profile.get(
                    "complete_duplicate_excess_count", 0
                ),
            },
        )

    row_completeness = table_profile.get("row_completeness", {})
    fully_complete_percentage = row_completeness.get("fully_complete_percentage")
    if fully_complete_percentage is not None and fully_complete_percentage < 90.0:
        add(
            "warning",
            "row_completeness_low",
            None,
            f"only {fully_complete_percentage:.1f}% of rows are fully complete",
            {
                "fully_complete_percentage": fully_complete_percentage,
                "rows_with_any_null": row_completeness.get("rows_with_any_null"),
            },
        )

    for column_profile in columns:
        name = column_profile["column_name"]
        numeric = column_profile.get("numeric") or {}
        text = column_profile.get("text") or {}
        datetime_block = column_profile.get("datetime") or {}
        shape = column_profile.get("shape") or (text.get("shape") or {})

        detected_format = datetime_block.get("detected_format")
        if detected_format:
            add(
                "info",
                "date_format_detected",
                name,
                f"{name}: {datetime_block.get('parse_success_percentage', 0):.1f}% parse as {detected_format}",
                {
                    "detected_format": detected_format,
                    "parse_success_percentage": datetime_block.get(
                        "parse_success_percentage"
                    ),
                    "format_ambiguous": datetime_block.get(
                        "format_ambiguous", False
                    ),
                },
            )

        if datetime_block.get("format_ambiguous"):
            add(
                "warning",
                "date_format_ambiguous",
                name,
                f"{name}: day/month order is ambiguous ({detected_format}) - confirm the real format",
                {
                    "detected_format": detected_format,
                    "format_confidence_percentage": datetime_block.get(
                        "format_confidence_percentage"
                    ),
                },
            )

        if numeric.get("constant") or (text and text.get("constant")):
            top_values = column_profile.get("categorical", {}).get("top_values") or []
            constant_value = top_values[0].get("value") if top_values else numeric.get("min")
            add(
                "info",
                "constant_column",
                name,
                f"{name}: every non-null value is identical",
                {"value": constant_value},
            )

        if numeric.get("near_constant") or (text and text.get("near_constant")):
            share = (
                numeric.get("top_value_share_percentage")
                if numeric.get("near_constant")
                else (text or {}).get("top_value_share_percentage")
            )
            add(
                "warning",
                "near_constant_column",
                name,
                f"{name}: one value covers {share:.1f}% of non-null rows",
                {"top_value_share_percentage": share},
            )

        if numeric.get("leading_zero_loss_suspected"):
            count = numeric.get("leading_zero_loss_count", 0)
            add(
                "warning",
                "leading_zero_loss_suspected",
                name,
                f"{name}: {count} value(s) shorter than the modal digit length - possible lost leading zeros",
                {
                    "leading_zero_loss_count": count,
                    "digit_length_distribution": numeric.get(
                        "digit_length_distribution"
                    ),
                },
            )
        elif column_profile.get("leading_zero_loss_suspected"):
            add(
                "warning",
                "leading_zero_loss_suspected",
                name,
                f"{name}: values with leading zeros sit next to shorter values of the same family - stored as text, keep it that way",
                {"evidence": "leading-zero strings observed"},
            )

        if numeric.get("code_like"):
            add(
                "info",
                "code_like_numeric",
                name,
                f"{name}: numeric column is code-like - statistics are not meaningful",
                {
                    "meaningful_statistics": False,
                    "integer_valued": numeric.get("integer_valued"),
                },
            )

        hygiene = (column_profile.get("text") or {}).get("disguised_missing") or {}
        if hygiene.get("disguised_missing_count"):
            tokens = [
                token["value"]
                for token in hygiene.get("disguised_missing_values", [])
            ]
            add(
                "warning",
                "disguised_missing_values",
                name,
                f"{name}: {hygiene['disguised_missing_count']} disguised missing value(s) ({', '.join(tokens)})",
                {
                    "disguised_missing_count": hygiene["disguised_missing_count"],
                    "disguised_missing_values": hygiene.get(
                        "disguised_missing_values"
                    ),
                },
            )

        if (text or {}).get("case_variant_groups"):
            add(
                "warning",
                "case_variants",
                name,
                f"{name}: {(text or {}).get('case_variant_groups')} case-variant group(s)",
                {
                    "case_variant_groups": (text or {}).get("case_variant_groups"),
                    "case_variant_examples": (text or {}).get(
                        "case_variant_examples"
                    ),
                },
            )

        if column_profile.get("leading_trailing_whitespace_count"):
            add(
                "warning",
                "whitespace_padding",
                name,
                f"{name}: {column_profile['leading_trailing_whitespace_count']} value(s) with leading/trailing whitespace",
                {
                    "leading_trailing_whitespace_count": column_profile[
                        "leading_trailing_whitespace_count"
                    ]
                },
            )

        integer_sequence = numeric.get("integer_sequence") or {}
        if integer_sequence.get("counter_like"):
            add(
                "info",
                "dense_integer_counter",
                name,
                f"{name}: integer values form a constant-step sequence (surrogate-counter evidence)",
                {
                    "dense": integer_sequence.get("dense"),
                    "step_one": integer_sequence.get("step_one"),
                },
            )

        if column_profile.get("identifier_repeats"):
            add(
                "info",
                "identifier_repeats",
                name,
                f"{name}: identifier-like column with repeated values ({column_profile.get('distinct_percentage', 0):.1f}% distinct)",
                {
                    "distinct_percentage": column_profile.get(
                        "distinct_percentage"
                    ),
                    "identifier_like_reasons": column_profile.get(
                        "identifier_like_reasons"
                    ),
                },
            )

        if shape and shape.get("dominant_shape"):
            if shape.get("is_regular"):
                add(
                    "info",
                    "regular_pattern",
                    name,
                    f"{name}: {shape.get('dominant_coverage_percentage', 0):.1f}% of values match shape {shape.get('collapsed_shape')}",
                    {
                        "dominant_shape": shape.get("dominant_shape"),
                        "dominant_coverage_percentage": shape.get(
                            "dominant_coverage_percentage"
                        ),
                        "suggested_regex": shape.get("suggested_regex"),
                    },
                )
            elif column_profile.get("identifier_like"):
                add(
                    "warning",
                    "irregular_pattern",
                    name,
                    f"{name}: identifier-like column is irregular (dominant shape covers {shape.get('dominant_coverage_percentage', 0):.1f}%)",
                    {
                        "dominant_shape": shape.get("dominant_shape"),
                        "dominant_coverage_percentage": shape.get(
                            "dominant_coverage_percentage"
                        ),
                    },
                )

    sampled_blocks = []
    for column_profile in columns:
        for block_name in ("text", "numeric", "shape"):
            block = column_profile.get(block_name)
            if isinstance(block, dict) and block.get("sampled"):
                sampled_blocks.append(
                    {
                        "column": column_profile["column_name"],
                        "analysis": block_name,
                        "sample_rows": block.get("sample_rows"),
                    }
                )
    composite_sampled = [
        candidate
        for candidate in composite_candidates
        if candidate.get("sampled")
    ]
    dependencies_sampled = table_profile.get("functional_dependencies", {}).get(
        "sampled", False
    )
    if sampled_blocks or composite_sampled or dependencies_sampled:
        add(
            "info",
            "sampled_analysis",
            None,
            "part of this profile was computed on deterministic samples",
            {
                "columns": sampled_blocks,
                "composite_sampled": bool(composite_sampled),
                "functional_dependencies_sampled": bool(dependencies_sampled),
            },
        )

    return observations[:PROFILING_MAX_OBSERVATIONS]


def profile_dataframe(
    dataframe: pd.DataFrame,
    table_name: str | None = None,
) -> dict[str, Any]:
    """
    Generate a reusable profile artifact for a dataframe.

    This function only profiles the data.
    It does not recommend DQ rules or execute them.
    """

    row_count = len(dataframe)
    column_count = len(dataframe.columns)

    columns: list[dict[str, Any]] = []
    column_distinct: dict[str, int] = {}
    column_non_null: dict[str, int] = {}
    column_meta: dict[str, dict[str, Any]] = {}

    # Duplicate column names are profiled positionally; keys stay unique by
    # adding a position suffix ONLY in the internal maps (the reported
    # column_name keeps the original name).
    seen_names: Counter = Counter()
    unique_column_keys: list[str] = []
    report_name_by_key: dict[str, str] = {}

    for position, column in enumerate(dataframe.columns):
        series = dataframe.iloc[:, position]
        name_key = str(column)
        seen_names[name_key] += 1
        unique_key = (
            name_key if seen_names[name_key] == 1 else f"{name_key}__pos{position}"
        )

        null_count = int(series.isna().sum())

        # The single shared value_counts pass for this column (every
        # distinct-weighted analysis below reuses it).
        value_counts, unhashable_values = _value_counts_safe(series)
        if unhashable_values:
            # Counting fell back to astype(str); distinct numbers below are
            # still exact for the reported keys.
            pass

        empty_string_count, whitespace_only_count = _vectorised_string_counts(
            series, value_counts=value_counts
        )

        if value_counts is not None:
            distinct_count = int(len(value_counts))
        else:
            distinct_count = int(series.nunique(dropna=True))

        non_null_count = int(series.notna().sum())

        duplicate_count, duplicate_excess_count = _duplicate_counts_from_value_counts(
            value_counts, non_null_count, distinct_count
        )

        distinct_percentage = (
            (distinct_count / non_null_count) * 100 if non_null_count else 0.0
        )

        identifier_name_signal = identifier_name_signal_from_name(str(column))

        identifier_completeness_percentage = (
            (non_null_count / row_count) * 100 if row_count else 0.0
        )

        identifier_uniqueness_percentage = distinct_percentage

        # Candidate identifier = name evidence + observed completeness +
        # observed uniqueness. Repeated values (e.g. a customer key in a
        # monthly fact table) legitimately fail this check without being a
        # DQ failure.
        identifier_signal = (
            non_null_count > 0
            and distinct_count == non_null_count
            and identifier_name_signal
        )

        column_profile: dict[str, Any] = {
            "column_name": str(column),
            "data_type": str(series.dtype),
            "row_count": row_count,
            "null_count": null_count,
            "null_percentage": (null_count / row_count) * 100 if row_count else 0.0,
            "empty_string_count": empty_string_count,
            "empty_string_percentage": (
                (empty_string_count / row_count) * 100 if row_count else 0.0
            ),
            "whitespace_only_count": whitespace_only_count,
            "whitespace_only_percentage": (
                (whitespace_only_count / row_count) * 100 if row_count else 0.0
            ),
            "distinct_count": distinct_count,
            "distinct_percentage": _clean_value(distinct_percentage),
            # Repeated-value statistics. These count REPEATED VALUES within
            # one column. They are NOT a duplication verdict: a monthly
            # activity table legitimately repeats every customer key, and a
            # low-cardinality category repeats every value. Whole-row
            # duplication is reported at table level below as
            # complete_duplicate_rows.
            "duplicate_count": duplicate_count,
            "duplicate_excess_count": duplicate_excess_count,
            "identifier_name_signal": identifier_name_signal,
            "identifier_completeness_percentage": _clean_value(
                identifier_completeness_percentage
            ),
            "identifier_uniqueness_percentage": _clean_value(
                identifier_uniqueness_percentage
            ),
            "identifier_signal": identifier_signal,
        }

        if unhashable_values:
            column_profile["unhashable_values"] = True

        # ------------------------------------------------------------------
        # Numeric profiling (Part D3).
        # ------------------------------------------------------------------
        numeric_block: dict[str, Any] = {}
        if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            # Convert the column to numeric EXACTLY once and share the
            # result with every numeric sub-analysis.
            numeric_series = pd.to_numeric(series, errors="coerce")

            numeric_block = _numeric_profile(
                series, value_counts=value_counts, numeric=numeric_series
            )

            integer_sequence = _integer_sequence_profile(numeric_series.dropna())
            numeric_block["integer_sequence"] = integer_sequence

            code_like = _numeric_code_like(
                str(column), numeric_block, integer_sequence, identifier_signal
            )
            numeric_block["code_like"] = code_like
            numeric_block["meaningful_statistics"] = not code_like

            sentinel = _sentinel_profile(
                numeric_series.dropna(),
                value_counts if value_counts is not None else pd.Series(dtype="float64"),
                non_null_count,
            )
            numeric_block.update(sentinel)

            leading_zero = _leading_zero_profile(
                numeric_series.dropna(),
                value_counts if value_counts is not None else pd.Series(dtype="float64"),
                code_like,
            )
            numeric_block.update(leading_zero)

            stored_as = "numeric"
            semantic_mismatch = code_like
            column_profile["stored_as"] = stored_as
            column_profile["semantic_storage_mismatch"] = semantic_mismatch

            if code_like and not numeric_block.get("constant"):
                shape_block = _numeric_shape_profile(numeric_series.dropna())
                if shape_block:
                    column_profile["shape"] = shape_block

            column_profile["numeric"] = numeric_block

            # Numeric identifier evidence: integer-valued, non-constant
            # numerics with a NAME signal or a true (unique) counter
            # sequence are identifier-like. Never for float-with-fractions
            # (integer_valued False) or bool (branch not entered).
            numeric_identifier_reasons: list[str] = []
            if (
                numeric_block.get("integer_valued")
                and not numeric_block.get("constant")
            ):
                if identifier_name_signal:
                    numeric_identifier_reasons = ["name_signal"]
                elif integer_sequence.get("counter_like"):
                    numeric_identifier_reasons = ["counter_like"]
            column_profile["identifier_like"] = bool(numeric_identifier_reasons)
            column_profile["identifier_like_reasons"] = numeric_identifier_reasons
            column_profile["identifier_repeats"] = bool(
                numeric_identifier_reasons and distinct_percentage < 100.0
            )

        # ------------------------------------------------------------------
        # Text profiling (Parts A/C/D).
        # ------------------------------------------------------------------
        if pd.api.types.is_string_dtype(series) or series.dtype == object:
            non_null_text = series.dropna().astype(str)

            text_block: dict[str, Any] = {}
            if not non_null_text.empty:
                lengths = non_null_text.str.len()

                text_block = {
                    "min_length": int(lengths.min()),
                    "max_length": int(lengths.max()),
                    "mean_length": _clean_value(lengths.mean()),
                    "median_length": _clean_value(lengths.median()),
                    "patterns": _pattern_summary(series, value_counts=value_counts),
                    "separator_characters": _separator_characters(
                        series, value_counts=value_counts
                    ),
                }

                try:
                    mean_length = float(lengths.mean())
                except (TypeError, ValueError):
                    mean_length = 0.0

                shape_block = _shape_profile(series, value_counts=value_counts)
                text_block["shape"] = shape_block

                # Disguised missing + case variants (distinct-set based).
                if value_counts is not None and not value_counts.empty:
                    hygiene = _disguised_missing_profile(
                        value_counts,
                        non_null_count,
                        row_count,
                    )
                    hygiene.update(
                        _case_variant_profile(value_counts, distinct_count)
                    )
                    text_block["disguised_missing"] = hygiene
                    text_block["normalized_distinct_count"] = hygiene.get(
                        "normalized_distinct_count"
                    )
                    text_block["case_variant_groups"] = hygiene.get(
                        "case_variant_groups"
                    )
                    text_block["case_variant_examples"] = hygiene.get(
                        "case_variant_examples"
                    )
                    if hygiene.get("skipped_reason"):
                        text_block["skipped_reason"] = hygiene["skipped_reason"]

                # Leading/trailing whitespace (distinct-weighted; identical
                # counts without a second full-column string pass).
                try:
                    padded_count: int | None = None
                    if value_counts is not None and not value_counts.empty:
                        distinct_pack = _distinct_strings_and_weights(value_counts)
                        if distinct_pack is not None:
                            strings, weights = distinct_pack
                            stripped_strings = strings.str.strip()
                            padded_mask = (
                                (strings != stripped_strings)
                                & (stripped_strings != "")
                            ).to_numpy(dtype="bool")
                            padded_count = int(weights[padded_mask].sum())
                    if padded_count is None:
                        as_string = series.astype("string")
                        non_null_strings = as_string.dropna()
                        padded = non_null_strings[
                            non_null_strings.notna()
                            & (non_null_strings != non_null_strings.str.strip())
                            & (non_null_strings.str.strip() != "")
                        ]
                        padded_count = int(len(padded))
                    column_profile["leading_trailing_whitespace_count"] = int(
                        padded_count
                    )
                except (TypeError, ValueError):
                    column_profile["leading_trailing_whitespace_count"] = 0

                # String code columns can carry the evidence directly
                # (values like "01234" sitting next to "1234").
                if (
                    value_counts is not None
                    and not value_counts.empty
                    and _column_name_has_token(str(column), CODE_LIKE_NAME_TOKENS)
                ):
                    if _leading_zero_evidence_from_strings(value_counts):
                        column_profile["leading_zero_loss_suspected"] = True

                # Text constant/near-constant (Part D1).
                if value_counts is not None and not value_counts.empty:
                    top_share = _top_value_share(value_counts, non_null_count)
                    text_block["top_value_share_percentage"] = _clean_value(
                        top_share * 100.0
                    )
                    text_block["constant"] = bool(distinct_count <= 1)
                    text_block["near_constant"] = bool(
                        distinct_count > 1
                        and top_share >= PROFILING_NEAR_CONSTANT_THRESHOLD
                    )

                # Identifier-likeness without uniqueness (Part D1).
                (
                    identifier_like,
                    identifier_like_reasons,
                ) = _identifier_like_evidence(
                    text_block,
                    shape_block,
                    identifier_name_signal,
                    identifier_signal,
                )
                column_profile["identifier_like"] = identifier_like
                column_profile["identifier_like_reasons"] = identifier_like_reasons
                column_profile["identifier_repeats"] = bool(
                    identifier_like and distinct_percentage < 100.0
                )

                # Non-string cells must not trigger identifier evidence.
                if identifier_like:
                    object_like = series.astype("string").dropna()
                    if object_like.empty:
                        column_profile["identifier_like"] = False
                        column_profile["identifier_like_reasons"] = []
                        column_profile["identifier_repeats"] = False

                stored_as = "string"
                all_numeric_like = bool(
                    text_block.get("patterns", {}).get("numeric_like", 0)
                    == int(len(non_null_text))
                    and int(len(non_null_text)) > 0
                )
                column_profile["stored_as"] = stored_as
                column_profile["semantic_storage_mismatch"] = bool(
                    all_numeric_like
                )

            if text_block:
                column_profile["text"] = text_block

            # Detect strongly date-like string columns (Part B).
            # This does not change the original dtype.
            try:
                datetime_like = _datetime_like_profile(
                    series, value_counts=value_counts
                )
            except Exception as error:  # pragma: no cover - defensive
                datetime_like = {}
                column_profile["datetime_error"] = _error_detail(error)

            if datetime_like:
                column_profile["datetime"] = datetime_like

        # True datetime dtype profiling.
        if pd.api.types.is_datetime64_any_dtype(series):
            try:
                column_profile["datetime"] = _datetime_profile(series)
            except Exception as error:  # pragma: no cover - defensive
                column_profile["datetime_error"] = _error_detail(error)

            column_profile["stored_as"] = "datetime"
            column_profile["semantic_storage_mismatch"] = False

        if pd.api.types.is_bool_dtype(series):
            column_profile["stored_as"] = "boolean"
            column_profile["semantic_storage_mismatch"] = False

        if column_profile.get("stored_as") is None:
            column_profile["stored_as"] = "other"
            column_profile["semantic_storage_mismatch"] = False

        # Lightweight categorical evidence for low-cardinality columns.
        # datetime64 columns are excluded: they already carry dedicated
        # datetime evidence (min/max/monotonic) and their top values would
        # be timestamps rather than categories.
        if (
            non_null_count > 0
            and distinct_count <= 50
            and distinct_count < non_null_count
            and not pd.api.types.is_datetime64_any_dtype(series)
        ):
            try:
                column_profile["categorical"] = _categorical_profile(
                    series,
                    distinct_count=distinct_count,
                    non_null_count=non_null_count,
                    value_counts=value_counts,
                )
            except TypeError:
                column_profile["categorical"] = _categorical_profile(
                    series.astype(str),
                    distinct_count=distinct_count,
                    non_null_count=non_null_count,
                )

        is_constant_column = distinct_count <= 1
        is_float_measure = bool(
            not numeric_block.get("integer_valued", True)
            and numeric_block.get("code_like", False) is False
            and numeric_block
            and not numeric_block.get("constant")
            and not numeric_block.get("integer_sequence", {}).get("counter_like")
        )
        is_free_text = bool(
            column_profile.get("text", {}).get("mean_length", 0) is not None
            and column_profile.get("text", {}).get("mean_length", 0) > PROFILING_SHAPE_MAX_MEAN_LENGTH
        )
        is_key_like = bool(
            column_profile.get("identifier_like")
            or (column_profile.get("datetime") is not None and not isinstance(column_profile.get("datetime"), str))
            or numeric_block.get("code_like")
            or (0 < distinct_count <= 50 and not is_constant_column)
        )

        column_meta[unique_key] = {
            "report_name": str(column),
            "is_constant": is_constant_column,
            "is_float_measure": is_float_measure,
            "is_free_text": is_free_text,
            "is_key_like": is_key_like,
        }

        unique_column_keys.append(unique_key)
        report_name_by_key[unique_key] = str(column)
        column_distinct[unique_key] = distinct_count
        column_non_null[unique_key] = non_null_count

        columns.append(column_profile)

    # Whole-row duplication evidence. Only a complete duplicate row is a
    # structural duplication observation; repeated values inside a single
    # column are NOT duplication failures by themselves.
    complete_duplicate_rows = int(dataframe.duplicated(keep=False).sum()) if row_count else 0
    complete_duplicate_excess = int(dataframe.duplicated(keep="first").sum()) if row_count else 0

    # Positional unique-key frame: cross-column analyses index columns by
    # the internal unique keys, never by the original (possibly non-string
    # or duplicated) labels. Reported names are mapped back below, so the
    # output always carries the original str(column) names.
    positional_frame = dataframe.copy(deep=False)
    if column_count and len(unique_column_keys) == column_count:
        positional_frame.columns = unique_column_keys

    def _map_report_names(names: Any) -> list[str]:
        """Map internal unique keys back to the reported column names."""
        mapped: list[str] = []
        seen: set[str] = set()
        for name in names:
            report_name = report_name_by_key.get(str(name), str(name))
            if report_name not in seen:
                seen.add(report_name)
                mapped.append(report_name)
        return mapped

    # Generic composite-uniqueness candidates. Never dataset-specific:
    # candidates are derived only from observed column statistics.
    composite_error: str | None = None
    try:
        composite_candidates = _composite_uniqueness_candidates(
            positional_frame,
            column_distinct,
            column_non_null,
            row_count,
            column_meta=column_meta,
        )
    except Exception as error:  # pragma: no cover - defensive
        composite_candidates = []
        composite_error = _error_detail(error)

    for candidate in composite_candidates:
        candidate["columns"] = _map_report_names(candidate.get("columns", []))
        candidate["key_like_members"] = _map_report_names(
            candidate.get("key_like_members", [])
        )

    # Row-level completeness (Part E2).
    row_completeness_error: str | None = None
    try:
        row_completeness = _row_completeness(positional_frame)
    except Exception as error:  # pragma: no cover - defensive
        row_completeness = {}
        row_completeness_error = _error_detail(error)

    for pattern in row_completeness.get("co_missing_patterns", []):
        pattern["columns"] = _map_report_names(pattern.get("columns", []))

    # Functional dependencies (Part E3). Observations on this data - never
    # rules.
    functional_dependencies_error: str | None = None
    try:
        functional_dependencies = _functional_dependencies(
            positional_frame, column_meta, column_distinct, column_non_null
        )
    except Exception as error:  # pragma: no cover - defensive
        functional_dependencies = {"dependencies": [], "sampled": False}
        functional_dependencies_error = _error_detail(error)

    for dependency in functional_dependencies.get("dependencies", []):
        dependency["determinant"] = report_name_by_key.get(
            str(dependency["determinant"]), str(dependency["determinant"])
        )
        dependency["dependent"] = report_name_by_key.get(
            str(dependency["dependent"]), str(dependency["dependent"])
        )

    table_profile: dict[str, Any] = {
        "table_name": table_name,
        "row_count": row_count,
        "column_count": column_count,
        "complete_duplicate_rows": complete_duplicate_rows,
        "complete_duplicate_excess_count": complete_duplicate_excess,
        "composite_uniqueness_candidates": composite_candidates,
        "row_completeness": row_completeness,
        "functional_dependencies": functional_dependencies,
        "columns": columns,
    }

    # Observations (Part E4): deterministic, evidence-carrying.
    observations_error: str | None = None
    try:
        observations = _observations(columns, table_profile, row_count)
    except Exception as error:  # pragma: no cover - defensive
        observations = []
        observations_error = _error_detail(error)

    table_profile["observations"] = observations

    # Per-analysis error recording (checklist: one failed optional analysis
    # must never abort the whole profile). Values carry the exception class
    # AND message (truncated to 200 chars) so nothing hides silently.
    for error_key, error_value in (
        ("composite_uniqueness_error", composite_error),
        ("row_completeness_error", row_completeness_error),
        ("functional_dependencies_error", functional_dependencies_error),
        ("observations_error", observations_error),
    ):
        if error_value is not None:
            table_profile[error_key] = error_value

    return table_profile


def _numeric_shape_profile(numeric: pd.Series) -> dict[str, Any]:
    """Shape block for code-like numeric columns, computed on int-string form."""
    numeric = numeric.dropna()
    finite = numeric[np.isfinite(numeric)]
    if finite.empty:
        return {}

    if not bool((finite == finite.round()).all()):
        return {}

    as_string = finite.abs().astype("int64").astype(str)
    return _shape_profile(as_string)


def _numeric_code_like(
    name: str,
    numeric_block: dict[str, Any],
    integer_sequence: dict[str, Any],
    identifier_signal: bool,
) -> bool:
    """code_like = name token OR integer counter OR integer identifier-like.

    Statistics are still computed for code-like columns (backward
    compatible) but flagged not meaningful.
    """
    if _column_name_has_token(name, CODE_LIKE_NAME_TOKENS):
        return True

    if numeric_block.get("integer_valued") and (
        integer_sequence.get("counter_like") or identifier_signal
    ):
        return True

    return False


def _identifier_like_evidence(
    text_block: dict[str, Any],
    shape_block: dict[str, Any],
    identifier_name_signal: bool,
    identifier_signal: bool,
) -> tuple[bool, list[str]]:
    """Identifier-likeness WITHOUT requiring uniqueness (Part D1).

    True when name signal OR (regular shape with >= 1 digit and constant
    length / short length range), never for float measures or free text.
    """
    reasons: list[str] = []

    if identifier_signal:
        reasons.append("unique_with_name_signal")

    if identifier_name_signal:
        reasons.append("name_signal")

    if shape_block and not shape_block.get("skipped_reason"):
        if shape_block.get("is_regular"):
            reasons.append("regular_shape")

    if reasons and not identifier_name_signal and not identifier_signal:
        # Regular shape alone is not enough: require >= 1 digit and a tight
        # length range (constant length or range <= 4).
        reasons = []
        dominant = shape_block.get("dominant_shape") or ""
        has_digit = "9" in dominant
        constant_length = bool(shape_block.get("constant_length"))
        length_range = 0
        if shape_block.get("length_min") is not None and shape_block.get("length_max") is not None:
            length_range = int(shape_block["length_max"]) - int(shape_block["length_min"])

        if has_digit and (constant_length or length_range <= PROFILING_IDENTIFIER_MAX_LENGTH_RANGE):
            if shape_block.get("is_regular"):
                reasons.append("regular_shape_with_digit")

    # Free text can never be identifier-like.
    mean_length = text_block.get("mean_length")
    if isinstance(mean_length, (int, float)) and mean_length > PROFILING_SHAPE_MAX_MEAN_LENGTH:
        return False, []

    return bool(reasons), reasons

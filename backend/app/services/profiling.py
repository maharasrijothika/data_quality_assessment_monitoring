from __future__ import annotations

import math
import re
from typing import Any

import numpy as np
import pandas as pd


def _clean_value(value: Any) -> Any:
    """Convert pandas/numpy values into JSON-safe Python values."""
    if value is None:
        return None

    if pd.isna(value):
        return None

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        value = float(value)

        if math.isnan(value) or math.isinf(value):
            return None

        return value

    if isinstance(value, (np.bool_,)):
        return bool(value)

    return value


def _is_empty_string(value: Any) -> bool:
    """Return True only for an actual empty string."""
    return isinstance(value, str) and value == ""


def _is_whitespace_only(value: Any) -> bool:
    """Return True only for strings containing whitespace and no other characters."""
    return isinstance(value, str) and value.strip() == "" and value != ""


def _pattern_summary(series: pd.Series) -> dict[str, int]:
    """Return simple structural text-pattern counts."""

    patterns = {
        "email_like": 0,
        "numeric_like": 0,
        "date_like": 0,
        "alphanumeric_like": 0,
        "contains_whitespace": 0,
        "contains_special_character": 0,
    }

    non_null = series.dropna().astype(str)

    for value in non_null:
        stripped = value.strip()

        # Email-like pattern
        if re.fullmatch(
            r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
            stripped,
        ):
            patterns["email_like"] += 1

        # Numeric-like pattern
        if re.fullmatch(
            r"[-+]?\d+(\.\d+)?",
            stripped,
        ):
            patterns["numeric_like"] += 1

        # Date/timestamp-like pattern
        if re.fullmatch(
            r"\d{4}[-/]\d{1,2}[-/]\d{1,2}"
            r"(?:[ T]\d{1,2}:\d{2}(?::\d{2})?(?:\.\d+)?)?",
            stripped,
        ):
            patterns["date_like"] += 1

        # Alphanumeric pattern
        if re.fullmatch(
            r"[A-Za-z0-9]+",
            stripped,
        ):
            patterns["alphanumeric_like"] += 1

        # Whitespace evidence
        if re.search(r"\s", value):
            patterns["contains_whitespace"] += 1

        # Special-character evidence
        if re.search(r"[^A-Za-z0-9\s]", value):
            patterns["contains_special_character"] += 1

    return patterns


def _datetime_like_profile(series: pd.Series) -> dict[str, Any]:
    """
    Detect strongly date-like string columns and generate datetime statistics.

    This is profiling evidence only. The original series is never modified.
    """

    non_null = series.dropna()

    if non_null.empty:
        return {}

    text = non_null.astype(str).str.strip()

    date_pattern = (
        r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}"
        r"(?:[ T]\d{1,2}:\d{2}(?::\d{2})?(?:\.\d+)?)?$"
    )

    date_like_mask = text.str.match(
        date_pattern,
        na=False,
    )

    date_like_count = int(date_like_mask.sum())

    if date_like_count == 0:
        return {}

    date_like_percentage = (
        date_like_count / len(non_null)
    ) * 100

    # Require strong evidence before treating a string column
    # as datetime-like. This prevents ordinary numeric/text columns
    # from being incorrectly interpreted as dates.
    if date_like_percentage < 95:
        return {}

    parsed = pd.to_datetime(
        text,
        errors="coerce",
        format="mixed",
    )

    valid = parsed.dropna()

    if valid.empty:
        return {}

    return {
        "count": int(len(valid)),
        "date_like_count": date_like_count,
        "date_like_percentage": _clean_value(
            date_like_percentage
        ),
        "format_valid_percentage": _clean_value(
            (len(valid) / len(non_null)) * 100
        ),
        "min": valid.min().isoformat(),
        "max": valid.max().isoformat(),
        "monotonic_increasing": bool(
            valid.is_monotonic_increasing
        ),
        "monotonic_decreasing": bool(
            valid.is_monotonic_decreasing
        ),
    }


def _numeric_profile(series: pd.Series) -> dict[str, Any]:
    """Generate statistical profiling information for numeric columns."""

    numeric = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    if numeric.empty:
        return {}

    q25 = numeric.quantile(0.25)
    q50 = numeric.quantile(0.50)
    q75 = numeric.quantile(0.75)

    iqr = q75 - q25

    mad = (
        numeric - numeric.median()
    ).abs().median()

    lower_bound = q25 - 1.5 * iqr
    upper_bound = q75 + 1.5 * iqr

    outlier_count = int(
        (
            (numeric < lower_bound)
            | (numeric > upper_bound)
        ).sum()
    )

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
        "outlier_count_iqr": outlier_count,
        "outlier_percentage_iqr": _clean_value(
            (outlier_count / len(numeric)) * 100
        ),
        "near_constant": bool(
            numeric.nunique(dropna=True) <= 1
        ),
    }


def _datetime_profile(series: pd.Series) -> dict[str, Any]:
    """Generate profiling information for true datetime columns."""

    parsed = pd.to_datetime(
        series,
        errors="coerce",
    )

    valid = parsed.dropna()

    if valid.empty:
        return {}

    return {
        "count": int(len(valid)),
        "min": valid.min().isoformat(),
        "max": valid.max().isoformat(),
        "format_valid_percentage": _clean_value(
            (len(valid) / len(series)) * 100
        ),
        "monotonic_increasing": bool(
            valid.is_monotonic_increasing
        ),
        "monotonic_decreasing": bool(
            valid.is_monotonic_decreasing
        ),
    }


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

    for column in dataframe.columns:
        series = dataframe[column]

        null_count = int(
            series.isna().sum()
        )

        empty_string_count = int(
            series.map(_is_empty_string).sum()
        )

        whitespace_only_count = int(
            series.map(_is_whitespace_only).sum()
        )

        distinct_count = int(
            series.nunique(dropna=True)
        )

        duplicate_count = int(
            series.dropna().duplicated(
                keep=False
            ).sum()
        )
        duplicate_excess_count = int(
            series.dropna().duplicated(
                keep="first"
            ).sum()
        )

        non_null_count = int(
            series.notna().sum()
        )

        distinct_percentage = (
            (distinct_count / non_null_count) * 100
            if non_null_count
            else 0.0
        )

        identifier_name_signal = any(
           token in str(column).lower()
           for token in (
               "id",
               "key",
               "code",
               "number",
               "no",
            )
        )

        identifier_completeness_percentage = (
            (non_null_count / row_count) * 100
            if row_count
            else 0.0
        )

        identifier_uniqueness_percentage = (
            (distinct_count / non_null_count) * 100
            if non_null_count
            else 0.0
        )

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
            "null_percentage": (
                (null_count / row_count) * 100
                if row_count
                else 0.0
            ),
            "empty_string_count": empty_string_count,
            "empty_string_percentage": (
                (empty_string_count / row_count) * 100
                if row_count
                else 0.0
            ),
            "whitespace_only_count": whitespace_only_count,
            "whitespace_only_percentage": (
                (whitespace_only_count / row_count) * 100
                if row_count
                else 0.0
            ),
            "distinct_count": distinct_count,
            "distinct_percentage": distinct_percentage,
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

        # Numeric profiling
        if pd.api.types.is_numeric_dtype(series):
            column_profile["numeric"] = _numeric_profile(
                series
            )

        # Text profiling
        if (
            pd.api.types.is_string_dtype(series)
            or series.dtype == object
        ):
            non_null_text = (
                series.dropna().astype(str)
            )

            if not non_null_text.empty:
                lengths = non_null_text.str.len()

                column_profile["text"] = {
                    "min_length": int(
                        lengths.min()
                    ),
                    "max_length": int(
                        lengths.max()
                    ),
                    "mean_length": _clean_value(
                        lengths.mean()
                    ),
                    "median_length": _clean_value(
                        lengths.median()
                    ),
                    "patterns": _pattern_summary(
                        series
                    ),
                }

            # Detect strongly date-like string columns.
            # This does not change the original dtype.
            datetime_like = _datetime_like_profile(
                series
            )

            if datetime_like:
                column_profile["datetime"] = (
                    datetime_like
                )

        # True datetime dtype profiling
        if pd.api.types.is_datetime64_any_dtype(
            series
        ):
            column_profile["datetime"] = (
                _datetime_profile(series)
            )

        columns.append(column_profile)

    return {
        "table_name": table_name,
        "row_count": row_count,
        "column_count": column_count,
        "columns": columns,
    }


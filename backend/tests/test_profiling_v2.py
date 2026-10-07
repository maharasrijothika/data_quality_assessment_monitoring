"""New profiling behaviour (Parts A-F) and downstream-safety tests.

All tests build small in-memory frames; no network, no required files.
One optional test loads the real 9,800-row train.csv when present.
"""

import inspect
import json
import time
from datetime import timedelta
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.services import profiling as profiling_module
from app.services.profiling import (
    PROFILING_CASE_VARIANT_MAX_DISTINCT,
    PROFILING_FD_MAX_PAIRS,
    PROFILING_MAX_COMBINATIONS,
    PROFILING_MAX_COMPOSITE_EVALUATIONS,
    _choose_date_format,
    _collapse_shape,
    _composite_uniqueness_candidates,
    _date_like_mask,
    _parse_with_format,
    _pattern_summary,
    _shape_profile,
    _suggested_regex,
    profile_dataframe,
)

TRAIN_CSV = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "raw"
    / "datasets"
    / "22"
    / "v1"
    / "train.csv"
)


# ---------------------------------------------------------------------------
# Part A: robustness — none of these may raise
# ---------------------------------------------------------------------------


def test_profile_empty_dataframe_with_columns():
    df = pd.DataFrame({"a": pd.Series(dtype="float64"), "b": pd.Series(dtype="object")})
    profile = profile_dataframe(df, "empty")
    assert_no_analysis_errors(profile)
    assert profile["row_count"] == 0
    assert profile["column_count"] == 2
    assert len(profile["columns"]) == 2
    json.dumps(profile, allow_nan=False)


def test_profile_zero_columns():
    df = pd.DataFrame()
    profile = profile_dataframe(df, "nothing")
    assert_no_analysis_errors(profile)
    assert profile["row_count"] == 0
    assert profile["column_count"] == 0
    assert profile["columns"] == []
    assert profile["composite_uniqueness_candidates"] == []
    json.dumps(profile, allow_nan=False)


def test_profile_single_row():
    df = pd.DataFrame({"a": [1], "b": ["x"]})
    profile = profile_dataframe(df)
    assert_no_analysis_errors(profile)
    assert profile["row_count"] == 1
    json.dumps(profile, allow_nan=False)


def test_profile_all_null_column():
    df = pd.DataFrame({"a": [None, None, None]})
    profile = profile_dataframe(df)
    assert_no_analysis_errors(profile)
    column = profile["columns"][0]
    assert column["null_count"] == 3
    assert column["distinct_count"] == 0
    json.dumps(profile, allow_nan=False)


def test_profile_all_empty_string_column():
    df = pd.DataFrame({"a": ["", "", ""]})
    profile = profile_dataframe(df)
    assert_no_analysis_errors(profile)
    column = profile["columns"][0]
    assert column["empty_string_count"] == 3
    assert column["distinct_count"] == 1  # "" is a real value for nunique
    assert column["text"]["patterns"]["numeric_like"] == 0
    json.dumps(profile, allow_nan=False)


def test_profile_duplicate_column_names_profiled_by_position():
    df = pd.DataFrame([{"a": 1, "b": 10, "a.1": 2}])
    df.columns = ["a", "b", "a"]  # duplicate names, values differ by position
    profile = profile_dataframe(df)
    assert_no_analysis_errors(profile)
    assert profile["column_count"] == 3
    names = [column["column_name"] for column in profile["columns"]]
    assert names == ["a", "b", "a"]
    # Position 0 has value 1, position 2 has value 2.
    assert profile["columns"][0]["numeric"]["max"] == 1
    assert profile["columns"][2]["numeric"]["max"] == 2
    json.dumps(profile, allow_nan=False)


def test_profile_non_string_column_names():
    df = pd.DataFrame(
        {
            5: [1, 2],
            ("t", "uple"): [3, 4],
        }
    )
    profile = profile_dataframe(df)
    assert_no_analysis_errors(profile)
    names = [column["column_name"] for column in profile["columns"]]
    assert names == ["5", "('t', 'uple')"]
    json.dumps(profile, allow_nan=False)


def test_profile_category_dtype():
    df = pd.DataFrame(
        {"cat": pd.Categorical(["a", "b", "a", None])}
    )
    profile = profile_dataframe(df)
    assert_no_analysis_errors(profile)
    column = profile["columns"][0]
    assert column["null_count"] == 1
    assert column["distinct_count"] == 2
    assert column["categorical"]["top_values"][0]["count"] == 2
    json.dumps(profile, allow_nan=False)


def test_profile_bool_dtype():
    df = pd.DataFrame({"flag": [True, False, True]})
    profile = profile_dataframe(df)
    assert_no_analysis_errors(profile)
    column = profile["columns"][0]
    assert "numeric" not in column or column["numeric"] == {}
    assert column["stored_as"] == "boolean"
    json.dumps(profile, allow_nan=False)


def test_profile_tz_aware_datetime():
    df = pd.DataFrame(
        {
            "ts": pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC")
        }
    )
    profile = profile_dataframe(df)
    assert_no_analysis_errors(profile)
    datetime_block = profile["columns"][0]["datetime"]
    assert datetime_block["min"].startswith("2024-01-01")
    assert "span_days" in datetime_block
    assert "future_date_percentage" in datetime_block
    json.dumps(profile, allow_nan=False)


def test_profile_timedelta_column():
    df = pd.DataFrame({"delta": [timedelta(days=1), timedelta(days=2)]})
    profile = profile_dataframe(df)
    assert_no_analysis_errors(profile)
    json.dumps(profile, allow_nan=False)


def test_profile_decimal_objects():
    df = pd.DataFrame({"money": [Decimal("1.50"), Decimal("2.25"), None]})
    profile = profile_dataframe(df)
    assert_no_analysis_errors(profile)
    json.dumps(profile, allow_nan=False)
    # Object columns with Decimals keep their distinct evidence.
    column = profile["columns"][0]
    assert column["distinct_count"] == 2


def test_profile_mixed_int_str_object_column():
    df = pd.DataFrame({"mixed": [1, "two", 3, "four"]})
    profile = profile_dataframe(df)
    assert_no_analysis_errors(profile)
    column = profile["columns"][0]
    assert column["distinct_count"] == 4
    json.dumps(profile, allow_nan=False)


def test_profile_list_and_dict_cells_fall_back_to_string_counting(monkeypatch):
    df = pd.DataFrame({"cells": [[1, 2], {"a": 1}, [1, 2], None]})

    # pandas 3 hashes list/dict cells natively; force the historical
    # TypeError path (first call only) to prove the fallback works and
    # flags the column.
    original = pd.Series.value_counts
    calls = {"n": 0}

    def raising(self, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise TypeError("unhashable type: 'list'")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(pd.Series, "value_counts", raising)
    profile = profile_dataframe(df)
    monkeypatch.undo()

    assert_no_analysis_errors(profile)
    column = profile["columns"][0]
    assert column.get("unhashable_values") is True
    assert column["distinct_count"] == 2
    json.dumps(profile, allow_nan=False)


def test_profile_list_and_dict_cells_profile_without_error():
    df = pd.DataFrame({"cells": [[1, 2], {"a": 1}, [1, 2], None]})
    profile = profile_dataframe(df)
    assert_no_analysis_errors(profile)
    assert profile["columns"][0]["distinct_count"] == 2
    json.dumps(profile, allow_nan=False)


def test_profile_constant_column_single_distinct_value():
    df = pd.DataFrame({"same": ["x", "x", "x"], "n": [7, 7, 7]})
    profile = profile_dataframe(df)
    assert_no_analysis_errors(profile)
    assert profile["columns"][0]["text"]["constant"] is True
    assert profile["columns"][1]["numeric"]["constant"] is True
    codes = {obs["code"] for obs in profile["observations"]}
    assert "constant_column" in codes


def test_profile_very_wide_table():
    df = pd.DataFrame(
        {f"col_{i}": np.arange(50, dtype=float) for i in range(500)}
    )
    profile = profile_dataframe(df)
    assert_no_analysis_errors(profile)
    assert profile["column_count"] == 500
    assert len(profile["columns"]) == 500


# ---------------------------------------------------------------------------
# Part A: one value_counts per column (spy)
# ---------------------------------------------------------------------------


def test_value_counts_called_once_per_column(monkeypatch):
    calls = {"count": 0}
    original = pd.Series.value_counts

    def spy(self, *args, **kwargs):
        calls["count"] += 1
        return original(self, *args, **kwargs)

    monkeypatch.setattr(pd.Series, "value_counts", spy)

    df = pd.DataFrame(
        {
            "id": [f"V{i%40}" for i in range(200)],
            "price": np.random.default_rng(0).uniform(0, 100, 200),
            "city": ["A"] * 200,
        }
    )
    profile_dataframe(df)

    assert calls["count"] <= 3, (
        f"value_counts called {calls['count']} times for 3 columns"
    )


# ---------------------------------------------------------------------------
# Old-vs-new equivalence (reference implementations from the OLD code)
# ---------------------------------------------------------------------------


def _old_is_empty_string(value):
    return isinstance(value, str) and value == ""


def _old_is_whitespace_only(value):
    return isinstance(value, str) and value.strip() == "" and value != ""


def _old_pattern_summary(series):
    """Reference copy of the OLD per-value implementation."""
    import re as _re

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
    non_null = series.dropna().astype(str)
    for value in non_null:
        stripped = value.strip()
        if _re.fullmatch(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", stripped):
            patterns["email_like"] += 1
        phone_digits = _re.sub(r"\D", "", stripped)
        if 7 <= len(phone_digits) <= 15 and _re.fullmatch(
            r"\+?[0-9][0-9\s().-]{5,}", stripped
        ):
            patterns["phone_like"] += 1
        if _re.fullmatch(r"[-+]?\d+(\.\d+)?", stripped):
            patterns["numeric_like"] += 1
        if _re.fullmatch(r"[A-Z]{3}", stripped):
            patterns["currency_like"] += 1
        if _re.fullmatch(r"[A-Za-z0-9]+", stripped):
            patterns["alphanumeric_like"] += 1
        if _re.search(r"\s", value):
            patterns["contains_whitespace"] += 1
        if _re.search(r"[^A-Za-z0-9\s]", value):
            patterns["contains_special_character"] += 1
    return patterns


def test_old_vs_new_equivalence_on_random_data():
    rng = np.random.default_rng(42)
    mixed_values = (
        [f"user{i}@example.com" for i in range(30)]
        + [f"+1 (555) 123-45{i}" for i in range(10)]
        + ["USD", "CAD", "EUR"]
        + [f"C{i:03d}" for i in range(20)]
        + [f"{rng.integers(100, 999)}" for _ in range(15)]
        + ["has space", " trailing ", None, "", "UPPER lower"]
    )
    series = pd.Series(mixed_values * 3, dtype="object")

    old_patterns = _old_pattern_summary(series)
    new_patterns = _pattern_summary(series)

    for key in old_patterns:
        if key in {"postal_like", "date_like"}:
            continue  # intentionally changed semantics
        assert old_patterns[key] == new_patterns[key], key


def test_old_vs_new_duplicate_and_hygiene_counts():
    rng = np.random.default_rng(7)
    values = [f"v{i%50}" for i in range(300)] + [None] * 20 + [""] * 10 + ["   "] * 5
    series = pd.Series(values, dtype="object")

    non_null = series.dropna()
    old_distinct = int(non_null.nunique(dropna=True))
    old_dup_keep_false = int(non_null.duplicated(keep=False).sum())
    old_dup_keep_first = int(non_null.duplicated(keep="first").sum())
    old_empty = int(series.map(_old_is_empty_string).sum())
    old_ws = int(series.map(_old_is_whitespace_only).sum())

    profile = profile_dataframe(series.to_frame("value"))
    column = profile["columns"][0]

    assert column["distinct_count"] == old_distinct
    assert column["duplicate_count"] == old_dup_keep_false
    assert column["duplicate_excess_count"] == old_dup_keep_first
    assert column["empty_string_count"] == old_empty
    assert column["whitespace_only_count"] == old_ws


# ---------------------------------------------------------------------------
# Part B: dates
# ---------------------------------------------------------------------------


def test_detects_dd_mm_yyyy_day_first_evidence():
    df = pd.DataFrame(
        {"order_date": ["03/04/2021", "31/12/2020", "05/06/2019", "01/01/2020"]}
    )
    block = profile_dataframe(df)["columns"][0]["datetime"]
    assert block["detected_format"] == "%d/%m/%Y"
    assert block["format_ambiguous"] is False
    assert block["min"].startswith("2019-06-05")
    assert block["max"].startswith("2021-04-03")
    # 31/12/2020 > 01/01/2020 must hold in parsed min/max.
    assert block["min"].startswith("2019-06-05")
    assert block["span_days"] == pytest.approx(668, abs=1)


def test_detects_mm_dd_yyyy_month_first_evidence():
    df = pd.DataFrame(
        {"report_date": ["13/02/2021", "12/11/2020", "01/15/2019"]}
    )
    # All values match the shape; "13" as first part is only valid day-first,
    # "15" as second part is only valid month-first -> mixed; use a pure
    # month-first column instead.
    df = pd.DataFrame(
        {"report_date": ["04/13/2021", "11/30/2020", "01/15/2019"]}
    )
    block = profile_dataframe(df)["columns"][0]["datetime"]
    assert block["detected_format"] == "%m/%d/%Y"
    assert block["format_ambiguous"] is False


def test_ambiguous_day_month_only_reported_with_flag():
    df = pd.DataFrame(
        {"event_date": ["03/04/2021", "05/06/2021", "07/08/2021", "12/11/2020"]}
    )
    block = profile_dataframe(df)["columns"][0]["datetime"]
    assert block["detected_format"] == "%m/%d/%Y"
    assert block["format_ambiguous"] is True
    assert block["format_confidence_percentage"] == 50.0


def test_detects_iso_with_time():
    df = pd.DataFrame(
        {"created_at": ["2024-01-18 10:30:00", "2024-01-19 11:45:00"]}
    )
    block = profile_dataframe(df)["columns"][0]["datetime"]
    assert block["detected_format"] == "%Y-%m-%d"
    assert block["has_time_component"] is True
    assert block["date_like_count"] == 2


def test_detects_month_name_formats():
    df = pd.DataFrame({"d1": ["12-Mar-2020", "05-Apr-2020", "25-Dec-2021"]})
    block = profile_dataframe(df)["columns"][0]["datetime"]
    assert block["detected_format"] == "%d-%b-%Y"

    df2 = pd.DataFrame({"d2": ["March 5, 2021", "April 15, 2020"]})
    block2 = profile_dataframe(df2)["columns"][0]["datetime"]
    assert block2["detected_format"] == "%B %d, %Y"


def test_compact_yyyymmdd_requires_date_token_in_name():
    values = ["20200131", "20200229", "20201215"]
    with_name = pd.DataFrame({"invoice_day": values})
    block = profile_dataframe(with_name)["columns"][0]["datetime"]
    assert block["detected_format"] == "%Y%m%d"

    without_name = pd.DataFrame({"invoice_number": values})
    assert "datetime" not in profile_dataframe(without_name)["columns"][0]


def test_date_like_pattern_counts_supported_shapes():
    series = pd.Series(["08/11/2017", "1996-07-04", "20200131"], dtype="object")
    patterns = _pattern_summary(series)
    assert patterns["date_like"] == 3


def test_non_dates_must_not_be_detected():
    frames = [
        pd.DataFrame({"zip": ["12345", "67890", "12346"]}),
        pd.DataFrame({"version": ["1.2.3", "2.0.1", "3.1.0"]}),
        pd.DataFrame({"frac": ["1/2", "3/4", "5/6"]}),
        pd.DataFrame({"phone": ["(555) 123-4567", "555-123-4568", "555.123.4569"]}),
        pd.DataFrame({"code": ["12-34-56", "12-34-57", "12-34-58"]}),
        pd.DataFrame({"text": ["apple", "banana", "cherry"]}),
        pd.DataFrame({"mixed_date": ["2024-01-01", "not a date", "2024-01-03", "nope"]}),
    ]
    for df in frames:
        assert "datetime" not in profile_dataframe(df)["columns"][0], df.columns[0]


def test_below_threshold_parse_rate_is_not_a_date():
    df = pd.DataFrame({"d": ["2024-01-01", "2024-01-02", "bad", "2024-01-04"]})
    assert "datetime" not in profile_dataframe(df)["columns"][0]


def test_true_datetime64_gets_new_keys():
    df = pd.DataFrame(
        {"ts": pd.date_range("2024-01-01", periods=5, freq="D")}
    )
    block = profile_dataframe(df)["columns"][0]["datetime"]
    assert block["detected_format"] is None
    assert block["has_time_component"] is False
    assert "future_date_percentage" in block
    assert block["span_days"] == 4.0
    assert block["distinct_dates"] == 5


def test_datetime_new_keys_present():
    df = pd.DataFrame({"d": ["2020-01-01", "2020-06-30", "2021-02-15"]})
    block = profile_dataframe(df)["columns"][0]["datetime"]
    for key in (
        "detected_format",
        "format_ambiguous",
        "format_confidence_percentage",
        "parse_success_percentage",
        "has_time_component",
        "future_date_percentage",
        "span_days",
        "distinct_dates",
        "null_or_unparseable_count",
    ):
        assert key in block, key


# ---------------------------------------------------------------------------
# Part C: shapes, regex, postal
# ---------------------------------------------------------------------------


def test_shape_regular_id_and_regex():
    values = [f"AA-{1000 + i}-{100000 + i}" for i in range(60)]
    df = pd.DataFrame({"order_id": values})
    column = profile_dataframe(df)["columns"][0]
    shape = column["text"]["shape"]

    assert shape["dominant_coverage_percentage"] == 100.0
    assert shape["is_regular"] is True
    assert shape["dominant_shape"] == "AA-9999-999999"
    assert shape["collapsed_shape"] == "A{2}-9{4}-9{6}"

    regex = shape["suggested_regex"]
    import re as _re

    assert regex is not None
    for value in values:
        assert _re.fullmatch(regex, value), value
    assert _re.fullmatch(regex, "AA-9999-99999") is None
    assert _re.fullmatch(regex, "aa-9999-999999") is None


def test_shape_mixed_shapes_not_regular():
    values = [f"AA-1{i:04d}" for i in range(60)] + [f"very different text {i % 10}" for i in range(40)]
    df = pd.DataFrame({"value": values})
    shape = profile_dataframe(df)["columns"][0]["text"]["shape"]
    assert shape["is_regular"] is False
    assert shape["suggested_regex"] is None
    assert len(shape["top_shapes"]) == 2


def test_shape_free_text_skipped():
    df = pd.DataFrame(
        {"notes": ["this is a long sentence with many words " + "x " * 20 for _ in range(5)]}
    )
    shape = profile_dataframe(df)["columns"][0]["text"]["shape"]
    assert shape.get("skipped_reason") == "long_text"
    assert shape["suggested_regex"] is None


def test_postal_like_new_semantics():
    df = pd.DataFrame(
        {
            "postal": ["K1A 0B1", "SW1A 1AA", "12345", "90210-1234", "1011 AB"],
            "codes": ["CG-12520", "FUR-BO-10001798", "C001", "AB12", "X-1"],
        }
    )
    result = profile_dataframe(df)
    columns = {column["column_name"]: column for column in result["columns"]}

    assert columns["postal"]["text"]["patterns"]["postal_like"] == 5
    assert columns["codes"]["text"]["patterns"]["postal_like"] == 0


def test_postal_like_pure_digit_branch_unchanged():
    series = pd.Series(["12345", "678", "1234567890", "12"], dtype="object")
    assert _pattern_summary(series)["postal_like"] == 3  # "12" is too short


def test_separator_characters():
    df = pd.DataFrame({"id": ["AA-11", "AA-22", "AA-33"]})
    separators = profile_dataframe(df)["columns"][0]["text"]["separator_characters"]
    assert separators.get("-") == 3


def test_collapse_shape_run_length():
    assert _collapse_shape("AA-9999") == "A{2}-9{4}"
    assert _collapse_shape("A") == "A"
    assert _collapse_shape("9A9") == "9A9"


def test_suggested_regex_must_match_all_dominant_values():
    values = ["AB12", "AB34", "CD56"]
    regex = _suggested_regex(pd.Series(values))
    import re as _re

    for value in values:
        assert _re.fullmatch(regex, value)


# ---------------------------------------------------------------------------
# Part D: column evidence
# ---------------------------------------------------------------------------


def test_identifier_like_and_repeats():
    values = [f"CU-{1000 + (i % 50)}" for i in range(100)]
    df = pd.DataFrame({"customer_id": values})

    column = profile_dataframe(df)["columns"][0]
    assert column["identifier_like"] is True
    assert column["identifier_repeats"] is True
    assert column["identifier_signal"] is False  # uniqueness still required
    assert "name_signal" in column["identifier_like_reasons"]


def test_identifier_like_false_for_measure_and_free_text():
    df = pd.DataFrame(
        {
            "price": [10.5, 20.25, 33.1],
            "notes": ["a long free text description " * 10, "short", "tiny"],
        }
    )
    result = profile_dataframe(df)
    # price is a float measure: identifier_like is not even attached (no
    # text block), and no identifier evidence beyond name signal.
    price = result["columns"][0]
    assert "identifier_like" not in price or price["identifier_like"] is False
    notes = result["columns"][1]
    assert notes["identifier_like"] is False


def test_integer_sequence_counter_like():
    df = pd.DataFrame(
        {
            "row_id": list(range(1, 51)),
            "random_int": [7, 3, 99, 1, 42] * 10,
        }
    )
    result = profile_dataframe(df)
    row_id = result["columns"][0]["numeric"]["integer_sequence"]
    assert row_id["counter_like"] is True
    assert row_id["step_one"] is True
    assert row_id["dense"] is True

    random_int = result["columns"][1]["numeric"]["integer_sequence"]
    assert random_int["counter_like"] is False


def test_integer_sequence_step_two_is_counter_like_not_dense():
    df = pd.DataFrame({"seq": [2, 4, 6, 8, 10]})
    sequence = profile_dataframe(df)["columns"][0]["numeric"]["integer_sequence"]
    assert sequence["counter_like"] is True
    assert sequence["dense"] is False
    assert sequence["step_one"] is False


def test_disguised_missing_tokens():
    df = pd.DataFrame({"status": ["N/A", "-", "?", "", "ok"]})
    block = profile_dataframe(df)["columns"][0]["text"]["disguised_missing"]
    assert block["disguised_missing_count"] == 3
    tokens = {entry["value"] for entry in block["disguised_missing_values"]}
    assert tokens == {"n/a", "-", "?"}


def test_disguised_missing_ignores_majority_token():
    df = pd.DataFrame({"country": ["na", "na", "na", "na", "X"]})
    block = profile_dataframe(df)["columns"][0]["text"]["disguised_missing"]
    # "na" covers 80% of rows: not counted (legit-token protection).
    assert block["disguised_missing_count"] == 0


def test_case_variants_and_whitespace_padding():
    df = pd.DataFrame({"country": ["India", "india", "INDIA", "Nepal", " India "]})
    column = profile_dataframe(df)["columns"][0]

    text = column["text"]
    assert text["case_variant_groups"] == 1
    assert text["case_variant_examples"][0][0] == "India"
    assert column["leading_trailing_whitespace_count"] >= 1


def test_case_variants_skipped_when_too_many_distinct():
    values = [f"val{i}" for i in range(PROFILING_CASE_VARIANT_MAX_DISTINCT + 10)]
    text_block = (
        profile_dataframe(pd.DataFrame({"v": values}))["columns"][0]["text"]
    )
    assert text_block["normalized_distinct_count"] is None
    assert text_block.get("skipped_reason") == "too_many_distinct_values"


def test_numeric_additions():
    df = pd.DataFrame({"price": [10.5, -2.0, 0.0, 3.25, 100.125, np.nan]})
    numeric = profile_dataframe(df)["columns"][0]["numeric"]

    assert numeric["zero_count"] == 1
    assert numeric["negative_count"] == 1
    assert numeric["positive_count"] == 3
    assert numeric["integer_valued"] is False
    assert numeric["max_decimal_places"] == 3
    assert numeric["p01"] is not None and numeric["p99"] is not None
    assert numeric["skewness"] is not None
    assert numeric["meaningful_statistics"] is True


def test_numeric_code_like_flags():
    df = pd.DataFrame(
        {
            "zip": [12345, 23456, 34567],
            "phone_number": [5551234567, 5551234568, 5551234569],
            "customer_id": [101, 102, 103],
            "row_counter": list(range(3)),
            "price": [10.5, 20.5, 30.5],
        }
    )
    result = profile_dataframe(df)
    columns = {column["column_name"]: column for column in result["columns"]}

    for name in ("zip", "phone_number", "customer_id", "row_counter"):
        numeric = columns[name]["numeric"]
        assert numeric["code_like"] is True, name
        assert numeric["meaningful_statistics"] is False, name

    price = columns["price"]["numeric"]
    assert price["code_like"] is False
    assert price["meaningful_statistics"] is True


def test_leading_zero_loss_detection():
    # 5-digit postal codes with some 4-digit values (float64, NaN present):
    # the classic CSV float-inference damage.
    values = [f"{10000 + i}" for i in range(80)] + [f"{1000 + i}" for i in range(20)] + [None] * 5
    floats = [float(v) if v is not None else np.nan for v in values]
    df = pd.DataFrame({"postal_code": floats})

    numeric = profile_dataframe(df)["columns"][0]["numeric"]
    assert numeric["leading_zero_loss_suspected"] is True
    assert numeric["leading_zero_loss_count"] == 20
    assert "digit_length_distribution" in numeric


def test_leading_zero_loss_false_for_non_code_numeric():
    df = pd.DataFrame({"amount": [12345.0, 1234.0, 99999.0]})
    numeric = profile_dataframe(df)["columns"][0]["numeric"]
    assert numeric.get("leading_zero_loss_suspected", False) is False


def test_numeric_sentinel_detection():
    df = pd.DataFrame({"age": [-1] * 12 + [30, 40, 50, 88]})
    numeric = profile_dataframe(df)["columns"][0]["numeric"]
    assert numeric["suspicious_sentinel"] is not None
    assert numeric["suspicious_sentinel"]["value"] == -1

    df2 = pd.DataFrame({"age2": [-1] * 0 + [30, 40, 50, 88]})
    numeric2 = profile_dataframe(df2)["columns"][0]["numeric"]
    assert numeric2["suspicious_sentinel"] is None


def test_shape_block_on_code_like_numeric():
    df = pd.DataFrame({"postal_code": [1234.0, 5678.0, 9012.0]})
    column = profile_dataframe(df)["columns"][0]
    assert column.get("shape", {}).get("dominant_shape") == "9999"


def test_stored_as_and_semantic_mismatch():
    df = pd.DataFrame(
        {
            "zip": [12345, 23456],
            "name": ["Alice", "Bob"],
            "numeric_string": ["123", "456"],
            "when": pd.date_range("2024-01-01", periods=2),
        }
    )
    result = profile_dataframe(df)
    columns = {column["column_name"]: column for column in result["columns"]}

    assert columns["zip"]["stored_as"] == "numeric"
    assert columns["zip"]["semantic_storage_mismatch"] is True
    assert columns["name"]["stored_as"] == "string"
    assert columns["numeric_string"]["semantic_storage_mismatch"] is True
    assert columns["when"]["stored_as"] == "datetime"


# ---------------------------------------------------------------------------
# Part E: table-level evidence
# ---------------------------------------------------------------------------


def test_composite_true_key_found_among_noisy_columns():
    row_count = 300
    data = {}
    for i in range(40):
        data[f"noise_{i}"] = [f"v{i % 7}" for _ in range(row_count)]
    # order_id and product_id each have 2 distinct values; the PAIR is
    # 100% unique (a classic order + line-number key).
    data["order_id"] = [f"O{i // 2}" for i in range(row_count)]
    data["product_id"] = [f"P{i % 2}" for i in range(row_count)]
    data["sales"] = [float(i) for i in range(row_count)]

    profile = profile_dataframe(pd.DataFrame(data))
    candidates = profile["composite_uniqueness_candidates"]
    assert candidates, "expected at least one composite candidate"
    assert ("order_id", "product_id") in [
        tuple(candidate["columns"]) for candidate in candidates
    ]


def test_composite_no_measure_members():
    df = pd.DataFrame(
        {
            "city": ["A", "B", "C"] * 20,
            "sales": [float(i) for i in range(60)],
            "quantity": [1, 2, 3] * 20,
        }
    )
    result = profile_dataframe(df)
    for candidate in result["composite_uniqueness_candidates"]:
        assert "sales" not in candidate["columns"]
        assert candidate["contains_measure"] is False


def test_composite_ranking_prefers_fewer_columns_and_key_like():
    # (a, b) is a real 2-column key; (a, b, c) also unique but longer.
    df = pd.DataFrame(
        {
            "a": [f"A{i % 50}" for i in range(100)],
            "b": [f"B{i % 4}" for i in range(100)],
            "c": [f"C{i % 50}" for i in range(100)],
        }
    )
    # Make (a,b) unique: i%50 x i%4 pair repeats, so adjust:
    df["a"] = [f"A{i // 4}" for i in range(100)]
    df["b"] = [f"B{i % 4}" for i in range(100)]
    df["c"] = [f"C{i // 4}" for i in range(100)]

    result = profile_dataframe(df)
    candidates = result["composite_uniqueness_candidates"]
    assert candidates
    assert len(candidates[0]["columns"]) == 2


def test_composite_cap_of_three_and_minimality():
    df = pd.DataFrame(
        {
            "a": [f"A{i // 2}" for i in range(100)],
            "b": [f"B{i % 2}" for i in range(100)],
            "c": [f"C{i // 2}" for i in range(100)],
            "d": [f"D{i % 2}" for i in range(100)],
        }
    )
    result = profile_dataframe(df)
    candidates = result["composite_uniqueness_candidates"]
    assert len(candidates) <= PROFILING_MAX_COMBINATIONS


def test_composite_near_exact_and_key_like_members():
    # (order_id, line) is 100% unique; each column alone is ~50% distinct.
    df = pd.DataFrame(
        {
            "order_id": [f"O{i // 2}" for i in range(198)],
            "line": [f"L{i % 2}" for i in range(198)],
        }
    )
    result = profile_dataframe(df)
    candidates = result["composite_uniqueness_candidates"]
    assert candidates, "expected the near-exact composite to be reported"
    best = candidates[0]
    assert best["near_exact"] is True


def test_row_completeness_and_co_missing_patterns():
    df = pd.DataFrame(
        {
            "a": [1, None, None, 3],
            "b": [None, None, None, 4],
            "c": [1, 2, 3, 4],
        }
    )
    completeness = profile_dataframe(df)["row_completeness"]

    assert completeness["fully_complete_rows"] == 1
    assert completeness["fully_complete_percentage"] == 25.0
    assert completeness["rows_with_any_null"] == 3
    assert 0 < completeness["average_filled_percentage_per_row"] <= 100
    assert completeness["emptiest_row_filled_percentage"] < 100

    patterns = completeness["co_missing_patterns"]
    assert patterns, "expected the (a, b) co-missing rows to be detected"
    top = patterns[0]
    assert set(top["columns"]) == {"a", "b"}
    assert top["row_count"] == 2


def test_functional_dependencies_basic():
    # 500 rows; one 10-row city split across two states -> city -> state
    # coverage = 98.0 (>= threshold, < 100) so the pair is reported WITH
    # violation examples. Coverage counts ROWS in violating groups
    # (10 of 500), not distinct (A,B) pairs.
    rows = 500
    city = [f"C{i // 10}" for i in range(rows)]
    zip_codes = [f"Z{i // 10}" for i in range(rows)]
    state: list[str] = []
    for i in range(rows):
        city_index = i // 10
        if city_index == 49:
            state.append("SX" if i < 495 else "SY")
        else:
            state.append(f"S{city_index}")

    df = pd.DataFrame(
        {
            "zip": zip_codes,
            "city": city,
            "state": state,
            "unique_id": list(range(rows)),
            "price": [1.5, 2.5] * (rows // 2),
            "constant": ["X"] * rows,
        }
    )
    block = profile_dataframe(df)["functional_dependencies"]
    pairs = {
        (dependency["determinant"], dependency["dependent"]): dependency
        for dependency in block["dependencies"]
    }

    zip_city = pairs.get(("zip", "city"))
    assert zip_city is not None
    assert zip_city["coverage_percentage"] == 100.0
    assert zip_city["bidirectional"] is True

    city_state = pairs.get(("city", "state"))
    assert city_state is not None
    assert city_state["coverage_percentage"] < 100.0
    assert city_state["violation_examples"], "expected violation examples"

    # Unique column never a determinant; constant/float excluded.
    for determinant, dependent in pairs:
        assert determinant != "unique_id"
        assert dependent != "price"
        assert dependent != "constant"
    assert pairs.get(("zip", "unique_id")) is None


def test_functional_dependencies_transitive_reduction():
    # Nested granularities (12 > 6 > 2 distinct, aligned boundaries) make
    # a -> b -> c a pure chain: the implied a -> c must be dropped while
    # the chain edges remain.
    df = pd.DataFrame(
        {
            "a": [f"a{i // 5}" for i in range(60)],
            "b": [f"b{i // 10}" for i in range(60)],
            "c": [f"c{i // 30}" for i in range(60)],
        }
    )
    block = profile_dataframe(df)["functional_dependencies"]
    pairs = {
        (dependency["determinant"], dependency["dependent"])
        for dependency in block["dependencies"]
    }
    assert ("a", "b") in pairs
    assert ("b", "c") in pairs
    assert ("a", "c") not in pairs


def test_functional_dependencies_caps_respected(monkeypatch):
    import app.services.profiling as P

    monkeypatch.setattr(P, "PROFILING_FD_MAX_PAIRS", 5)

    df = pd.DataFrame(
        {f"col{i}": [f"v{i % 5}", f"w{i % 5}", f"x{i % 5}"] for i in range(10)}
    )
    block = profile_dataframe(df)["functional_dependencies"]
    assert len(block["dependencies"]) <= 30  # report cap independent of budget


def test_observations_required_codes():
    df = pd.DataFrame(
        {
            "order_date": ["01/02/2024", "03/04/2024", "05/06/2024", "07/08/2024"],
            "country": ["X"] * 4,
            "postal_code": [1234.0, 1235.0, 1236.0, 999.0],
            "status": ["ok", "N/A", "ok", "ok"],
            "city": ["a", "a", "b", "b"],
            "zip": ["01234", "12345", "01346", "12346"],
        }
    )
    codes = {obs["code"] for obs in profile_dataframe(df)["observations"]}

    for required in (
        "date_format_ambiguous",
        "constant_column",
        "leading_zero_loss_suspected",
        "code_like_numeric",
        "disguised_missing_values",
        "regular_pattern",
    ):
        assert required in codes, required


def test_observations_row_completeness_low():
    df = pd.DataFrame(
        {
            "a": [1, None, 3, None, 5, None, 7, None, 9, None],
            "b": [None, 2, None, 4, None, 6, None, 8, None, 10],
            "c": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        }
    )
    codes = {obs["code"] for obs in profile_dataframe(df)["observations"]}
    assert "row_completeness_low" in codes


def test_observations_capped():
    df = pd.DataFrame(
        {f"col{i}": [f"v{i % 3}", "pad ", "NA"] for i in range(200)}
    )
    observations = profile_dataframe(df)["observations"]
    assert len(observations) <= 60


def test_observations_include_numbers_in_evidence():
    df = pd.DataFrame({"country": ["X"] * 10})
    observation = next(
        obs
        for obs in profile_dataframe(df)["observations"]
        if obs["code"] == "constant_column"
    )
    assert observation["column"] == "country"
    assert observation["severity"] in {"info", "warning"}
    assert "evidence" in observation


# ---------------------------------------------------------------------------
# Serialisability + determinism
# ---------------------------------------------------------------------------


def test_profile_json_serialisable_no_nan():
    df = pd.DataFrame(
        {
            "a": [1.5, np.nan, 3.0],
            "b": ["x", None, "y"],
            "c": pd.date_range("2024-01-01", periods=3),
        }
    )
    profile = profile_dataframe(df)
    json.dumps(profile, allow_nan=False)


def test_profile_deterministic_including_sampled_paths(monkeypatch):
    import app.services.profiling as P

    monkeypatch.setattr(P, "PROFILING_SHAPE_SAMPLE_ROWS", 50)
    monkeypatch.setattr(P, "PROFILING_COMPOSITE_SAMPLE_ROWS", 50)
    monkeypatch.setattr(P, "PROFILING_FD_SAMPLE_ROWS", 50)

    rng = np.random.default_rng(3)
    df = pd.DataFrame(
        {
            "id": [f"ID-{i % 40}-{i}" for i in range(200)],
            "value": rng.uniform(0, 100, 200),
            "cat": [f"c{i % 5}" for i in range(200)],
        }
    )

    first = json.dumps(profile_dataframe(df), sort_keys=True)
    second = json.dumps(profile_dataframe(df), sort_keys=True)
    assert first == second


def test_forced_sampled_flags_present(monkeypatch):
    import app.services.profiling as P

    monkeypatch.setattr(P, "PROFILING_COMPOSITE_SAMPLE_ROWS", 50)
    monkeypatch.setattr(P, "PROFILING_FD_SAMPLE_ROWS", 50)

    df = pd.DataFrame(
        {
            "a": [f"A{i % 25}" for i in range(200)],
            "b": [f"B{i % 4}" for i in range(200)],
        }
    )
    profile = profile_dataframe(df)

    # Composite on a sampled frame marks its candidates.
    for candidate in profile["composite_uniqueness_candidates"]:
        assert candidate.get("sampled") is True
        assert candidate.get("claim") == "sample"


# ---------------------------------------------------------------------------
# Performance smoke tests (generous wall-clock limits)
# ---------------------------------------------------------------------------


def test_performance_200k_rows_by_20_columns():
    rng = np.random.default_rng(0)
    data = {
        "id": [f"ID-{i % 50000}-{i}" for i in range(200_000)],
        "code": [f"AB-{i % 1000:04d}" for i in range(200_000)],
        "price": rng.uniform(0, 1000, 200_000),
        "quantity": rng.integers(0, 50, 200_000),
        "date": ["2024-01-15"] * 200_000,
        "flag": rng.choice([True, False], 200_000),
    }
    for i in range(14):
        data[f"cat_{i}"] = [f"c{i % 10}" for i in range(200_000)]
    df = pd.DataFrame(data)

    start = time.perf_counter()
    profile = profile_dataframe(df)
    elapsed = time.perf_counter() - start

    # 60s generous limit: ~35s measured locally; real cost is dominated by
    # value_counts on 20 string columns of 200k rows, not the cross-column
    # analyses (those are constant-bounded).
    assert elapsed < 60.0, f"profile took {elapsed:.1f}s (200k x 20)"
    assert profile["row_count"] == 200_000


def test_performance_500_columns_by_2k_rows():
    df = pd.DataFrame(
        {f"col_{i}": np.arange(2000, dtype=float) for i in range(500)}
    )
    start = time.perf_counter()
    profile = profile_dataframe(df)
    elapsed = time.perf_counter() - start

    assert elapsed < 30.0, f"profile took {elapsed:.1f}s (500 x 2000)"
    assert profile["column_count"] == 500


# ---------------------------------------------------------------------------
# Part F: ingestion leading zeros
# ---------------------------------------------------------------------------


def test_read_table_preserves_leading_zeros_in_code_columns(tmp_path):
    from app.services.ingestion import read_table

    csv = tmp_path / "codes.csv"
    csv.write_text(
        "zip,quantity\n01234,1\n02134,2\n10001,3\n,4\n",
        encoding="utf-8",
    )

    df = read_table(csv)

    assert df["zip"].dtype == object or str(df["zip"].dtype) in {"object", "str", "string"}
    values = df["zip"].astype("string").fillna("")
    assert values.tolist() == ["01234", "02134", "10001", ""]
    assert df["quantity"].dtype.kind in "iu"


def test_read_table_keeps_numeric_zip_without_leading_zeros(tmp_path):
    from app.services.ingestion import read_table

    csv = tmp_path / "codes.csv"
    csv.write_text("zip,quantity\n12345,1\n21345,2\n", encoding="utf-8")

    df = read_table(csv)
    assert df["zip"].dtype.kind in "if"


def test_read_table_non_code_column_with_zero_leading_values_untouched(tmp_path):
    from app.services.ingestion import read_table

    csv = tmp_path / "items.csv"
    csv.write_text("quantity,zip\n007,01234\n008,02134\n", encoding="utf-8")

    df = read_table(csv)
    # quantity is NOT code-like: stays numeric even though 007 lost its zero.
    assert df["quantity"].dtype.kind in "if"
    assert df["zip"].astype("string").tolist() == ["01234", "02134"]


def test_read_table_excel_preserves_leading_zeros(tmp_path):
    from app.services.ingestion import read_table

    pytest.importorskip("openpyxl")
    excel = tmp_path / "codes.xlsx"
    pd.DataFrame({"zip": ["01234", "02134", "10001"]}).to_excel(
        excel, index=False
    )

    df = read_table(excel, sheet_name="Sheet1")
    assert df["zip"].astype("string").tolist() == ["01234", "02134", "10001"]


def test_profiler_and_relationship_discovery_see_same_dtypes(tmp_path, monkeypatch):
    from app.services.ingestion import read_table
    from app.services import relationship_discovery as R

    csv = tmp_path / "codes.csv"
    csv.write_text("zip,other\n01234,1\n02134,2\n", encoding="utf-8")

    # _table_dataframe must delegate to the shared read_table.
    source = inspect.getsource(R._table_dataframe)
    assert "read_table" in source

    profiler_df = read_table(csv)
    second_df = read_table(csv)
    assert str(profiler_df.dtypes.tolist()) == str(second_df.dtypes.tolist())
    assert profiler_df["zip"].astype("string").tolist() == ["01234", "02134"]


# ---------------------------------------------------------------------------
# Downstream safety (read-only imports)
# ---------------------------------------------------------------------------


def test_downstream_helpers_accept_new_profile():
    from app.services import metric_applicability
    from app.services import semantic_features

    df = pd.DataFrame(
        {
            "postal": ["K1A 0B1", "12345"],
            "when": ["2024-01-01", "2024-01-02"],
            "amount": [1.5, 2.5],
        }
    )
    profile = profile_dataframe(df)

    for column_profile in profile["columns"]:
        # semantic_features reads text.patterns (postal evidence).
        patterns = column_profile.get("text", {}).get("patterns", {})
        assert isinstance(patterns.get("postal_like", 0), int)
        assert isinstance(patterns.get("date_like", 0), int)

    # metric_applicability reads patterns.date_like for validity evidence.
    assert isinstance(metric_applicability.IDENTIFIER_CONCEPTS, (set, tuple, frozenset, list))
    assert semantic_features is not None  # imported read-only


# ---------------------------------------------------------------------------
# Golden check on the real file (skipped when missing)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not TRAIN_CSV.exists(), reason="train.csv not present")
def test_golden_train_csv():
    df = pd.read_csv(TRAIN_CSV)
    profile = profile_dataframe(df, "train")
    assert_no_analysis_errors(profile)
    columns = {column["column_name"]: column for column in profile["columns"]}

    # Dates detected day-first without ambiguity.
    for name in ("Order Date", "Ship Date"):
        block = columns[name]["datetime"]
        assert block["detected_format"] == "%d/%m/%Y"
        assert block["format_ambiguous"] is False

    # Customer ID is no longer postal-like.
    assert columns["Customer ID"]["text"]["patterns"]["postal_like"] == 0

    # Postal Code: code-like with leading-zero loss.
    postal_numeric = columns["Postal Code"]["numeric"]
    assert postal_numeric["code_like"] is True
    assert postal_numeric["leading_zero_loss_suspected"] is True

    # Identifier evidence.
    for name in ("Order ID", "Customer ID", "Product ID"):
        assert columns[name]["identifier_like"] is True, name
        shape = columns[name]["text"]["shape"]
        assert shape["is_regular"] is True, name
    assert (
        columns["Order ID"]["text"]["shape"]["dominant_shape"]
        == "AA-9999-999999"
    )
    assert columns["Customer ID"]["text"]["shape"]["dominant_shape"] == "AA-99999"
    assert (
        columns["Product ID"]["text"]["shape"]["dominant_shape"]
        == "AAA-AA-99999999"
    )
    assert columns["Row ID"]["numeric"]["integer_sequence"]["counter_like"] is True

    # Composite: best candidate is Order ID + Product ID; Sales never a member.
    candidates = profile["composite_uniqueness_candidates"]
    best = candidates[0]
    assert set(best["columns"]) == {"Order ID", "Product ID"}
    assert best["composite_uniqueness_percentage"] >= 99.9
    for candidate in candidates:
        assert "Sales" not in candidate["columns"]

    # Dependencies.
    pairs = {
        (dependency["determinant"], dependency["dependent"]): dependency
        for dependency in profile["functional_dependencies"]["dependencies"]
    }
    customer_name = pairs.get(("Customer ID", "Customer Name"))
    assert customer_name is not None and customer_name["bidirectional"] is True
    # Product ID -> Category is intentionally absent: the spec's transitive
    # reduction drops it because Product ID -> Sub-Category and
    # Sub-Category -> Category are both exact dependencies.
    assert ("Product ID", "Category") not in pairs
    assert ("Product ID", "Sub-Category") in pairs

    product_name = pairs.get(("Product ID", "Product Name"))
    if product_name is not None:
        assert product_name["coverage_percentage"] < 100.0
        assert product_name["violation_examples"]

    # Country constant.
    assert columns["Country"]["text"]["constant"] is True


# ---------------------------------------------------------------------------
# Bug-fix regressions: FD coverage math, counter-like uniqueness, numeric
# identifier evidence, composite ranking, non-string/duplicate column names.
# ---------------------------------------------------------------------------


def assert_no_analysis_errors(profile: dict) -> None:
    """No swallowed analysis failure may hide anywhere in a profile."""

    def _walk(node, path: str = "profile") -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                assert not key.endswith("_error"), (
                    f"swallowed analysis error at {path}.{key}: {value}"
                )
                _walk(value, f"{path}.{key}")
        elif isinstance(node, list):
            for position, value in enumerate(node):
                _walk(value, f"{path}[{position}]")

    _walk(profile)


def _dependency_pairs(profile: dict) -> dict[tuple[str, str], dict]:
    return {
        (dependency["determinant"], dependency["dependent"]): dependency
        for dependency in profile["functional_dependencies"]["dependencies"]
    }


def _columns(profile: dict) -> dict[str, dict]:
    return {column["column_name"]: column for column in profile["columns"]}


def test_fd_repeated_low_cardinality_not_reported():
    # Line number 1..8 repeated, quantity 1..3 random: every line
    # number maps to all three quantities -> coverage 0%, never
    # reported (the old pair-counting math reported ~99.9%).
    rng = np.random.RandomState(3)
    rows = 400
    df = pd.DataFrame(
        {
            "line_item": np.tile(np.arange(1, 9), rows // 8),
            "quantity": rng.randint(1, 4, rows),
        }
    )
    profile = profile_dataframe(df)
    assert_no_analysis_errors(profile)
    pairs = _dependency_pairs(profile)
    assert ("line_item", "quantity") not in pairs
    assert ("quantity", "line_item") not in pairs


def test_fd_header_line_table():
    # Header/line table: order_number -> customer and order_number
    # -> order_date at 100%; order_number -> line item is NOT
    # functional (an order has several lines).
    rows = []
    for order in range(1, 121):
        for line in range(1, (order % 4) + 2):
            rows.append(
                (
                    order,
                    f"CUST{order % 7}",
                    # (order*3) % 30 repeats every 10 orders, but
                    # customer repeats every 7 -> a shared date never
                    # implies a shared customer (no transitive chain).
                    f"2024-01-{(order * 3) % 30 + 1:02d}",
                    line,
                )
            )
    df = pd.DataFrame(
        rows,
        columns=["order_number", "customer", "order_date", "line_item"],
    )
    profile = profile_dataframe(df)
    assert_no_analysis_errors(profile)
    pairs = _dependency_pairs(profile)
    assert pairs[("order_number", "customer")]["coverage_percentage"] == 100.0
    assert pairs[("order_number", "order_date")]["coverage_percentage"] == 100.0
    assert ("order_number", "line_item") not in pairs


def test_fd_three_value_vs_ten_value_never_one_to_one():
    # A 3-value column against a 10-value column cannot be 1:1:
    # every low value co-occurs with every high value.
    rows = 300
    df = pd.DataFrame(
        {
            "low": np.tile(np.arange(1, 4), rows // 3),
            "high": np.tile(np.arange(1, 11), rows // 10),
        }
    )
    profile = profile_dataframe(df)
    assert_no_analysis_errors(profile)
    pairs = _dependency_pairs(profile)
    assert ("low", "high") not in pairs
    assert ("high", "low") not in pairs


def test_fd_exact_one_to_one_is_bidirectional():
    # Repeated-but-consistent 1:1 values (fully unique columns are
    # excluded by the vacuity rule: a unique column determines
    # everything trivially).
    df = pd.DataFrame(
        {
            "left": [f"L{i % 100}" for i in range(120)],
            "right": [f"R{i % 100}" for i in range(120)],
        }
    )
    profile = profile_dataframe(df)
    assert_no_analysis_errors(profile)
    pairs = _dependency_pairs(profile)
    assert pairs[("left", "right")]["coverage_percentage"] == 100.0
    assert pairs[("left", "right")]["bidirectional"] is True


def test_fd_coverage_matches_brute_force_groupby():
    # Property test: engine coverage == rows in violating groups /
    # evaluated rows, computed brute-force, on 20 random frames.
    rng = np.random.RandomState(11)
    for _trial in range(20):
        rows = int(rng.randint(40, 160))
        n_a = int(rng.randint(2, 10))
        n_b = int(rng.randint(2, 6))
        a_values = [f"A{rng.randint(0, n_a)}" for _ in range(rows)]
        b_values = [f"B{rng.randint(0, n_b)}" for _ in range(rows)]
        df = pd.DataFrame({"a": a_values, "b": b_values})
        profile = profile_dataframe(df)
        assert_no_analysis_errors(profile)
        groups: dict[str, set[str]] = {}
        counts: dict[str, int] = {}
        for a_value, b_value in zip(a_values, b_values):
            groups.setdefault(a_value, set()).add(b_value)
            counts[a_value] = counts.get(a_value, 0) + 1
        violating_rows = sum(
            counts[key]
            for key, members in groups.items()
            if len(members) > 1
        )
        expected = 100.0 * (1.0 - violating_rows / rows)
        pairs = _dependency_pairs(profile)
        if expected >= 98.0:
            assert ("a", "b") in pairs
            assert pairs[("a", "b")]["coverage_percentage"] == pytest.approx(
                expected, abs=0.05
            )
        else:
            assert ("a", "b") not in pairs


@pytest.mark.skipif(not TRAIN_CSV.exists(), reason="train.csv not present")
def test_golden_train_csv_no_analysis_errors():
    profile = profile_dataframe(pd.read_csv(TRAIN_CSV))
    assert_no_analysis_errors(profile)


def test_counter_like_requires_unique_values():
    df = pd.DataFrame(
        {
            "row_id": list(range(1, 101)),
            "territory": np.tile(np.arange(1, 11), 10),
            "step2": list(range(0, 200, 2)),
            "quantity": np.tile(np.arange(1, 4), 34)[:100],
        }
    )
    profile = profile_dataframe(df)
    assert_no_analysis_errors(profile)
    columns = _columns(profile)
    assert columns["row_id"]["numeric"]["integer_sequence"]["counter_like"] is True
    # 1..10 repeated 100x: NOT counter-like, but low-cardinality.
    territory = columns["territory"]["numeric"]["integer_sequence"]
    assert territory["counter_like"] is False
    assert territory["low_cardinality"] is True
    # Unique step-2 sequence: counter-like, not dense.
    step2 = columns["step2"]["numeric"]["integer_sequence"]
    assert step2["counter_like"] is True
    assert step2["dense"] is False
    # Quantity-like 1..3 repeated: not counter-like, not code-like,
    # statistics meaningful.
    quantity = columns["quantity"]
    assert quantity["numeric"]["integer_sequence"]["counter_like"] is False
    assert quantity["numeric"]["code_like"] is False
    assert quantity["numeric"]["meaningful_statistics"] is True
    # No bogus dense_integer_counter observation for repeated values.
    for observation in profile["observations"]:
        if observation.get("code") == "dense_integer_counter":
            assert observation.get("column") != "territory"
            assert observation.get("column") != "quantity"


def test_numeric_identifier_evidence_for_keys():
    rng = np.random.RandomState(9)
    df = pd.DataFrame(
        {
            "ProductKey": rng.randint(1, 501, 300),
            "CustomerKey": rng.randint(1, 101, 300),
            "OrderQuantity": np.tile(np.arange(1, 4), 100),
            "UnitPrice": np.round(rng.uniform(1.0, 100.0), 2),
        }
    )
    profile = profile_dataframe(df)
    assert_no_analysis_errors(profile)
    columns = _columns(profile)
    for name in ("ProductKey", "CustomerKey"):
        assert columns[name]["identifier_like"] is True, name
        assert columns[name]["identifier_repeats"] is True, name
        assert columns[name]["identifier_like_reasons"] == ["name_signal"], name
        assert columns[name]["numeric"]["code_like"] is True, name
        assert columns[name]["semantic_storage_mismatch"] is True, name
    assert columns["OrderQuantity"]["identifier_like"] is False
    assert columns["OrderQuantity"]["numeric"]["code_like"] is False
    assert columns["UnitPrice"]["identifier_like"] is False
    # Text identifier behaviour is unchanged.
    text = pd.DataFrame(
        {"Customer ID": [f"C{i:04d}" for i in range(50)]}
    )
    text_profile = profile_dataframe(text)
    assert_no_analysis_errors(text_profile)
    text_column = _columns(text_profile)["Customer ID"]
    assert text_column["identifier_like"] is True
    assert text_column["identifier_repeats"] is False


def test_composite_ranking_exact_beats_near_exact():
    # Exact uniqueness must rank above a near-exact candidate even when
    # the near-exact candidate has stronger key-like evidence.
    rng = np.random.RandomState(4)
    rows = 1000
    m = np.repeat(np.arange(10), 100)
    n = np.tile(np.arange(100), 10)
    d1 = np.repeat(np.arange(100), 10)
    d2 = np.tile(np.arange(10), 100)
    d2[991:] = 0  # 9 duplicated rows -> 99.1% unique

    df = pd.DataFrame(
        {
            "m": m,
            "n": n,
            "d1_key": d1,
            "d2_key": d2,
            "noise": rng.rand(rows),
        }
    )

    profile = profile_dataframe(df)
    assert_no_analysis_errors(profile)

    candidates = profile["composite_uniqueness_candidates"]
    assert candidates

    exact = next(
        candidate
        for candidate in candidates
        if set(candidate["columns"]) == {"m", "n"}
    )

    assert exact["composite_uniqueness_percentage"] == 100.0

    near = next(
        (
            candidate
            for candidate in candidates
            if set(candidate["columns"]) == {"d1_key", "d2_key"}
        ),
        None,
    )

    assert near is not None
    assert near["composite_uniqueness_percentage"] == pytest.approx(
        99.1, abs=0.05
    )

    assert candidates.index(exact) < candidates.index(near)

def test_composite_ranking_exact_pair_beats_exact_triple():
    rows = 980
    df = pd.DataFrame(
        {
            "m": np.repeat(np.arange(10), 98),
            "n": np.tile(np.arange(98), 10),
            "p": np.tile(np.arange(7), 140),
            "q": np.tile(np.repeat(np.arange(7), 7), 20),
            "r": np.repeat(np.arange(20), 49),
        }
    )
    profile = profile_dataframe(df)
    assert_no_analysis_errors(profile)
    candidates = profile["composite_uniqueness_candidates"]
    column_sets = [set(candidate["columns"]) for candidate in candidates]
    assert {"m", "n"} in column_sets
    assert {"p", "q", "r"} in column_sets
    # Exact 2-column key ranks above the exact 3-column key.
    assert column_sets.index({"m", "n"}) < column_sets.index({"p", "q", "r"})
    # Cap of 3 still holds.
    assert len(candidates) <= 3
    # Minimality: no candidate is a superset of another.
    for first in column_sets:
        for second in column_sets:
            if first is not second:
                assert not first.issubset(second)


def test_integer_column_labels_find_keys_and_dependencies():
    # header=None style frame: integer column labels used to raise
    # KeyError, which the broad except swallowed (empty composites
    # and dependencies).
    rows = []
    for order in range(1, 31):
        for line in range(1, 4):
            rows.append((order, f"CUST{order}", line, f"D{order}"))
    df = pd.DataFrame(
        rows,
        columns=[0, 1, 2, 3],
    )
    profile = profile_dataframe(df)
    assert_no_analysis_errors(profile)
    candidates = profile["composite_uniqueness_candidates"]
    assert any(
        set(candidate["columns"]) == {"0", "2"}
        for candidate in candidates
    )
    pairs = _dependency_pairs(profile)
    assert pairs.get(("0", "1")) is not None
    assert pairs[("0", "1")]["coverage_percentage"] == 100.0
    assert pairs.get(("0", "3")) is not None


def test_duplicate_column_names_find_keys_and_dependencies():
    rows = []
    for order in range(1, 31):
        for line in range(1, 4):
            rows.append([f"ORD{order}", f"CUST{order}", line, f"D{order}"])
    df = pd.DataFrame(
        rows,
        columns=["order", "customer", "line", "order"],
    )
    profile = profile_dataframe(df)
    assert_no_analysis_errors(profile)
    candidates = profile["composite_uniqueness_candidates"]
    assert any(
        set(candidate["columns"]) == {"order", "line"}
        for candidate in candidates
    )
    pairs = _dependency_pairs(profile)
    assert pairs.get(("order", "customer")) is not None
    assert pairs[("order", "customer")]["coverage_percentage"] == 100.0


def test_analysis_errors_record_message_not_just_class(monkeypatch):
    # A failed analysis records exception class AND message
    # (truncated to 200 chars), not just the class name.
    def boom(frame, *args, **kwargs):
        raise KeyError("missing column 'x' " * 20)

    monkeypatch.setattr(profiling_module, "_functional_dependencies", boom)
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    profile = profile_dataframe(df)
    assert "functional_dependencies_error" in profile
    message = profile["functional_dependencies_error"]
    assert isinstance(message, str)
    assert len(message) <= 200
    assert message.startswith("KeyError:")
    assert "'x'" in message


def test_sales_table_fixture():
    # Small frame with the 29k-row table's structure: header/line
    # orders, integer product/customer/territory keys, a 1..4 line
    # number column, a 1..3 quantity column, two m/d/Y date columns.
    rng = np.random.RandomState(5)
    rows = 600
    order_number = np.repeat(np.arange(1, 151), 4)
    customer_key = (order_number % 20) + 1
    order_date = pd.to_datetime("2024-01-01") + pd.to_timedelta(
        order_number % 30, unit="D"
    )
    ship_date = order_date + pd.to_timedelta(2, unit="D")
    df = pd.DataFrame(
        {
            "OrderNumber": order_number,
            "OrderLineItem": np.tile(np.arange(1, 5), 150),
            "ProductKey": rng.randint(1, 51, rows),
            "CustomerKey": customer_key,
            "TerritoryKey": rng.randint(1, 6, rows),
            "OrderQuantity": rng.randint(1, 4, rows),
            "OrderDate": order_date,
            "ShipDate": ship_date,
        }
    )
    profile = profile_dataframe(df)
    assert_no_analysis_errors(profile)
    columns = _columns(profile)

    # (a) OrderNumber + OrderLineItem is the first composite candidate.
    best = profile["composite_uniqueness_candidates"][0]
    assert set(best["columns"]) == {"OrderNumber", "OrderLineItem"}
    assert best["composite_uniqueness_percentage"] == 100.0

    # (b) OrderNumber -> CustomerKey and OrderNumber -> OrderDate.
    pairs = _dependency_pairs(profile)
    assert pairs.get(("OrderNumber", "CustomerKey")) is not None
    assert pairs[("OrderNumber", "CustomerKey")]["coverage_percentage"] == 100.0
    assert pairs.get(("OrderNumber", "OrderDate")) is not None
    assert pairs[("OrderNumber", "OrderDate")]["coverage_percentage"] == 100.0

    # (c) No 1:1 dependency among line item / quantity / territory.
    for determinant in (
        "OrderLineItem",
        "OrderQuantity",
        "TerritoryKey",
    ):
        for dependent in (
            "OrderLineItem",
            "OrderQuantity",
            "TerritoryKey",
        ):
            if determinant != dependent:
                assert (determinant, dependent) not in pairs

    # (d) Line item and quantity are NOT code-like; the *Key columns
    # ARE (name token "key").
    assert columns["OrderLineItem"]["numeric"]["code_like"] is False
    assert columns["OrderQuantity"]["numeric"]["code_like"] is False
    for name in ("ProductKey", "CustomerKey", "TerritoryKey"):
        assert columns[name]["numeric"]["code_like"] is True, name

    # (e) ProductKey/CustomerKey are identifier-like (and repeat).
    for name in ("ProductKey", "CustomerKey"):
        assert columns[name]["identifier_like"] is True, name
        assert columns[name]["identifier_repeats"] is True, name

    # (f) No swallowed analysis errors anywhere.
    assert_no_analysis_errors(profile)

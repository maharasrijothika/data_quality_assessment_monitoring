import inspect

import pandas as pd

from app.services import profiling as profiling_module
from app.services.profiling import (
    PROFILING_MAX_COMBINATIONS,
    identifier_name_signal_from_name,
    profile_dataframe,
)


def test_profile_basic_dataset():
    df = pd.DataFrame(
        {
            "customer_id": [1, 2, 3, None],
            "email": [
                "a@test.com",
                "bad",
                "b@test.com",
                " ",
            ],
            "age": [20, 30, 40, 200],
        }
    )

    profile = profile_dataframe(df, "customers")

    assert profile["table_name"] == "customers"
    assert profile["row_count"] == 4
    assert profile["column_count"] == 3


def test_profile_completeness_categories():
    df = pd.DataFrame(
        {
            "value": [
                None,
                "",
                "   ",
                "valid",
            ]
        }
    )

    profile = profile_dataframe(df, "test")

    column = profile["columns"][0]

    assert column["null_count"] == 1
    assert column["empty_string_count"] == 1
    assert column["whitespace_only_count"] == 1


def test_profile_numeric_statistics_and_outliers():
    df = pd.DataFrame(
        {
            "age": [20, 30, 40, 200]
        }
    )

    profile = profile_dataframe(df, "customers")

    numeric = profile["columns"][0]["numeric"]

    assert numeric["min"] == 20
    assert numeric["max"] == 200
    assert numeric["median"] == 35
    assert numeric["outlier_count_iqr"] == 1


def test_profile_identifier_signal():
    df = pd.DataFrame(
        {
            "customer_id": [1, 2, 3, 4],
            "name": ["A", "B", "C", "D"],
        }
    )

    profile = profile_dataframe(df, "customers")

    customer_id = profile["columns"][0]
    name = profile["columns"][1]

    assert customer_id["identifier_signal"] is True
    assert name["identifier_signal"] is False


def test_profile_text_patterns():
    df = pd.DataFrame(
        {
            "email": [
                "a@test.com",
                "b@test.com",
                "invalid",
                " ",
            ]
        }
    )

    profile = profile_dataframe(df, "customers")

    patterns = profile["columns"][0]["text"]["patterns"]

    assert patterns["email_like"] == 2
    assert patterns["contains_whitespace"] == 1


def test_profile_empty_dataframe():
    df = pd.DataFrame(columns=["id", "name"])

    profile = profile_dataframe(df, "empty")

    assert profile["row_count"] == 0
    assert profile["column_count"] == 2
    assert len(profile["columns"]) == 2
    assert profile["complete_duplicate_rows"] == 0
    assert profile["composite_uniqueness_candidates"] == []


def test_profile_detects_date_like_string_column():
    df = pd.DataFrame({
        "OrderDate": [
            "1996-07-04",
            "1996-07-05",
            "1996-07-08",
            "1996-07-09",
        ]
    })

    result = profile_dataframe(df)

    column = result["columns"][0]

    assert column["data_type"] in {"object", "str", "string"}
    assert "datetime" in column

    assert column["datetime"]["date_like_count"] == 4
    assert column["datetime"]["date_like_percentage"] == 100.0
    assert column["datetime"]["min"].startswith("1996-07-04")
    assert column["datetime"]["max"].startswith("1996-07-09")


def test_profile_detects_datetime_string_column():
    df = pd.DataFrame({
        "created_at": [
            "2024-01-18 10:30:00",
            "2024-01-19 11:45:00",
            "2024-01-20 12:15:00",
        ]
    })

    result = profile_dataframe(df)

    column = result["columns"][0]

    assert "datetime" in column
    assert column["datetime"]["date_like_count"] == 3
    assert column["datetime"]["date_like_percentage"] == 100.0


def test_profile_does_not_classify_mixed_text_as_datetime():
    df = pd.DataFrame({
        "value": [
            "2024-01-18",
            "hello",
            "customer",
            "ABC123",
        ]
    })

    result = profile_dataframe(df)

    column = result["columns"][0]

    assert "datetime" not in column


def test_profile_does_not_convert_original_string_dtype():
    df = pd.DataFrame({
        "OrderDate": [
            "1996-07-04",
            "1996-07-05",
        ]
    })

    original_dtype = df["OrderDate"].dtype
    original_values = df["OrderDate"].tolist()

    result = profile_dataframe(df)

    column = result["columns"][0]

    # Profiling must detect the semantic pattern,
    # but must not modify the original dataframe.
    assert df["OrderDate"].dtype == original_dtype
    assert df["OrderDate"].tolist() == original_values

    assert column["data_type"] in {"object", "str", "string"}
    assert "datetime" in column


def test_profile_duplicate_statistics():
    df = pd.DataFrame({
        "customer_id": [
            "C001",
            "C001",
            "C001",
            "C002",
            "C003",
        ]
    })

    result = profile_dataframe(df)

    column = result["columns"][0]

    # All rows belonging to a duplicated value group.
    assert column["duplicate_count"] == 3

    # Duplicate occurrences beyond the first occurrence.
    assert column["duplicate_excess_count"] == 2


def test_profile_identifier_signal_requires_name_evidence():
    df = pd.DataFrame({
        "customer_id": [
            "C001",
            "C002",
            "C003",
            "C004",
        ]
    })

    result = profile_dataframe(df)

    column = result["columns"][0]

    assert column["identifier_signal"] is True


def test_profile_unique_non_identifier_column_is_not_identifier():
    df = pd.DataFrame({
        "email_address": [
            "a@example.com",
            "b@example.com",
            "c@example.com",
            "d@example.com",
        ]
    })

    result = profile_dataframe(df)

    column = result["columns"][0]

    assert column["identifier_signal"] is False


def test_profile_duplicate_identifier_is_not_strong_identifier_signal():
    df = pd.DataFrame({
        "customer_id": [
            "C001",
            "C001",
            "C002",
            "C003",
        ]
    })

    result = profile_dataframe(df)

    column = result["columns"][0]

    assert column["identifier_signal"] is False


def test_profile_identifier_evidence():
    df = pd.DataFrame({
        "customer_id": [
            "C001",
            "C002",
            "C003",
            "C004",
        ]
    })

    result = profile_dataframe(df)

    column = result["columns"][0]

    assert column["identifier_signal"] is True
    assert column["identifier_name_signal"] is True
    assert column["identifier_completeness_percentage"] == 100.0
    assert column["identifier_uniqueness_percentage"] == 100.0


def test_profile_identifier_evidence_with_nulls():
    df = pd.DataFrame({
        "customer_id": [
            "C001",
            "C002",
            None,
            "C004",
        ]
    })

    result = profile_dataframe(df)

    column = result["columns"][0]

    assert column["identifier_name_signal"] is True
    assert column["identifier_completeness_percentage"] == 75.0
    assert column["identifier_uniqueness_percentage"] == 100.0


def test_profile_non_identifier_column_has_no_identifier_name_signal():
    df = pd.DataFrame({
        "email_address": [
            "a@example.com",
            "b@example.com",
            "c@example.com",
            "d@example.com",
        ]
    })

    result = profile_dataframe(df)

    column = result["columns"][0]

    assert column["identifier_name_signal"] is False


# ---------------------------------------------------------------------------
# Token-aware identifier NAME matching (audit point 3)
# ---------------------------------------------------------------------------


def test_identifier_name_signal_is_token_aware():
    # Positive cases: token-boundary matches.
    assert identifier_name_signal_from_name("customer_id") is True
    assert identifier_name_signal_from_name("product_code") is True
    assert identifier_name_signal_from_name("Loyalty Number") is True
    assert identifier_name_signal_from_name("LoyaltyNumber") is True
    assert identifier_name_signal_from_name("CustomerID") is True
    assert identifier_name_signal_from_name("VideoId") is True
    assert identifier_name_signal_from_name("reference_code") is True

    # Negative cases: substring false positives must NOT match.
    assert identifier_name_signal_from_name("note") is False
    assert identifier_name_signal_from_name("annotation") is False
    assert identifier_name_signal_from_name("humidity") is False
    assert identifier_name_signal_from_name("description") is False
    assert identifier_name_signal_from_name("denominator") is False


def test_profile_identifier_name_matching_avoids_substrings():
    df = pd.DataFrame({
        "description": [
            "first",
            "second",
            "third",
            "fourth",
        ]
    })

    result = profile_dataframe(df)

    column = result["columns"][0]

    assert column["identifier_name_signal"] is False


def test_profile_identifier_name_matching_detects_identifier_tokens():
    df = pd.DataFrame({
        "customer_id": [
            "C001",
            "C002",
            "C003",
            "C004",
        ]
    })

    result = profile_dataframe(df)

    column = result["columns"][0]

    assert column["identifier_name_signal"] is True


# ---------------------------------------------------------------------------
# Complete duplicate rows at table level (audit point 4)
# ---------------------------------------------------------------------------


def test_profile_complete_duplicate_rows():
    df = pd.DataFrame({
        "customer_id": ["C001", "C001", "C002"],
        "country": ["Canada", "Canada", "Germany"],
    })

    result = profile_dataframe(df, "customers")

    # Two rows are full row duplicates of each other.
    assert result["complete_duplicate_rows"] == 2
    assert result["complete_duplicate_excess_count"] == 1


def test_profile_repeated_column_values_are_not_complete_duplicates():
    df = pd.DataFrame({
        "customer_id": ["C001", "C001", "C002"],
        "month": [1, 2, 1],
    })

    result = profile_dataframe(df, "activity")

    # The key repeats (monthly grain), but no COMPLETE row is duplicated.
    assert result["complete_duplicate_rows"] == 0
    assert result["complete_duplicate_excess_count"] == 0

    column = result["columns"][0]
    assert column["duplicate_count"] == 2
    assert column["duplicate_excess_count"] == 1
    assert column["identifier_signal"] is False


# ---------------------------------------------------------------------------
# Lightweight categorical profiling (audit point 5)
# ---------------------------------------------------------------------------


def test_profile_categorical_block_for_low_cardinality_column():
    df = pd.DataFrame({
        "Country": [
            "Canada",
            "Canada",
            "Canada",
            "Germany",
        ]
    })

    result = profile_dataframe(df)

    column = result["columns"][0]
    categorical = column["categorical"]

    assert categorical["category_count"] == 2
    assert categorical["top_values"][0]["value"] == "Canada"
    assert categorical["top_values"][0]["count"] == 3
    assert categorical["top_values"][0]["percentage"] == 75.0
    assert categorical["top_values_returned"] == 2


def test_profile_categorical_block_limited_to_top_n():
    values = [f"cat_{i}" for i in range(20)] + ["cat_0"] * 5
    df = pd.DataFrame({"Category": values})

    result = profile_dataframe(df)

    categorical = result["columns"][0]["categorical"]

    assert categorical["category_count"] == 20
    assert categorical["top_values_returned"] <= 10


def test_profile_no_categorical_block_for_high_cardinality_column():
    df = pd.DataFrame({
        "Email": [f"user{i}@example.com" for i in range(60)]
    })

    result = profile_dataframe(df)

    column = result["columns"][0]

    assert "categorical" not in column


# ---------------------------------------------------------------------------
# Constant vs near-constant (audit point 7)
# ---------------------------------------------------------------------------


def test_profile_constant_numeric_column():
    df = pd.DataFrame({"flag": [7, 7, 7, 7]})

    result = profile_dataframe(df)

    numeric = result["columns"][0]["numeric"]

    assert numeric["constant"] is True
    assert numeric["near_constant"] is False


def test_profile_near_constant_numeric_column():
    # One dominant value (98% of rows) with another value also present.
    df = pd.DataFrame({"flag": [0] * 98 + [1] * 2})

    result = profile_dataframe(df)

    numeric = result["columns"][0]["numeric"]

    assert numeric["constant"] is False
    assert numeric["near_constant"] is True
    assert numeric["top_value_share_percentage"] == 98.0


def test_profile_varied_numeric_column_is_not_near_constant():
    df = pd.DataFrame({"value": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})

    result = profile_dataframe(df)

    numeric = result["columns"][0]["numeric"]

    assert numeric["constant"] is False
    assert numeric["near_constant"] is False


# ---------------------------------------------------------------------------
# Generic composite uniqueness profiling (audit point 8)
# ---------------------------------------------------------------------------


def test_profile_fact_table_composite_key_is_detected():
    """Monthly fact-table shape: single key repeats, composite is unique."""
    df = pd.DataFrame({
        "Loyalty Number": [1, 1, 1, 1, 2, 2],
        "Year": [2017, 2017, 2018, 2018, 2017, 2017],
        "Month": [1, 2, 1, 2, 1, 2],
        "Total Flights": [0, 2, 1, 0, 3, 5],
    })

    result = profile_dataframe(df, "Customer Flight Activity")

    # The single key repeats by design: NOT an identifier in this table.
    loyalty = result["columns"][0]
    assert loyalty["identifier_name_signal"] is True
    assert loyalty["identifier_signal"] is False

    candidates = {
        tuple(candidate["columns"]): candidate
        for candidate in result["composite_uniqueness_candidates"]
    }

    assert ("Loyalty Number", "Year", "Month") in candidates
    composite = candidates[("Loyalty Number", "Year", "Month")]
    assert composite["composite_distinct_count"] == 6
    assert composite["composite_uniqueness_percentage"] == 100.0
    assert composite["duplicate_composite_rows"] == 0
    assert composite["duplicate_composite_excess_count"] == 0


def test_profile_composite_duplicate_excess_is_reported():
    # 200 distinct (customer, year) combinations plus one repeated row.
    customer_ids = list(range(100)) * 2
    years = [2017] * 100 + [2018] * 100
    customer_ids.append(0)
    years.append(2017)

    df = pd.DataFrame({
        "CustomerID": customer_ids,
        "Year": years,
    })

    result = profile_dataframe(df)

    candidates = result["composite_uniqueness_candidates"]
    assert len(candidates) == 1

    composite = candidates[0]
    assert composite["columns"] == ["CustomerID", "Year"]
    assert composite["composite_distinct_count"] == 200
    assert composite["composite_uniqueness_percentage"] > 99.0
    assert composite["duplicate_composite_rows"] == 2
    assert composite["duplicate_composite_excess_count"] == 1


def test_profile_composite_skipped_when_single_column_is_unique():
    df = pd.DataFrame({
        "OrderID": list(range(1, 201)),
        "CustomerID": [i % 100 for i in range(200)],
        "Year": [2017 if i < 100 else 2018 for i in range(200)],
    })

    result = profile_dataframe(df)

    # OrderID is already unique: composites involving it would be
    # redundant and are not reported.
    for candidate in result["composite_uniqueness_candidates"]:
        assert "OrderID" not in candidate["columns"]


def test_profile_composite_skipped_when_member_is_constant():
    df = pd.DataFrame({
        "CustomerID": [i % 50 for i in range(200)],
        "Country": ["Canada"] * 200,
    })

    result = profile_dataframe(df)

    # A constant column cannot contribute discriminating power.
    assert result["composite_uniqueness_candidates"] == []


def test_profile_composite_candidates_capped():
    # Build a table with several composite candidates; the report must
    # stay bounded by PROFILING_MAX_COMBINATIONS.
    rows = 300
    df = pd.DataFrame({
        "A": [i % 150 for i in range(rows)],
        "B": [(i * 7) % 150 for i in range(rows)],
        "C": [i % 200 for i in range(rows)],
    })

    result = profile_dataframe(df)

    assert len(result["composite_uniqueness_candidates"]) <= PROFILING_MAX_COMBINATIONS


# ---------------------------------------------------------------------------
# Semantic-evidence pattern counters (audit point 9)
# ---------------------------------------------------------------------------


def test_profile_phone_postal_currency_patterns():
    df = pd.DataFrame({
        "phone": ["+1 (555) 123-4567", "555-123-4567"],
        "postal": ["K1A 0B1", "12345"],
        "currency": ["USD", "CAD"],
        "code": ["C001", "C002"],
    })

    result = profile_dataframe(df)

    columns = {
        column["column_name"]: column for column in result["columns"]
    }

    assert columns["phone"]["text"]["patterns"]["phone_like"] == 2
    assert columns["postal"]["text"]["patterns"]["postal_like"] == 2
    assert columns["currency"]["text"]["patterns"]["currency_like"] == 2
    # Alnum codes are not postal codes.
    assert columns["code"]["text"]["patterns"]["postal_like"] == 0


# ---------------------------------------------------------------------------
# Profiler stays lightweight and deterministic (audit point 10)
# ---------------------------------------------------------------------------


def test_profiler_contains_no_ml_dependencies():
    source = inspect.getsource(profiling_module).lower()

    forbidden = (
        "sklearn",
        "scikit",
        "xgboost",
        "lightgbm",
        "isolationforest",
        "kmeans",
        "dbscan",
        "torch",
        "transformers",
        "openai",
        "cluster",
    )

    for token in forbidden:
        assert token not in source, (
            f"profiler must stay free of ML/analytics deps; found {token!r}"
        )


def test_profile_identifier_name_matching_detects_common_identifier_names():
    df = pd.DataFrame({
        "product_code": [
            "P001",
            "P002",
            "P003",
            "P004",
        ]
    })

    result = profile_dataframe(df)

    column = result["columns"][0]

    assert column["identifier_name_signal"] is True

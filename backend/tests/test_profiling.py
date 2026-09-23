import pandas as pd

from app.services.profiling import profile_dataframe


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
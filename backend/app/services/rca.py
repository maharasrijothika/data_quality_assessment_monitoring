"""Stage 09b: Root Cause Analysis.

RCA starts from actual rule failures and reports observed concentrations
using correlation language only ("concentrated in", "observed alongside").
No causal claims are made.
"""

import pandas as pd
from sqlalchemy.orm import Session

from app.models import Rule, RuleExecution
from app.services.rule_execution import _load_table
from app.services.storage import RAW_DATA_DIR

MAX_PATTERNS = 5
MIN_GROUP_SIZE_FOR_CONCENTRATION = 2


def _value_repr(value) -> str:
    if value is None:
        return "(missing)"
    try:
        if pd.isna(value):
            return "(missing)"
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _failures_dataframe(
    dataframe: pd.DataFrame,
    failed_mask: pd.Series,
) -> pd.DataFrame:
    return dataframe.loc[failed_mask].copy()


def _categorical_concentration(
    failures: pd.DataFrame,
    column: str,
    total_failures: int,
) -> dict | None:
    """Observed concentration of failures across one categorical column."""
    if column not in failures.columns:
        return None

    series = failures[column].map(_value_repr)

    non_null = series[series != "(missing)"]

    if non_null.empty:
        return None

    counts = non_null.value_counts()
    top = counts.head(MAX_PATTERNS)

    return {
        "column": column,
        "distribution": [
            {
                "value": value,
                "count": int(count),
                "percentage": round(count / total_failures * 100, 1),
            }
            for value, count in top.items()
        ],
    }


def _whitespace_pattern_concentration(
    failures: pd.DataFrame,
    column: str,
    total_failures: int,
) -> dict | None:
    """Text-quality breakdown for failed values in one column."""
    if column not in failures.columns:
        return None

    series = failures[column].dropna()

    if series.empty:
        return None

    text = series.astype(str)

    categories = {
        "leading_or_trailing_whitespace": 0,
        "empty_string": 0,
        "other": 0,
    }

    for value in text:
        if value == "":
            categories["empty_string"] += 1
        elif value != value.strip():
            categories["leading_or_trailing_whitespace"] += 1
        else:
            categories["other"] += 1

    return {
        "column": column,
        "distribution": [
            {
                "value": name,
                "count": count,
                "percentage": round(count / total_failures * 100, 1),
            }
            for name, count in categories.items()
            if count > 0
        ],
    }


def _find_repeated_failure_values(
    failures: pd.DataFrame,
    rule_columns: list[str],
    total_failures: int,
) -> list[dict]:
    """Which distinct failing values repeat most (duplicate-evidence view)."""
    patterns = []

    for column in rule_columns:
        if column not in failures.columns:
            continue

        series = failures[column].map(_value_repr)
        counts = series.value_counts()

        for value, count in counts.head(3).items():
            if count >= MIN_GROUP_SIZE_FOR_CONCENTRATION:
                patterns.append(
                    {
                        "column": column,
                        "value": value,
                        "count": int(count),
                        "percentage": round(count / total_failures * 100, 1),
                    }
                )

    patterns.sort(key=lambda item: item["count"], reverse=True)

    return patterns[:MAX_PATTERNS]


def analyze_rule_failures(
    db: Session,
    rule: Rule,
    execution: RuleExecution,
    dataset_id: int,
    version_number: int,
    stored_filename: str,
) -> dict:
    """Analyze one rule execution's failures and produce evidence."""
    rule_json = rule.rule_json

    rule_columns = []
    if rule_json.get("column"):
        rule_columns.append(rule_json["column"])
    for column in rule_json.get("columns", []) or []:
        if column not in rule_columns:
            rule_columns.append(column)

    table_name = rule_json.get("table", rule.table_name)

    from app.services import storage as storage_module

    path = (
        storage_module.RAW_DATA_DIR
        / str(dataset_id)
        / f"v{version_number}"
        / stored_filename
    )

    if not path.exists():
        return {
            "status": "unavailable",
            "message": "Raw file not available for analysis.",
        }

    dataframe = _load_table(dataset_id, version_number, stored_filename)

    # Recompute the failure mask deterministically using the execution engine.
    from app.services.rule_execution import (
        RuleExecutionError,
        execute_rule_on_dataframe,
    )

    try:
        result = execute_rule_on_dataframe(dataframe, rule_json)
    except RuleExecutionError:
        return {
            "status": "unavailable",
            "message": "Rule could not be re-executed for analysis.",
        }

    rule_columns = [c for c in rule_columns if c in dataframe.columns]

    if result["failed_rows"] == 0 or not rule_columns:
        return {
            "status": "no_failures",
            "rule_name": rule.rule_name,
            "failed_rows": 0,
        }

    from app.services.rule_execution import (
        RuleExecutionError,
        get_failed_mask,
    )

    rule_type = rule_json.get("type") or rule_json.get("rule_template")

    try:
        failed_mask = get_failed_mask(dataframe, rule_json)
    except RuleExecutionError:
        return {
            "status": "unavailable",
            "message": "Unsupported rule template for analysis.",
        }

    failures = _failures_dataframe(dataframe, failed_mask)
    total_failures = result["failed_rows"]

    analysis: dict = {
        "status": "analyzed",
        "rule_name": rule.rule_name,
        "rule_type": rule_type,
        "failed_rows": total_failures,
        "failure_percentage": result["violation_rate"],
        "patterns": [],
        "concentration": [],
        "language_note": (
            "Percentages describe observed concentrations among failed rows "
            "only. They do not establish causality."
        ),
    }

    # Value-pattern concentration on the rule columns themselves.
    for column in rule_columns:
        whitespace = _whitespace_pattern_concentration(
            failures, column, total_failures
        )
        if whitespace is not None and whitespace["distribution"]:
            analysis["patterns"].append(whitespace)

    # Categorical concentration across other columns in the table.
    categorical_candidates = [
        column
        for column in dataframe.columns
        if column not in rule_columns
        and dataframe[column].dtype == object
        and dataframe[column].nunique(dropna=True) <= 50
    ]

    for column in categorical_candidates[:8]:
        concentration = _categorical_concentration(
            failures, column, total_failures
        )
        if concentration is not None and len(concentration["distribution"]) > 1:
            analysis["concentration"].append(concentration)

    analysis["repeated_failure_values"] = _find_repeated_failure_values(
        failures, rule_columns, total_failures
    )

    return analysis

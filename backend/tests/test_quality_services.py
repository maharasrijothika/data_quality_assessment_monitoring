"""Unit tests for the safe rule execution engine, scoring, RCA, remediation
and monitoring services."""

import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.models import Dataset, DatasetVersion, Rule, RuleExecution
from app.services.rule_execution import (
    RuleExecutionError,
    execute_rule_on_dataframe,
)


def make_dataframe():
    return pd.DataFrame(
        {
            "customer_id": ["C1", "C2", "C2", None, "C5"],
            "email": [
                "a@x.com",
                "bad-email",
                "c@x.com",
                "   ",
                "e@x.com",
            ],
        }
    )


# ---------------------------------------------------------------------------
# Execution engine
# ---------------------------------------------------------------------------


def test_completeness_counts_null_and_whitespace():
    result = execute_rule_on_dataframe(
        make_dataframe(),
        {
            "type": "completeness",
            "table": "customers",
            "column": "customer_id",
            "condition": "not_null",
            "treat_empty_string_as_null": True,
            "treat_whitespace_as_null": True,
        },
    )

    assert result["total_rows"] == 5
    assert result["applicable_rows"] == 5
    assert result["failed_rows"] == 1
    assert result["passed_rows"] == 4


def test_uniqueness_flags_all_duplicate_rows():
    result = execute_rule_on_dataframe(
        make_dataframe(),
        {
            "type": "uniqueness",
            "table": "customers",
            "columns": ["customer_id"],
            "ignore_nulls": True,
        },
    )

    # C1, C2 (x2 -> both flagged), C5 applicable; null not applicable.
    assert result["applicable_rows"] == 4
    assert result["failed_rows"] == 2


def test_validity_email_syntax():
    result = execute_rule_on_dataframe(
        make_dataframe(),
        {
            "type": "validity",
            "table": "customers",
            "column": "email",
            "check": "email_syntax",
        },
    )

    # bad-email fails; whitespace-only is missing, not a validity failure.
    assert result["failed_rows"] == 1
    assert result["failure_examples"][0]["email"] == "bad-email"


def test_unsupported_rule_type_rejected():
    with pytest.raises(RuleExecutionError):
        execute_rule_on_dataframe(
            make_dataframe(),
            {"type": "freeform_python", "code": "import os"},
        )


def test_missing_column_rejected():
    with pytest.raises(RuleExecutionError):
        execute_rule_on_dataframe(
            make_dataframe(),
            {
                "type": "completeness",
                "table": "customers",
                "column": "does_not_exist",
                "condition": "not_null",
            },
        )


def test_failure_examples_are_bounded():
    dataframe = pd.DataFrame(
        {
            "email": [f"bad{i}" for i in range(100)],
        }
    )

    result = execute_rule_on_dataframe(
        dataframe,
        {
            "type": "validity",
            "table": "customers",
            "column": "email",
            "check": "email_syntax",
        },
    )

    assert result["failed_rows"] == 100
    assert len(result["failure_examples"]) <= 20


def test_execution_requires_approved_rule():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(bind=engine)
    db = sessionmaker(bind=engine)()

    dataset = Dataset(dataset_name="D")
    db.add(dataset)
    db.flush()

    version = DatasetVersion(
        dataset_id=dataset.dataset_id,
        version_number=1,
        schema_fingerprint="s",
        content_fingerprint="c",
    )
    db.add(version)
    db.flush()

    rule = Rule(
        dataset_id=dataset.dataset_id,
        version_id=version.version_id,
        table_name="customers",
        rule_name="r",
        rule_json={"type": "completeness", "column": "x"},
        metric="completeness",
        status="recommended",
    )
    db.add(rule)
    db.flush()

    from app.services.rule_execution import execute_rule

    with pytest.raises(RuleExecutionError):
        execute_rule(
            db=db,
            rule=rule,
            dataset_id=dataset.dataset_id,
            version_id=version.version_id,
            version_number=1,
        )


# ---------------------------------------------------------------------------
# Validation engine
# ---------------------------------------------------------------------------


def test_validation_rejects_unknown_type():
    from app.services.rule_validation import validate_rule_dict

    engine = create_engine("sqlite://")
    Base.metadata.create_all(bind=engine)
    db = sessionmaker(bind=engine)()

    result = validate_rule_dict(db, 1, 1, {"type": "made_up"})

    assert result["status"] == "INVALID"


def test_validation_flags_missing_column():
    from app.services.rule_validation import validate_rule_dict
    from app.models import StoredProfile

    engine = create_engine("sqlite://")
    Base.metadata.create_all(bind=engine)
    db = sessionmaker(bind=engine)()

    db.add(
        StoredProfile(
            version_id=1,
            profile_json={
                "tables": [
                    {
                        "table_name": "customers",
                        "columns": [
                            {"column_name": "email", "row_count": 10}
                        ],
                    }
                ]
            },
        )
    )
    db.commit()

    result = validate_rule_dict(
        db,
        1,
        1,
        {
            "type": "completeness",
            "table": "customers",
            "column": "ghost_column",
            "condition": "not_null",
        },
    )

    assert result["status"] == "INVALID"
    assert any("not found" in issue["message"] for issue in result["issues"])


def test_validation_warns_on_low_distinct_uniqueness():
    from app.services.rule_validation import validate_rule_dict
    from app.models import StoredProfile

    engine = create_engine("sqlite://")
    Base.metadata.create_all(bind=engine)
    db = sessionmaker(bind=engine)()

    db.add(
        StoredProfile(
            version_id=1,
            profile_json={
                "tables": [
                    {
                        "table_name": "customers",
                        "columns": [
                            {
                                "column_name": "city",
                                "row_count": 10,
                                "distinct_percentage": 30.0,
                            }
                        ],
                    }
                ]
            },
        )
    )
    db.commit()

    result = validate_rule_dict(
        db,
        1,
        1,
        {
            "type": "uniqueness",
            "table": "customers",
            "columns": ["city"],
            "ignore_nulls": True,
        },
    )

    assert result["status"] == "NEEDS_REVIEW"


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _score_setup():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(bind=engine)
    db = sessionmaker(bind=engine)()

    dataset = Dataset(dataset_name="D")
    db.add(dataset)
    db.flush()

    version = DatasetVersion(
        dataset_id=dataset.dataset_id,
        version_number=1,
        schema_fingerprint="s",
        content_fingerprint="c",
    )
    db.add(version)
    db.flush()

    rules = []
    for metric in ("completeness", "validity"):
        rule = Rule(
            dataset_id=dataset.dataset_id,
            version_id=version.version_id,
            table_name="customers",
            rule_name=f"{metric} rule",
            rule_json={"type": metric, "column": "x"},
            metric=metric,
            status="approved",
        )
        db.add(rule)
        rules.append(rule)
    db.flush()

    # completeness: 100 applicable, 10 failed -> 90
    # validity: 50 applicable, 25 failed -> 50
    db.add(
        RuleExecution(
            rule_id=rules[0].rule_id,
            version_id=version.version_id,
            status="passed",
            total_rows=100,
            applicable_rows=100,
            passed_rows=90,
            failed_rows=10,
            pass_rate=90.0,
            violation_rate=10.0,
        )
    )
    db.add(
        RuleExecution(
            rule_id=rules[1].rule_id,
            version_id=version.version_id,
            status="passed",
            total_rows=50,
            applicable_rows=50,
            passed_rows=25,
            failed_rows=25,
            pass_rate=50.0,
            violation_rate=50.0,
        )
    )
    db.commit()

    return db, version.version_id


def test_scoring_aggregates_by_applicable_records():
    from app.services.scoring import compute_dq_score

    db, version_id = _score_setup()

    score = compute_dq_score(db, version_id)

    # Mean of 90 and 50.
    assert score.overall_score == 70.0
    assert score.metric_count == 2

    metrics = {m["metric"]: m for m in score.details_json["metrics"]}
    assert metrics["completeness"]["score"] == 90.0
    assert metrics["validity"]["score"] == 50.0


def test_scoring_exposes_critical_failures():
    from app.services.scoring import compute_dq_score

    db, version_id = _score_setup()

    score = compute_dq_score(db, version_id)

    critical = score.details_json["critical_failures"]

    # The 50% validity metric must be visible, not hidden by the average.
    assert any(item["metric"] == "validity" for item in critical)


def test_scoring_weighted_mode():
    from app.services.scoring import compute_dq_score

    db, version_id = _score_setup()

    score = compute_dq_score(
        db,
        version_id,
        weighted=True,
        weights={"completeness": 3.0, "validity": 1.0},
    )

    # (90*3 + 50*1) / 4 = 80
    assert score.overall_score == 80.0
    assert score.weighted == 1

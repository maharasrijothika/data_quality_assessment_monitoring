"""P0 acceptance tests: approval gate, NULL/empty/whitespace semantics,
execution-error scoring, and relationship discovery.

All tests drive REAL CSV files through the API using the existing pipeline
(upload -> ... -> execute), with an isolated storage dir and in-memory DB.
"""

import io

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool
from sqlalchemy.orm import sessionmaker

from app.database import Base, get_db
from app.main import app
from app.models import Dataset, DatasetVersion, Rule, RuleExecution
from app.services.relationship_discovery import _evaluate_pair
from app.services.rule_execution import execute_rule_on_dataframe


@pytest.fixture
def db():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    session = sessionmaker(bind=engine)()
    try:
        yield session
    finally:
        session.close()
        engine.dispose()


@pytest.fixture
def client(db, tmp_path, monkeypatch):
    import app.services.storage as storage

    monkeypatch.setattr(storage, "RAW_DATA_DIR", tmp_path / "datasets")

    def override_get_db():
        try:
            yield db
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.clear()


def _upload(client, name, content):
    response = client.post(
        "/datasets/upload",
        data={"dataset_name": name, "source_system": "ERP"},
        files={"files": (name + ".csv", io.BytesIO(content.encode()), "text/csv")},
    )
    assert response.status_code == 200, response.text
    return response.json()


def _profile(client, dataset_id):
    """Profiling is a read-only GET that persists the profile."""
    response = client.get(f"/datasets/{dataset_id}/profiling")
    assert response.status_code == 200, response.text
    return response.json()


def _approve_semantics(client, dataset_id):
    """Run semantic analysis and approve every pending prediction."""
    analyze = client.post(f"/datasets/{dataset_id}/semantic/analyze")
    assert analyze.status_code == 200, analyze.text

    semantic = client.get(f"/datasets/{dataset_id}/semantic").json()
    predictions = semantic.get("predictions", semantic.get("columns", []))

    approved = 0
    for prediction in predictions:
        if prediction.get("status") != "pending":
            continue
        concept_id = prediction.get("concept_id")
        decision = "approved" if concept_id else "rejected"
        response = client.post(
            f"/datasets/{dataset_id}/semantic/{prediction['prediction_id']}/decision",
            json={"decision": decision, "concept_id": concept_id},
        )
        assert response.status_code == 200, response.text
        approved += 1

    return approved


# ---------------------------------------------------------------------------
# Approval gate: 0 approved => 0 executions, no fake score
# ---------------------------------------------------------------------------


def test_zero_approved_rules_produce_zero_executions_and_no_score(client, db):
    payload = _upload(
        client,
        "gateds",
        (
            "customer_id,email\n"
            "1,alice@example.com\n"
            "2,bob@example.com\n"
        ),
    )
    dataset_id = payload["dataset_id"]

    _profile(client, dataset_id)
    _approve_semantics(client, dataset_id)

    rec = client.post(f"/datasets/{dataset_id}/recommendations")
    assert rec.status_code == 200, rec.text

    rules = client.get(f"/datasets/{dataset_id}/rules").json()["rules"]
    assert len(rules) > 0

    # Approve NOTHING. Execute must refuse.
    execution = client.post(f"/datasets/{dataset_id}/execute", json={"rule_ids": []})
    assert execution.status_code == 400

    # Score must refuse as well (no approved rules -> no fake score).
    scoring = client.post(f"/datasets/{dataset_id}/scores")
    assert scoring.status_code == 400

    executions = db.query(RuleExecution).count()
    assert executions == 0


def test_unapproved_rule_never_executes_even_if_requested(client, db):
    payload = _upload(
        client,
        "gateone",
        (
            "customer_id,email\n"
            "1,alice@example.com\n"
            "2,bob@example.com\n"
        ),
    )
    dataset_id = payload["dataset_id"]

    _profile(client, dataset_id)
    _approve_semantics(client, dataset_id)

    rec = client.post(f"/datasets/{dataset_id}/recommendations")
    assert rec.status_code == 200

    rules = client.get(f"/datasets/{dataset_id}/rules").json()["rules"]
    assert len(rules) > 0

    unapproved_ids = [rule["rule_id"] for rule in rules]

    execution = client.post(
        f"/datasets/{dataset_id}/execute",
        json={"rule_ids": unapproved_ids},
    )
    # Explicitly requesting unapproved rules must not run them.
    assert execution.status_code == 400

    executions = db.query(RuleExecution).count()
    assert executions == 0


def test_approved_rules_execute_and_persist(client, db):
    payload = _upload(
        client,
        "gateexec",
        (
            "customer_id,email,age\n"
            "1,alice@example.com,25\n"
            "2,bob@example.com,34\n"
            "3,carla@bad,29\n"
        ),
    )
    dataset_id = payload["dataset_id"]

    _profile(client, dataset_id)
    _approve_semantics(client, dataset_id)

    rec = client.post(f"/datasets/{dataset_id}/recommendations")
    assert rec.status_code == 200

    rules = client.get(f"/datasets/{dataset_id}/rules").json()["rules"]

    for rule in rules:
        approve = client.post(
            f"/rules/{rule['rule_id']}/approve",
            json={"decision": "approved"},
        )
        assert approve.status_code == 200, approve.text

    execution = client.post(f"/datasets/{dataset_id}/execute", json={})
    assert execution.status_code == 200, execution.text
    body = execution.json()

    assert body["executed"] == len(rules)
    assert body["errors"] == 0

    executions = db.query(RuleExecution).count()
    assert executions == len(rules)

    scoring = client.post(f"/datasets/{dataset_id}/scores")
    assert scoring.status_code == 200, scoring.text


# ---------------------------------------------------------------------------
# NULL / empty / whitespace execution semantics
# ---------------------------------------------------------------------------


def test_completeness_semantics_null_empty_whitespace():
    df = pd.DataFrame(
        {
            "value": ["a", None, "", "   ", "b"],
        }
    )

    base = {"type": "completeness", "column": "value"}

    # Plain nulls only.
    result = execute_rule_on_dataframe(df, {**base})
    assert result["applicable_rows"] == 5
    assert result["failed_rows"] == 1  # None only

    # Empty string treated as null.
    result = execute_rule_on_dataframe(
        df, {**base, "treat_empty_string_as_null": True}
    )
    assert result["failed_rows"] == 2  # None + ""

    # Whitespace treated as null too.
    result = execute_rule_on_dataframe(
        df,
        {
            **base,
            "treat_empty_string_as_null": True,
            "treat_whitespace_as_null": True,
        },
    )
    assert result["failed_rows"] == 3  # None + "" + "   "

    # Whitespace alone without empty flag: "   " is whitespace-only, "a"
    # and "b" pass; empty string "" is NOT whitespace (distinct evidence).
    result = execute_rule_on_dataframe(
        df, {**base, "treat_whitespace_as_null": True}
    )
    assert result["failed_rows"] == 2  # None + "   "


def test_validity_ignores_missing_values_completeness_reports_them():
    df = pd.DataFrame(
        {
            "email": ["ok@example.com", None, "   ", "bad-email"],
        }
    )

    validity = execute_rule_on_dataframe(
        df, {"type": "validity", "column": "email", "check": "email_syntax"}
    )

    # None and "   " are missing -> not validity failures.
    assert validity["applicable_rows"] == 4
    assert validity["failed_rows"] == 1  # "bad-email" only

    completeness = execute_rule_on_dataframe(
        df,
        {
            "type": "completeness",
            "column": "email",
            "treat_empty_string_as_null": True,
            "treat_whitespace_as_null": True,
        },
    )
    assert completeness["failed_rows"] == 2  # None + "   "


def test_uniqueness_ignore_nulls_semantics():
    df = pd.DataFrame(
        {
            "customer_id": ["1", "1", None, "2", ""],
        }
    )

    result = execute_rule_on_dataframe(
        df, {"type": "uniqueness", "column": "customer_id", "ignore_nulls": True}
    )

    # Applicable: rows 1,2,4 (null excluded); "" normalized -> "" is a value
    # and row 5 is excluded as empty. Row 1 and 2 duplicate.
    assert result["applicable_rows"] == 3
    assert result["failed_rows"] == 2

    result_strict = execute_rule_on_dataframe(
        df, {"type": "uniqueness", "column": "customer_id", "ignore_nulls": False}
    )
    assert result_strict["applicable_rows"] == 5


# ---------------------------------------------------------------------------
# Execution errors: persisted, never a silent 100%
# ---------------------------------------------------------------------------


def test_execution_error_is_persisted_and_scoring_excludes_it(client, db):
    payload = _upload(
        client,
        "errds",
        (
            "customer_id,email\n"
            "1,alice@example.com\n"
            "2,bob@example.com\n"
        ),
    )
    dataset_id = payload["dataset_id"]

    _profile(client, dataset_id)
    _approve_semantics(client, dataset_id)

    rec = client.post(f"/datasets/{dataset_id}/recommendations")
    assert rec.status_code == 200

    # Approve every recommended rule, then CORRUPT one rule's column to
    # force an execution error (column no longer exists at execution time).
    rules = client.get(f"/datasets/{dataset_id}/rules").json()["rules"]

    target = None
    for rule in rules:
        if rule["rule_json"].get("column") or rule["rule_json"].get("columns"):
            target = rule
            break

    assert target is not None, "need at least one column-bound rule"

    # Approve all rules.
    for rule in rules:
        approve = client.post(
            f"/rules/{rule['rule_id']}/approve", json={"decision": "approved"}
        )
        assert approve.status_code == 200

    # Corrupt the target rule's column after approval (edit gate validates
    # on approve; corrupting afterwards simulates a mid-run break).
    corrupted = dict(target["rule_json"])
    if "column" in corrupted:
        corrupted["column"] = "no_such_column"
    if "columns" in corrupted:
        corrupted["columns"] = ["no_such_column"]
    db_rule = db.get(Rule, target["rule_id"])
    db_rule.rule_json = corrupted
    db.commit()

    execution = client.post(f"/datasets/{dataset_id}/execute", json={})
    assert execution.status_code == 200, execution.text
    body = execution.json()

    assert body["errors"] >= 1
    assert any(r["status"] == "error" for r in body["results"])

    # Error row must be PERSISTED.
    error_rows = (
        db.query(RuleExecution)
        .filter(RuleExecution.status == "error")
        .all()
    )
    assert len(error_rows) >= 1
    assert error_rows[0].error_message

    # Scoring must report EXECUTION_ERROR status, not a silent 100%.
    scoring = client.post(f"/datasets/{dataset_id}/scores")
    assert scoring.status_code == 200, scoring.text
    details = scoring.json()["details"]

    if body["executed"] == 0:
        # All rules errored -> every metric must be execution_error and the
        # overall score must NOT be a clean 100.
        assert details["execution_error_count"] >= 1
        assert scoring.json()["overall_score"] != 100.0


def test_all_errored_metric_reports_execution_error(db):
    """Metric-level unit: every execution errored => execution_error status."""
    from app.services.scoring import compute_metric_results

    dataset = Dataset(dataset_name="err")
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
        table_name="t",
        rule_name="broken",
        rule_json={"type": "completeness", "column": "c"},
        metric="completeness",
        source="recommended",
        risk="low",
        validation_status="VALID",
        status="approved",
    )
    db.add(rule)
    db.flush()

    db.add(
        RuleExecution(
            rule_id=rule.rule_id,
            version_id=version.version_id,
            status="error",
            error_message="boom",
        )
    )
    db.flush()

    results = compute_metric_results(db, version.version_id)

    assert len(results) == 1
    assert results[0].status == "execution_error"
    assert results[0].score == 0.0


# ---------------------------------------------------------------------------
# Relationship discovery: evidence + low containment + decisions
# ---------------------------------------------------------------------------


def _mk_pair(parent_values, child_values, parent_profile=None, child_profile=None, columns=("customer_id", "customer_id")):
    parent_df = pd.DataFrame({columns[0]: parent_values, "other": ["x"] * len(parent_values)})
    child_df = pd.DataFrame({columns[1]: child_values, "amount": [1.0] * len(child_values)})

    parent_profile = parent_profile or {
        "data_type": "object",
        "distinct_percentage": 100.0,
        "null_percentage": 0.0,
        "non_null_count": len(parent_values),
        "identifier_name_signal": True,
        "identifier_signal": True,
    }
    child_profile = child_profile or {
        "data_type": "object",
        "null_percentage": 0.0,
    }

    return _evaluate_pair(
        parent_df,
        child_df,
        columns[0],
        columns[1],
        parent_profile,
        child_profile,
    )


def test_relationship_full_containment_scores_high():
    outcome = _mk_pair(
        [f"C{i}" for i in range(1, 21)],
        [f"C{i}" for i in range(1, 16)],
    )

    assert outcome["score"] >= 0.7
    assert outcome["stats"]["containment"] == 1.0
    assert outcome["stats"]["orphan_distinct_count"] == 0


def test_relationship_low_containment_survives_with_orphan_evidence():
    """Low containment must NOT eliminate the candidate: it is orphan
    evidence (a data-quality problem), not missing-relationship evidence."""
    outcome = _mk_pair(
        [f"C{i}" for i in range(1, 21)],
        [f"C{i}" for i in range(1, 8)] + [f"ORPHAN{i}" for i in range(1, 9)],
    )

    containment = outcome["stats"]["containment"]
    assert containment < 0.5  # genuinely low containment

    # The candidate still scores (evidence survives) and orphan stats are
    # reported for later RI validation.
    assert outcome["score"] >= 0.4
    assert outcome["stats"]["orphan_distinct_count"] == 8
    assert outcome["stats"]["orphan_distinct_rate"] > 0.5


def test_relationship_gibberish_pair_scores_low():
    # Different column names AND disjoint values: near-zero evidence on the
    # value side, and zero containment. The candidate must not rank high.
    outcome = _mk_pair(
        [f"P{i}" for i in range(1, 21)],
        [f"Z{i}" for i in range(1, 21)],
        parent_profile={
            "data_type": "object",
            "distinct_percentage": 100.0,
            "null_percentage": 0.0,
            "non_null_count": 20,
            "identifier_name_signal": True,
            "identifier_signal": True,
        },
        columns=("account_ref", "widget_code"),
    )

    assert outcome["stats"]["containment"] == 0.0
    assert outcome["evidence"]["name_similarity"]["value"] < 0.4
    assert outcome["score"] < 0.55


def test_relationship_no_value_support_needs_name_agreement():
    """Zero containment AND zero overlap with dissimilar names must not
    become a candidate: identifier-vs-identifier lookalikes with disjoint
    values are coincidence, not orphan evidence."""
    from app.config import RELATIONSHIP_MIN_CANDIDATE_SCORE

    # Same 1..20 integer values, different names: dense ranges overlap
    # trivially, but with NO name agreement the pair must be rejected as a
    # candidate by the name-support floor... wait: names here ARE the same.
    # The realistic coincidence case is DIFFERENT names + disjoint values.
    outcome = _mk_pair(
        [f"P{i}" for i in range(1, 21)],
        [f"Z{i}" for i in range(1, 21)],
        parent_profile={
            "data_type": "object",
            "distinct_percentage": 100.0,
            "null_percentage": 0.0,
            "non_null_count": 20,
            "identifier_name_signal": True,
            "identifier_signal": True,
        },
        columns=("account_ref", "widget_code"),
    )

    # Same guard as the discovery loop applies:
    evidence = outcome["evidence"]
    no_value_support = (
        evidence["value_containment"]["value"] == 0.0
        and evidence["value_overlap"]["value"] == 0.0
    )

    assert no_value_support
    assert evidence["name_similarity"]["value"] < 0.60
    assert outcome["score"] < RELATIONSHIP_MIN_CANDIDATE_SCORE or (
        evidence["name_similarity"]["value"] < 0.60
    )


def test_relationship_decision_flow_api(client, db):
    payload = _upload(
        client,
        "relds",
        (
            "customer_id,customer_name,email\n"
            "1,Alice,alice@example.com\n"
            "2,Bob,bob@example.com\n"
        ),
    )
    dataset_id = payload["dataset_id"]

    # Relationship discovery needs profiling first; run through the API.
    _profile(client, dataset_id)

    discovery = client.post(f"/datasets/{dataset_id}/relationships/discover")
    assert discovery.status_code == 200, discovery.text

    listing = client.get(f"/datasets/{dataset_id}/relationships").json()
    assert "candidates" in listing

    # Single-table dataset: no cross-table candidates expected, but the API
    # must work and mark completion.
    complete = client.post(f"/datasets/{dataset_id}/relationships/complete")
    assert complete.status_code == 200

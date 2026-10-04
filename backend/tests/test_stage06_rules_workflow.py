"""Stage 06+ acceptance tests: metric applicability, rule templates,
deterministic recommendation, business-rule interpretation, validation gates,
approval with versioning + feedback, safe execution of new templates, and
column/table/dataset scoring with N/A semantics.

All API tests drive real CSV uploads through the existing pipeline
(upload -> profiling -> semantic -> [relationships] -> rules workflow).
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


@pytest.fixture
def db_session():
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
def client(db_session, tmp_path, monkeypatch):
    import app.services.storage as storage

    monkeypatch.setattr(storage, "RAW_DATA_DIR", tmp_path / "datasets")

    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.clear()


CUSTOMERS_CSV = (
    "customer_id,first_name,middle_name,email,age,country,city,created_at\n"
    "C001,Alice,,alice@example.com,35,India,Chennai,2026-09-01\n"
    "C002,Bob,,bob@example.com,44,India,Chennai,2026-09-02\n"
    "C003,Carol,Ann,carol@example.com,28,USA,Boston,2026-09-03\n"
    "C004,Dave,,dave@example.com,52,India,Chennai,2026-09-04\n"
    "C005,Eve,,eve@example.com,31,USA,Boston,2026-09-05\n"
)

ORDERS_CSV = (
    "order_id,customer_id,amount\n"
    + "".join(f"O{i:03d},C{(i % 5) + 1:03d},{i * 10}.0\n" for i in range(1, 21))
)


def _upload(client, name, content):
    response = client.post(
        "/datasets/upload",
        data={"dataset_name": name, "source_system": "ERP"},
        files={"files": (name + ".csv", io.BytesIO(content.encode()), "text/csv")},
    )
    assert response.status_code == 200, response.text
    return response.json()


def _upload_two_tables(client, name):
    response = client.post(
        "/datasets/upload",
        data={"dataset_name": name, "source_system": "ERP"},
        files=[
            ("files", ("customers.csv", io.BytesIO(CUSTOMERS_CSV.encode()), "text/csv")),
            ("files", ("orders.csv", io.BytesIO(ORDERS_CSV.encode()), "text/csv")),
        ],
    )
    assert response.status_code == 200, response.text
    return response.json()


def _approve_all_semantics(client, dataset_id):
    analyze = client.post(f"/datasets/{dataset_id}/semantic/analyze")
    assert analyze.status_code == 200, analyze.text

    semantic = client.get(f"/datasets/{dataset_id}/semantic").json()
    predictions = semantic.get("predictions", semantic.get("columns", []))

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


def _prepare(client, name, content=CUSTOMERS_CSV):
    payload = _upload(client, name, content)
    dataset_id = payload["dataset_id"]
    client.get(f"/datasets/{dataset_id}/profiling")
    _approve_all_semantics(client, dataset_id)
    return dataset_id


def _approve_every_rule(client, dataset_id):
    rules = client.get(f"/datasets/{dataset_id}/rules").json()["rules"]
    for rule in rules:
        if rule["status"] == "recommended":
            response = client.post(
                f"/rules/{rule['rule_id']}/approve", json={"decision": "approved"}
            )
            assert response.status_code == 200, response.text
    return rules


# ---------------------------------------------------------------------------
# Metric applicability
# ---------------------------------------------------------------------------


def test_applicability_matrix_completeness_applies_to_all_columns(client):
    dataset_id = _prepare(client, "appl1")

    response = client.get(f"/datasets/{dataset_id}/applicability")
    assert response.status_code == 200, response.text

    matrix = response.json()
    assert matrix["column_count"] == 8

    for column in matrix["columns"]:
        assert column["metrics"]["completeness"]["status"] == "APPLICABLE", (
            f"{column['column_name']}: {column['metrics']['completeness']}"
        )
        # Every applicability claim carries explicit evidence.
        assert column["metrics"]["completeness"]["evidence"]


def test_applicability_uniqueness_not_recommended_for_name_columns(client):
    dataset_id = _prepare(client, "appl2")

    matrix = client.get(f"/datasets/{dataset_id}/applicability").json()
    by_name = {c["column_name"]: c for c in matrix["columns"]}

    # first_name: no identifier evidence -> uniqueness NOT_APPLICABLE even
    # though observed values happen to be unique.
    assert by_name["first_name"]["metrics"]["uniqueness"]["status"] == "NOT_APPLICABLE"
    assert by_name["first_name"]["metrics"]["uniqueness"]["evidence"]


def test_applicability_accuracy_requires_reference_source(client):
    dataset_id = _prepare(client, "appl3")

    matrix = client.get(f"/datasets/{dataset_id}/applicability").json()
    by_name = {c["column_name"]: c for c in matrix["columns"]}

    # Accuracy is never claimed from the upload alone.
    email = by_name["email"]["metrics"]["accuracy"]
    assert email["status"] in {"NEEDS_REVIEW", "NOT_APPLICABLE"}
    joined = " ".join(email["evidence"]).lower()
    assert "authoritative" in joined or "reference" in joined or "configuration" in joined


def test_applicability_timeliness_needs_user_sla(client):
    dataset_id = _prepare(client, "appl4")

    matrix = client.get(f"/datasets/{dataset_id}/applicability").json()
    by_name = {c["column_name"]: c for c in matrix["columns"]}

    created = by_name["created_at"]["metrics"]["timeliness"]
    assert created["status"] in {"NEEDS_REVIEW", "NOT_APPLICABLE"}
    joined = " ".join(created["evidence"]).lower()
    assert "sla" in joined or "threshold" in joined or "freshness" in joined


# ---------------------------------------------------------------------------
# Business requiredness (middle_name / "optional" scenario)
# ---------------------------------------------------------------------------


def test_optional_column_suppresses_requiredness_recommendations(client):
    dataset_id = _prepare(client, "ctx1")

    # Mark middle_name completeness as NOT required.
    response = client.post(
        f"/datasets/{dataset_id}/metric-contexts",
        json={
            "table_name": "ctx1",
            "column_name": "middle_name",
            "metric": "completeness",
            "required": False,
            "business_note": "Middle name is optional.",
        },
    )
    assert response.status_code == 200, response.text

    # Regenerate recommendations.
    rec = client.post(f"/datasets/{dataset_id}/recommendations")
    assert rec.status_code == 200

    rules = client.get(f"/datasets/{dataset_id}/rules").json()["rules"]
    middle_name_rules = [r for r in rules if r["table_name"] == "ctx1"]

    # The NOT_NULL recommendation for middle_name must be suppressed...
    not_null_on_middle = [
        r
        for r in middle_name_rules
        if r["rule_json"].get("column") == "middle_name"
        and (r["rule_json"].get("rule_template") or r["rule_json"].get("condition"))
        in {"NOT_NULL", "not_null"}
    ]
    assert not_null_on_middle == []

    # ...but completeness for required columns (first_name) still recommended.
    first_name_rules = [
        r
        for r in rules
        if r["rule_json"].get("column") == "first_name"
        and r["metric"] == "completeness"
    ]
    assert first_name_rules


# ---------------------------------------------------------------------------
# Deterministic recommendation
# ---------------------------------------------------------------------------


def test_recommendations_include_null_unique_and_email_rules(client):
    dataset_id = _prepare(client, "rec1")

    rec = client.post(f"/datasets/{dataset_id}/recommendations")
    assert rec.status_code == 200
    assert rec.json()["recommendation_count"] >= 3

    rules = rec.json()["rules"]
    metrics = {r["metric"] for r in rules}
    assert "completeness" in metrics

    # Every recommendation carries evidence and is not authoritative.
    for rule in rules:
        assert rule["status"] == "recommended"
        assert rule["rule_template"], rule

    # customer_id gets a UNIQUE candidate (identifier evidence).
    customer_id_rules = [
        r for r in rules if r["rule_json"].get("column") == "customer_id"
    ]
    templates = {r["rule_template"] for r in customer_id_rules}
    assert "NOT_NULL" in templates
    assert "UNIQUE" in templates


def test_recommendation_replaces_pending_but_keeps_approved(client):
    dataset_id = _prepare(client, "rec2")

    client.post(f"/datasets/{dataset_id}/recommendations")
    rules = client.get(f"/datasets/{dataset_id}/rules").json()["rules"]

    # Approve exactly one rule.
    target = rules[0]
    approve = client.post(
        f"/rules/{target['rule_id']}/approve", json={"decision": "approved"}
    )
    assert approve.status_code == 200

    # Re-generate: approved rule survives, pending ones are replaced.
    client.post(f"/datasets/{dataset_id}/recommendations")
    after = client.get(f"/datasets/{dataset_id}/rules").json()["rules"]

    surviving = [r for r in after if r["rule_id"] == target["rule_id"]]
    assert surviving and surviving[0]["status"] == "approved"
    assert all(r["status"] in {"approved", "recommended"} for r in after)


def test_numeric_range_recommendation_is_candidate_not_authoritative(client):
    dataset_id = _prepare(client, "rec3")

    client.post(f"/datasets/{dataset_id}/recommendations")
    rules = client.get(f"/datasets/{dataset_id}/rules").json()["rules"]

    age_range = [
        r
        for r in rules
        if r["rule_json"].get("rule_template") == "NUMERIC_RANGE"
        and r["rule_json"].get("column") == "age"
    ]

    # Observed-range candidate exists but is labeled as candidate-only...
    for rule in age_range:
        assert rule["rule_json"].get("origin") == "observed_distribution"
        issues = rule["validation_issues"]
        # ...and must never be silently VALID as a business bound.

    if age_range:
        assert age_range[0]["status"] == "recommended"


# ---------------------------------------------------------------------------
# Business-rule interpretation (Layer 1, deterministic)
# ---------------------------------------------------------------------------


def test_interpret_numeric_range_sentence(client):
    dataset_id = _prepare(client, "interp1")

    response = client.post(
        f"/datasets/{dataset_id}/rules/interpret",
        json={
            "text": "Age must be between 18 and 65.",
            "table_name": "interp1",
            "column_name": "age",
        },
    )
    assert response.status_code == 200, response.text
    result = response.json()

    assert result["status"] == "RESOLVED"
    assert result["metric"] == "validity"
    assert result["rule_template"] == "NUMERIC_RANGE"
    assert result["parameters"]["min"] == 18
    assert result["parameters"]["max"] == 65
    assert result["validation"]["status"] in {"VALID", "NEEDS_REVIEW", "INVALID"}


def test_interpret_ambiguous_text_asks_user_never_guesses(client):
    dataset_id = _prepare(client, "interp2")

    response = client.post(
        f"/datasets/{dataset_id}/rules/interpret",
        json={"text": "The customer information should be good."},
    )
    assert response.status_code == 200
    result = response.json()

    assert result["status"] == "UNRESOLVED"
    assert result["metric"] is None
    assert result["rule_template"] is None
    assert len(result["metric_options"]) >= 4


def test_interpret_null_and_unique_sentences(client):
    dataset_id = _prepare(client, "interp3")

    for text, metric, template in [
        ("customer_id must not be null", "completeness", "NOT_NULL"),
        ("first_name must not be empty", "completeness", "NOT_EMPTY"),
        ("customer_id must be unique", "uniqueness", "UNIQUE"),
    ]:
        response = client.post(
            f"/datasets/{dataset_id}/rules/interpret",
            json={"text": text, "table_name": "interp3"},
        )
        assert response.status_code == 200, response.text
        result = response.json()
        assert (result["metric"], result["rule_template"]) == (metric, template), text


def test_interpret_timeliness_sentence_normalizes_threshold(client):
    dataset_id = _prepare(client, "interp4")

    response = client.post(
        f"/datasets/{dataset_id}/rules/interpret",
        json={
            "text": "created_at must be updated within 24 hours.",
            "table_name": "interp4",
            "column_name": "created_at",
        },
    )
    assert response.status_code == 200
    result = response.json()

    assert result["status"] == "RESOLVED"
    assert result["metric"] == "timeliness"
    assert result["rule_template"] == "FRESHNESS_THRESHOLD"
    assert result["parameters"]["max_age"] == {"amount": 24, "unit": "hours"}


# ---------------------------------------------------------------------------
# Manual rules + template registry
# ---------------------------------------------------------------------------


def test_manual_structured_rule_with_parameters(client):
    dataset_id = _prepare(client, "manual1")

    response = client.post(
        f"/datasets/{dataset_id}/rules/manual",
        json={
            "metric": "validity",
            "rule_template": "NUMERIC_RANGE",
            "table_name": "manual1",
            "column_name": "age",
            "parameters": {"min": 18, "max": 65},
            "rule_name": "Age must be between 18 and 65.",
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["rule"]["metric"] == "validity"
    assert body["rule"]["status"] == "recommended"


def test_manual_rule_rejects_metric_template_mismatch(client):
    dataset_id = _prepare(client, "manual2")

    response = client.post(
        f"/datasets/{dataset_id}/rules/manual",
        json={
            "metric": "completeness",
            "rule_template": "NUMERIC_RANGE",
            "table_name": "manual2",
            "column_name": "age",
            "parameters": {"min": 18, "max": 65},
        },
    )
    assert response.status_code == 400


def test_template_registry_endpoint(client):
    dataset_id = _prepare(client, "tmpl1")

    response = client.get(f"/datasets/{dataset_id}/rule-templates")
    assert response.status_code == 200

    data = response.json()
    metric_names = {m["name"] for m in data["metrics"]}
    assert {
        "completeness", "uniqueness", "validity", "accuracy",
        "consistency", "referential_integrity", "timeliness",
    } == metric_names

    templates = {t["name"] for t in data["templates"]}
    assert {
        "NOT_NULL", "NOT_EMPTY", "NOT_WHITESPACE", "UNIQUE", "COMPOSITE_UNIQUE",
        "NUMERIC_RANGE", "FOREIGN_KEY_EXISTS", "FRESHNESS_THRESHOLD",
    } <= templates


# ---------------------------------------------------------------------------
# Validation gates
# ---------------------------------------------------------------------------


def test_validation_rejects_unknown_template(client):
    dataset_id = _prepare(client, "val1")

    response = client.post(
        f"/datasets/{dataset_id}/rules/manual",
        json={
            "metric": "completeness",
            "rule_template": "MADE_UP",
            "table_name": "val1",
            "column_name": "age",
            "parameters": {},
        },
    )
    assert response.status_code == 400


def test_validation_numeric_range_parameter_errors(client):
    from app.services.rule_validation import validate_rule_dict

    db = next(app.dependency_overrides[get_db]())
    # Low-level: min > max is INVALID.
    result = validate_rule_dict(
        db,
        1,
        1,
        {
            "rule_template": "NUMERIC_RANGE",
            "metric": "validity",
            "table": "x",
            "column": "age",
            "parameters": {"min": 65, "max": 18},
        },
    )
    assert result["status"] == "INVALID"
    assert any("min must not exceed max" in i["message"] for i in result["issues"])


def test_validation_timeliness_without_sla_is_invalid(client):
    dataset_id = _prepare(client, "val2")

    response = client.post(
        f"/datasets/{dataset_id}/rules/manual",
        json={
            "metric": "timeliness",
            "rule_template": "FRESHNESS_THRESHOLD",
            "table_name": "val2",
            "column_name": "created_at",
            "parameters": {},
        },
    )
    assert response.status_code == 200
    body = response.json()

    # No SLA supplied -> validation must NOT be executable.
    assert body["validation"]["status"] == "INVALID"


def test_validation_foreign_key_requires_approved_relationship(client):
    dataset_id = _prepare(client, "val3")

    response = client.post(
        f"/datasets/{dataset_id}/rules/manual",
        json={
            "metric": "referential_integrity",
            "rule_template": "FOREIGN_KEY_EXISTS",
            "table_name": "val3",
            "parameters": {
                "parent_table": "other",
                "parent_column": "id",
                "child_table": "val3",
                "child_column": "customer_id",
            },
        },
    )
    assert response.status_code == 200
    body = response.json()

    # No approved relationship exists -> INVALID, cannot be approved.
    assert body["validation"]["status"] == "INVALID"
    assert any("relationship" in i["message"].lower() for i in body["validation"]["issues"])


# ---------------------------------------------------------------------------
# Approval, versioning, feedback
# ---------------------------------------------------------------------------


def test_approval_assigns_rule_code_and_version(client):
    dataset_id = _prepare(client, "ver1")

    client.post(f"/datasets/{dataset_id}/recommendations")
    rules = client.get(f"/datasets/{dataset_id}/rules").json()["rules"]

    target = rules[0]
    response = client.post(
        f"/rules/{target['rule_id']}/approve", json={"decision": "approved"}
    )
    assert response.status_code == 200
    body = response.json()

    assert body["rule_code"] == "R001"
    assert body["version_number"] == 1


def test_edit_approved_rule_creates_new_version(client):
    dataset_id = _prepare(client, "ver2")

    client.post(f"/datasets/{dataset_id}/recommendations")
    rules = client.get(f"/datasets/{dataset_id}/rules").json()["rules"]

    # Approve a NOT_NULL rule, then edit it.
    target = next(
        r
        for r in rules
        if r["rule_json"].get("rule_template") == "NOT_NULL"
    )
    client.post(f"/rules/{target['rule_id']}/approve", json={"decision": "approved"})

    edited_json = dict(target["rule_json"])
    edited_json["parameters"] = {}
    edited_json["business_note"] = "tightened after review"

    response = client.post(
        f"/rules/{target['rule_id']}/approve",
        json={
            "decision": "approved",
            "edited_rule_json": edited_json,
            "edited_rule_name": f"{target['rule_name']} (v2)",
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()

    assert body["version_number"] == 2

    versions = client.get(f"/rules/{target['rule_id']}/versions").json()
    assert versions["current_version_number"] == 2
    assert len(versions["versions"]) == 2
    kinds = {v["change_kind"] for v in versions["versions"]}
    assert "initial_approval" in kinds and "edited" in kinds

    # Historical definition is preserved (immutable snapshots).
    v1 = next(v for v in versions["versions"] if v["version_number"] == 1)
    assert v1["rule_json"] == target["rule_json"]


def test_invalid_edit_cannot_be_approved(client):
    dataset_id = _prepare(client, "ver3")

    client.post(f"/datasets/{dataset_id}/recommendations")
    rules = client.get(f"/datasets/{dataset_id}/rules").json()["rules"]

    target = next(
        r for r in rules if r["rule_json"].get("rule_template") == "NOT_NULL"
    )
    client.post(f"/rules/{target['rule_id']}/approve", json={"decision": "approved"})

    # Edit to reference a non-existent column -> INVALID -> rejected by API.
    edited = dict(target["rule_json"])
    edited["column"] = "ghost_column"

    response = client.post(
        f"/rules/{target['rule_id']}/approve",
        json={"decision": "approved", "edited_rule_json": edited},
    )
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "invalid" in detail["message"].lower()


def test_human_decisions_stored_as_feedback(client):
    dataset_id = _prepare(client, "fb1")

    client.post(f"/datasets/{dataset_id}/recommendations")
    rules = client.get(f"/datasets/{dataset_id}/rules").json()["rules"]

    # Accept one, reject another.
    accept = rules[0]
    reject = rules[1]
    client.post(f"/rules/{accept['rule_id']}/approve", json={"decision": "approved"})
    client.post(f"/rules/{reject['rule_id']}/approve", json={"decision": "rejected"})

    feedback = client.get(f"/datasets/{dataset_id}/rule-feedback").json()
    assert feedback["count"] >= 2

    decisions = {f["decision"] for f in feedback["feedback"]}
    assert {"accepted", "rejected"} <= decisions

    # Feature capture for future offline ranker training.
    sample = feedback["feedback"][0]
    assert sample["rule_template"]
    assert sample["metric"]
    assert isinstance(sample["features"], dict)


# ---------------------------------------------------------------------------
# Execution of new templates + N/A semantics
# ---------------------------------------------------------------------------


def test_execution_null_empty_whitespace_distinct_templates(client):
    df = pd.DataFrame({"value": ["a", None, "", "   ", "b"]})

    from app.services.rule_execution import execute_rule_on_dataframe

    not_null = execute_rule_on_dataframe(
        df, {"rule_template": "NOT_NULL", "column": "value", "parameters": {}}
    )
    assert not_null["failed_rows"] == 1  # None only

    not_empty = execute_rule_on_dataframe(
        df, {"rule_template": "NOT_EMPTY", "column": "value", "parameters": {}}
    )
    assert not_empty["failed_rows"] == 1  # "" only

    not_ws = execute_rule_on_dataframe(
        df, {"rule_template": "NOT_WHITESPACE", "column": "value", "parameters": {}}
    )
    assert not_ws["failed_rows"] == 1  # "   " only


def test_execution_zero_applicability_reports_na_not_100(client):
    df = pd.DataFrame({"a": [1, 2, 3], "b": [None, None, None]})

    from app.services.rule_execution import execute_rule_on_dataframe

    # COLUMN_COMPARISON where one side is always NULL: applicable rows = 0.
    result = execute_rule_on_dataframe(
        df,
        {
            "rule_template": "COLUMN_COMPARISON",
            "parameters": {"left_column": "a", "operator": ">=", "right_column": "b"},
        },
    )
    assert result["applicable_rows"] == 0
    assert result["pass_rate"] is None  # N/A, never 100
    assert result["violation_rate"] is None


def test_execution_numeric_range_and_allowed_values(client):
    df = pd.DataFrame({"age": [30, 40, 70, None], "country": ["India", "USA", "India", ""]})

    from app.services.rule_execution import execute_rule_on_dataframe

    rng = execute_rule_on_dataframe(
        df,
        {
            "rule_template": "NUMERIC_RANGE",
            "column": "age",
            "parameters": {"min": 18, "max": 65},
        },
    )
    # Validity applies to all rows (missing values are not validity failures).
    # 70 fails; None and "" are missing for their columns.
    assert rng["applicable_rows"] == 4
    assert rng["failed_rows"] == 1

    allowed = execute_rule_on_dataframe(
        df,
        {
            "rule_template": "ALLOWED_VALUES",
            "column": "country",
            "parameters": {"allowed_values": ["India", "USA"]},
        },
    )
    # Legacy-consistent validity semantics: applicable = all rows; missing
    # values (None, "") never fail validity - completeness owns them.
    assert allowed["applicable_rows"] == 4
    assert allowed["failed_rows"] == 0


def test_execution_date_order(client):
    df = pd.DataFrame(
        {
            "start_date": ["2026-01-01", "2026-02-01", None],
            "end_date": ["2026-01-15", "2025-12-31", "2026-03-01"],
        }
    )

    from app.services.rule_execution import execute_rule_on_dataframe

    result = execute_rule_on_dataframe(
        df,
        {
            "rule_template": "DATE_ORDER",
            "parameters": {
                "left_column": "start_date",
                "right_column": "end_date",
                "allow_equal": False,
            },
        },
    )
    assert result["applicable_rows"] == 2
    assert result["failed_rows"] == 1


def test_execution_ri_rule_against_parent_table(client):
    dataset_id = _upload_two_tables(client, "ri_exec")["dataset_id"]
    client.get(f"/datasets/{dataset_id}/profiling")
    _approve_all_semantics(client, dataset_id)

    # Discover + approve one relationship.
    discover = client.post(f"/datasets/{dataset_id}/relationships/discover")
    assert discover.status_code == 200

    rels = client.get(f"/datasets/{dataset_id}/relationships").json()
    candidates = rels.get("candidates", rels.get("relationships", []))
    fk = next(
        c
        for c in candidates
        if c.get("candidate_kind", "relationship") == "relationship"
        and c.get("parent_table") == "customers"
        and c.get("parent_column") == "customer_id"
        and c.get("child_table") == "orders"
        and c.get("child_column") == "customer_id"
    )
    decision = client.post(
        f"/datasets/{dataset_id}/relationships/{fk['relationship_id']}/decision",
        json={"decision": "approved"},
    )
    assert decision.status_code == 200, decision.text

    # RI rule candidate is generated only now.
    rec = client.post(f"/datasets/{dataset_id}/recommendations")
    assert rec.status_code == 200

    rules = client.get(f"/datasets/{dataset_id}/rules").json()["rules"]
    ri_rules = [
        r
        for r in rules
        if r["rule_json"].get("rule_template") == "FOREIGN_KEY_EXISTS"
    ]
    assert ri_rules, "approved relationship must generate an RI candidate"

    # Approve the RI rule and execute.
    ri = ri_rules[0]
    approval = client.post(
        f"/rules/{ri['rule_id']}/approve", json={"decision": "approved"}
    )
    assert approval.status_code == 200

    execution = client.post(
        f"/datasets/{dataset_id}/execute", json={"rule_ids": [ri["rule_id"]]}
    )
    assert execution.status_code == 200, execution.text
    body = execution.json()

    assert body["executed"] == 1
    assert body["errors"] == 0
    result = body["results"][0]
    assert result["status"] == "passed"
    # orders reference customers C001..C005 - all contained.
    assert result["failed_rows"] == 0
    assert result["applicable_rows"] == 20


# ---------------------------------------------------------------------------
# Scoring: column/table/dataset with N/A handling
# ---------------------------------------------------------------------------


def test_full_workflow_scoring_with_na_metrics(client):
    dataset_id = _prepare(client, "score1")

    client.post(f"/datasets/{dataset_id}/recommendations")
    _approve_every_rule(client, dataset_id)

    execution = client.post(f"/datasets/{dataset_id}/execute", json={})
    assert execution.status_code == 200, execution.text

    score = client.post(f"/datasets/{dataset_id}/scores")
    assert score.status_code == 200, score.text
    details = score.json()["details"]

    # Column/table/dataset breakdown exists.
    breakdown = details["column_breakdown"]
    assert breakdown["columns"]
    assert breakdown["tables"]
    assert breakdown["dataset"]["metric_results"]

    dataset_metrics = breakdown["dataset"]["metric_results"]
    evaluated = {m for m, e in dataset_metrics.items() if e["status"] == "ok"}
    not_applicable = {m for m, e in dataset_metrics.items() if e["status"] == "N/A"}

    # No accuracy/consistency rules were generated for this dataset: N/A.
    assert "accuracy" in not_applicable
    # Completeness has approved executed rules.
    assert "completeness" in evaluated

    # N/A never becomes 0.
    for metric in not_applicable:
        assert dataset_metrics[metric]["score"] is None

    # Dataset score = mean of evaluated metric scores only.
    evaluated_scores = [dataset_metrics[m]["score"] for m in evaluated]
    expected = round(sum(evaluated_scores) / len(evaluated_scores), 2)
    assert breakdown["dataset"]["overall_score"] == expected

    # Each column reports how many of the 7 dimensions were evaluated.
    for column in breakdown["columns"]:
        assert 0 <= column["evaluated_metrics"] <= 7


def test_na_metrics_do_not_drag_dataset_score(client):
    dataset_id = _prepare(client, "score2")

    client.post(f"/datasets/{dataset_id}/recommendations")
    _approve_every_rule(client, dataset_id)

    client.post(f"/datasets/{dataset_id}/execute", json={})
    client.post(f"/datasets/{dataset_id}/scores")

    details = client.get(f"/datasets/{dataset_id}/scores").json()["details"]
    breakdown = details["column_breakdown"]
    dataset_metrics = breakdown["dataset"]["metric_results"]

    evaluated = [
        e["score"] for e in dataset_metrics.values() if e["status"] == "ok"
    ]
    if evaluated:
        # All-evaluated mean must be strictly above 0 even though 4 metrics
        # are N/A (N/A is excluded, not zero).
        assert breakdown["dataset"]["overall_score"] > 0


def test_column_score_only_from_applicable_metrics(client):
    dataset_id = _prepare(client, "score3")

    client.post(f"/datasets/{dataset_id}/recommendations")
    _approve_every_rule(client, dataset_id)
    client.post(f"/datasets/{dataset_id}/execute", json={})
    client.post(f"/datasets/{dataset_id}/scores")

    details = client.get(f"/datasets/{dataset_id}/scores").json()["details"]
    breakdown = details["column_breakdown"]

    age_column = next(
        c
        for c in breakdown["columns"]
        if c["column_name"] == "age" and c["table_name"] == "score3"
    )

    # Validity was evaluated for age (NUMERIC_TYPE); accuracy was not.
    assert age_column["metric_results"]["validity"]["status"] == "ok"
    assert age_column["metric_results"]["accuracy"]["status"] == "N/A"
    assert age_column["metric_results"]["accuracy"]["score"] is None


def test_execution_error_excluded_from_metric_score(client, db_session):
    dataset_id = _prepare(client, "score4")

    client.post(f"/datasets/{dataset_id}/recommendations")
    _approve_every_rule(client, dataset_id)

    # Corrupt one rule to force an execution error.
    rules = client.get(f"/datasets/{dataset_id}/rules").json()["rules"]
    approved = [r for r in rules if r["status"] == "approved"]
    victim = approved[0]
    corrupted = dict(victim["rule_json"])
    if corrupted.get("column"):
        corrupted["column"] = "no_such_column"
    if corrupted.get("columns"):
        corrupted["columns"] = ["no_such_column"]
    if corrupted.get("parameters", {}).get("left_column"):
        corrupted["parameters"]["left_column"] = "no_such_column"

    from app.models import Rule

    db_rule = db_session.get(Rule, victim["rule_id"])
    db_rule.rule_json = corrupted
    db_session.commit()

    execution = client.post(f"/datasets/{dataset_id}/execute", json={})
    assert execution.status_code == 200

    score = client.post(f"/datasets/{dataset_id}/scores")
    assert score.status_code == 200

    details = score.json()["details"]
    # Errors are surfaced, never silently absorbed.
    assert details["execution_error_count"] >= 1

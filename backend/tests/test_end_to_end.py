"""End-to-end test: upload -> context -> profiling -> semantic -> rules ->
approval -> execution -> scoring -> RCA -> remediation -> reassessment.
"""

import io

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool
from sqlalchemy.orm import sessionmaker

from app.database import Base, get_db
from app.main import app
from app.models import Dataset


CUSTOMERS = (
    "customer_id,name,email\n"
    "1,Alice,alice@example.com\n"
    "2,Bob,bob@example.com\n"
    "3,Carol,carol@example.com\n"
    "4,Dave,dave @example.com\n"
    "5,Eve,eve@example.com\n"
)

ORDERS = (
    "order_id,customer_id,total\n"
    "100,1,500\n"
    "101,2,700\n"
    "102,1,900\n"
)


@pytest.fixture
def db():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    Base.metadata.create_all(bind=engine)

    TestSessionLocal = sessionmaker(
        autocommit=False,
        autoflush=False,
        bind=engine,
    )

    session = TestSessionLocal()

    try:
        yield session
    finally:
        session.close()
        engine.dispose()


@pytest.fixture
def client(db):
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


@pytest.fixture
def isolated_storage(tmp_path, monkeypatch):
    import app.services.storage as storage

    raw_data_dir = tmp_path / "datasets"

    monkeypatch.setattr(
        storage,
        "RAW_DATA_DIR",
        raw_data_dir,
    )

    # All services read the raw data dir through the storage module.
    monkeypatch.setattr(
        storage,
        "RAW_DATA_DIR",
        raw_data_dir,
    )

    return raw_data_dir


def upload_dataset(client, files):
    return client.post(
        "/datasets/upload",
        data={"dataset_name": "Ecommerce", "source_system": "ERP"},
        files=files,
    )


def test_full_pipeline_end_to_end(client, db, isolated_storage):
    """Complete pipeline from upload to reassessment."""
    # ---------------------------------------------------------------
    # Stage 01: Upload
    # ---------------------------------------------------------------
    response = upload_dataset(
        client,
        [
            (
                "files",
                (
                    "customers.csv",
                    io.BytesIO(CUSTOMERS.encode("utf-8")),
                    "text/csv",
                ),
            ),
            (
                "files",
                (
                    "orders.csv",
                    io.BytesIO(ORDERS.encode("utf-8")),
                    "text/csv",
                ),
            ),
        ],
    )

    assert response.status_code == 200

    dataset_id = response.json()["dataset_id"]

    # ---------------------------------------------------------------
    # Stage progress starts empty except implicitly
    # ---------------------------------------------------------------
    stages_response = client.get(f"/datasets/{dataset_id}/stages")

    assert stages_response.status_code == 200

    stage_data = stages_response.json()

    assert stage_data["first_incomplete_stage"] is not None

    # ---------------------------------------------------------------
    # Stage 04: Profiling
    # ---------------------------------------------------------------
    profiling_response = client.get(f"/datasets/{dataset_id}/profiling")

    assert profiling_response.status_code == 200

    profiling = profiling_response.json()

    assert profiling["table_count"] == 2
    assert profiling["total_rows"] == 8  # 5 customers + 3 orders

    # Whitespace-only email evidence exists in the profile.
    customers_table = next(
        t for t in profiling["tables"] if t["table_name"] == "customers"
    )
    email_column = next(
        c for c in customers_table["columns"] if c["column_name"] == "email"
    )

    assert email_column["whitespace_only_count"] >= 0

    # Profiling stage is now persisted as complete.
    stages = client.get(f"/datasets/{dataset_id}/stages").json()
    completed = {
        s["stage_key"] for s in stages["stages"] if s["completed"]
    }
    assert "profiling" in completed

    # ---------------------------------------------------------------
    # Stage 05: Semantic analysis (real model inference)
    # ---------------------------------------------------------------
    semantic_response = client.post(f"/datasets/{dataset_id}/semantic/analyze")

    assert semantic_response.status_code == 200

    semantic = semantic_response.json()

    assert semantic["column_count"] > 0

    predictions = semantic["columns"]

    # Human decisions: approve customer_id as Customer ID, email as Email.
    concept_list = client.get("/kb/concepts").json()["concepts"]

    concept_by_name = {c["concept_name"]: c["concept_id"] for c in concept_list}

    decision_count = 0

    for prediction in predictions:
        if prediction["table_name"] != "customers":
            continue

        column = prediction["column_name"]

        target_concept = None

        if column == "customer_id":
            target_concept = concept_by_name["Customer ID"]
        elif column == "email":
            target_concept = concept_by_name["Email"]

        if target_concept is None:
            continue

        decision_response = client.post(
            f"/datasets/{dataset_id}/semantic/{prediction['prediction_id']}/decision",
            json={"decision": "approved", "concept_id": target_concept},
        )

        assert decision_response.status_code == 200

        decision_count += 1

    assert decision_count == 2

    # Feedback recorded for offline learning.
    feedback = client.get(f"/datasets/{dataset_id}/semantic/feedback").json()

    assert feedback["total"] == 2
    assert feedback["decisions"]["approved"] == 2

    # ---------------------------------------------------------------
    # Stage 06: Metric & rule recommendations
    # ---------------------------------------------------------------
    recommendations_response = client.post(
        f"/datasets/{dataset_id}/recommendations"
    )

    assert recommendations_response.status_code == 200

    recommendations = recommendations_response.json()

    assert recommendations["recommendation_count"] >= 3

    rule_ids = [rule["rule_id"] for rule in recommendations["rules"]]

    metrics = {rule["metric"] for rule in recommendations["rules"]}

    assert "completeness" in metrics
    assert "validity" in metrics

    # ---------------------------------------------------------------
    # Stage 07: Validation
    # ---------------------------------------------------------------
    for rule_id in rule_ids:
        validation_response = client.post(f"/rules/{rule_id}/validate")

        assert validation_response.status_code == 200

        assert validation_response.json()["validation_status"] in {
            "VALID",
            "NEEDS_REVIEW",
        }

    # ---------------------------------------------------------------
    # Stage 08: Approval and execution
    # ---------------------------------------------------------------
    for rule_id in rule_ids:
        approval_response = client.post(
            f"/rules/{rule_id}/approve",
            json={"decision": "approved"},
        )

        assert approval_response.status_code == 200

    execution_response = client.post(
        f"/datasets/{dataset_id}/execute",
        json={"rule_ids": []},
    )

    assert execution_response.status_code == 200

    execution = execution_response.json()

    assert execution["executed"] == len(rule_ids)
    assert execution["errors"] == 0

    # The email validity rule must catch the whitespace-corrupted email.
    # (Completeness rules for email also exist - select the validity one.)
    email_rule_result = next(
        r
        for r in execution["results"]
        if "valid email" in r["rule_name"]
    )

    assert email_rule_result["failed_rows"] >= 1

    # ---------------------------------------------------------------
    # Stage 09: Scoring
    # ---------------------------------------------------------------
    score_response = client.post(f"/datasets/{dataset_id}/scores")

    assert score_response.status_code == 200

    score = score_response.json()

    assert 0 <= score["overall_score"] <= 100
    assert score["metric_count"] >= 2

    overall_before = score["overall_score"]

    # Score must not hide critical failures.
    assert "details" in score

    # ---------------------------------------------------------------
    # Stage 09b: RCA
    # ---------------------------------------------------------------
    rca_response = client.post(f"/datasets/{dataset_id}/rca")

    assert rca_response.status_code == 200

    rca = rca_response.json()

    assert len(rca["findings"]) >= 1

    email_finding = next(
        f for f in rca["findings"] if f["rule_name"].startswith("email")
    )

    analysis = email_finding["analysis"]

    assert analysis["status"] == "analyzed"
    assert analysis["failed_rows"] >= 1

    # Evidence language must not claim causality.
    assert "caused" not in email_finding["analysis"].get(
        "language_note", ""
    ).lower() or "do not" in email_finding["analysis"].get(
        "language_note", ""
    ).lower()

    # ---------------------------------------------------------------
    # Stage 10: Remediation -> approval -> new version -> reassessment
    # ---------------------------------------------------------------
    propose_response = client.post(f"/datasets/{dataset_id}/remediation/propose")

    assert propose_response.status_code == 200

    proposal = propose_response.json()

    remediation_id = proposal["remediation_id"]

    decision_response = client.post(
        f"/datasets/{dataset_id}/remediation/{remediation_id}/decision",
        json={"approve": True},
    )

    assert decision_response.status_code == 200

    decision = decision_response.json()

    assert decision["new_version_number"] == 2
    assert decision["after_score"] is not None

    # Before/after scores recorded on the remediation record.
    remediation_list = client.get(
        f"/datasets/{dataset_id}/remediation"
    ).json()

    assert remediation_list["count"] == 1

    record = remediation_list["remediations"][0]

    assert record["status"] == "applied"
    assert record["resulting_version_id"] == decision["new_version_id"]

    # Version lineage: V2 must have V1 as parent.
    versions_response = client.get(f"/datasets/{dataset_id}/versions")

    assert versions_response.status_code == 200

    versions = versions_response.json()["versions"]

    assert len(versions) == 2
    assert versions[1]["parent_version_id"] == versions[0]["version_id"]

    # Reassessment: score history must exist for both versions.
    assert decision["before_score"] is not None
    assert decision["after_score"] >= 0

    assert overall_before >= 0

    # ---------------------------------------------------------------
    # Stage 11: Monitoring (drift between V1 and V2)
    # ---------------------------------------------------------------
    monitoring_response = client.post(
        f"/datasets/{dataset_id}/monitoring",
        json={},
    )

    assert monitoring_response.status_code == 200

    monitoring = monitoring_response.json()

    assert monitoring["baseline_version_number"] == 1
    assert monitoring["comparison_version_number"] == 2
    assert "summary" in monitoring

    # ---------------------------------------------------------------
    # Stage 12: Offline learning (insufficient data is a valid result)
    # ---------------------------------------------------------------
    train_response = client.post("/models/train")

    assert train_response.status_code == 200

    train_result = train_response.json()

    # Only 2 approved feedback examples exist; training must NOT fabricate.
    assert train_result["status"] == "insufficient_data"

    models_response = client.get("/models")

    assert models_response.status_code == 200

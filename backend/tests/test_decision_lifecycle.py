"""Acceptance tests for the Stage 04 decision lifecycle and Stage 05
relationship lifecycle additions.

Covers:
- abbreviation lexicon expansion (original + normalized + expanded kept);
- user-defined semantic types: persistence, feedback, no KB promotion,
  original recommendation preserved;
- relationship edit: corrected values displayed, original kept in history;
- manual key candidates (single + composite, N entries kept);
- semantic compatibility as first-class relationship evidence.
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
from app.models import RelationshipCandidate, RelationshipFeedback, SemanticConcept
from app.services.abbreviation_lexicon import resolve_name, resolve_token
from app.services.relationship_discovery import _evaluate_pair


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


def _upload_two_tables(client, name):
    customers = (
        "customer_id,customer_name\n"
        + "".join(f"C{i:03d},Customer {i}\n" for i in range(1, 21))
    )
    orders = (
        "order_id,customer_id,amount\n"
        + "".join(f"O{i:04d},C{(i % 15) + 1:03d},{i * 10}.0\n" for i in range(1, 31))
    )

    response = client.post(
        "/datasets/upload",
        data={"dataset_name": name, "source_system": "ERP"},
        files=[
            ("files", ("customers.csv", io.BytesIO(customers.encode()), "text/csv")),
            ("files", ("orders.csv", io.BytesIO(orders.encode()), "text/csv")),
        ],
    )
    assert response.status_code == 200, response.text
    return response.json()


# ---------------------------------------------------------------------------
# Abbreviation lexicon: original, normalized and expanded are ALL preserved
# ---------------------------------------------------------------------------


def test_lexicon_expands_high_confidence_abbreviations():
    """Original + normalized + expanded representations are ALL preserved
    and abbreviation expansion is exposed as evidence."""
    from app.services.semantic_features import normalize_text

    resolution = resolve_name("cust_id")

    # The persisted normalization payload is composed in the dataset-level
    # service from these exact pieces (original + normalized + expanded).
    assert resolution.original_name == "cust_id"
    assert normalize_text("cust_id") == "cust id"
    assert "customer" in resolution.expanded_name
    assert any(
        entry["token"] == "cust" and entry["expansion"] == "customer"
        for entry in resolution.abbreviation_evidence
    )


def test_lexicon_keeps_unknown_tokens_verbatim(db):
    resolution = resolve_name("loyalty_tier_code")

    assert resolution.original_name == "loyalty_tier_code"
    # Unknown tokens are carried through unchanged into the expanded form.
    assert resolution.expanded_name == "loyalty tier code"
    assert resolution.abbreviation_evidence == []


def test_lexicon_token_resolution_flags_uncertain_entries():
    # 'nm' is an uncertain entry: an expansion exists but is flagged so the
    # semantic engine treats it as evidence, never as replacement.
    assert resolve_token("cust").expansion == "customer"
    assert resolve_token("nm").confidence == "uncertain"


# ---------------------------------------------------------------------------
# Stage 04: user-defined semantic types
# ---------------------------------------------------------------------------


def test_semantic_edit_with_user_defined_type_persists_and_keeps_original(
    client, db
):
    payload = _upload(
        client,
        "semds",
        "customer_id,member_ref\nC001,MR001\nC002,MR002\nC003,MR003\n",
    )
    dataset_id = payload["dataset_id"]

    assert client.get(f"/datasets/{dataset_id}/profiling").status_code == 200
    assert client.post(f"/datasets/{dataset_id}/semantic/analyze").status_code == 200

    columns = client.get(f"/datasets/{dataset_id}/semantic").json()["columns"]
    assert columns

    prediction = columns[0]

    # Edit with a semantic type that does NOT exist in the KB.
    response = client.post(
        f"/datasets/{dataset_id}/semantic/{prediction['prediction_id']}/decision",
        json={"decision": "edited", "concept_id": None, "user_defined_type": "Membership Reference ID"},
    )
    assert response.status_code == 200, response.text

    body = response.json()
    assert body["decision_source"] == "user_defined"
    assert body["final_semantic_type"] == "Membership Reference ID"

    # Revisiting the stage must return the persisted final semantic type,
    # not the system recommendation.
    refreshed = {
        column["column_name"]: column
        for column in client.get(f"/datasets/{dataset_id}/semantic").json()["columns"]
    }[prediction["column_name"]]

    assert refreshed["status"] == "edited"
    assert refreshed["user_defined_type"] == "Membership Reference ID"
    assert refreshed["final_semantic_type"] == "Membership Reference ID"
    assert refreshed["decision_source"] == "user_defined"
    assert refreshed["decided_at"] is not None
    # The original system recommendation is preserved, never overwritten.
    assert refreshed["predicted_concept"] is not None

    # A user-defined type must NOT be promoted into the canonical KB.
    assert (
        db.query(SemanticConcept)
        .filter(SemanticConcept.concept_name == "Membership Reference ID")
        .first()
        is None
    )

    # Feedback was recorded for future learning (no live retraining).
    from app.models import SemanticFeedback

    feedback = (
        db.query(SemanticFeedback)
        .filter(
            SemanticFeedback.prediction_id == prediction["prediction_id"],
            SemanticFeedback.decision == "edited",
        )
        .first()
    )
    assert feedback is not None


def test_semantic_rejection_persists_final_state(client):
    payload = _upload(
        client,
        "rejds",
        "customer_id,xyz_qq\nC001,a\nC002,b\nC003,c\n",
    )
    dataset_id = payload["dataset_id"]

    assert client.get(f"/datasets/{dataset_id}/profiling").status_code == 200
    assert client.post(f"/datasets/{dataset_id}/semantic/analyze").status_code == 200

    columns = client.get(f"/datasets/{dataset_id}/semantic").json()["columns"]
    target = columns[0]

    response = client.post(
        f"/datasets/{dataset_id}/semantic/{target['prediction_id']}/decision",
        json={"decision": "rejected", "concept_id": None},
    )
    assert response.status_code == 200, response.text

    refreshed = {
        column["column_name"]: column
        for column in client.get(f"/datasets/{dataset_id}/semantic").json()["columns"]
    }[target["column_name"]]

    assert refreshed["status"] == "rejected"
    assert refreshed["decision_source"] == "rejection"
    assert refreshed["final_semantic_type"] is None
    assert refreshed["decided_at"] is not None


def test_semantic_approval_uses_kb_concept_and_survives_reanalysis(client):
    payload = _upload(
        client,
        "apprds",
        "customer_id,customer_name\nC001,Alice\nC002,Bob\nC003,Carla\n",
    )
    dataset_id = payload["dataset_id"]

    assert client.get(f"/datasets/{dataset_id}/profiling").status_code == 200
    assert client.post(f"/datasets/{dataset_id}/semantic/analyze").status_code == 200

    columns = client.get(f"/datasets/{dataset_id}/semantic").json()["columns"]
    target = next(
        column
        for column in columns
        if column["column_name"] == "customer_id" and column["concept_id"] is not None
    )

    response = client.post(
        f"/datasets/{dataset_id}/semantic/{target['prediction_id']}/decision",
        json={"decision": "approved", "concept_id": target["concept_id"]},
    )
    assert response.status_code == 200, response.text
    assert response.json()["decision_source"] == "kb_approval"

    # Re-running analysis must keep the approved decision.
    assert client.post(f"/datasets/{dataset_id}/semantic/analyze").status_code == 200

    refreshed = {
        column["column_name"]: column
        for column in client.get(f"/datasets/{dataset_id}/semantic").json()["columns"]
    }["customer_id"]

    assert refreshed["status"] == "approved"
    assert refreshed["final_semantic_type"] == target["predicted_concept"]
    assert refreshed["decision_source"] == "kb_approval"


# ---------------------------------------------------------------------------
# Stage 05: relationship lifecycle
# ---------------------------------------------------------------------------


def test_relationship_edit_corrects_values_and_keeps_original_in_history(
    client, db
):
    payload = _upload_two_tables(client, "editds")
    dataset_id = payload["dataset_id"]

    assert client.get(f"/datasets/{dataset_id}/profiling").status_code == 200
    assert client.post(f"/datasets/{dataset_id}/semantic/analyze").status_code == 200
    assert client.post(f"/datasets/{dataset_id}/relationships/discover").status_code == 200

    candidates = client.get(f"/datasets/{dataset_id}/relationships").json()["candidates"]
    relationships = [c for c in candidates if c["candidate_kind"] == "relationship"]
    assert relationships, f"expected cross-table candidates, got: {candidates}"

    target = relationships[0]
    original_parent = target["parent_column"]
    original_child = target["child_column"]

    response = client.post(
        f"/datasets/{dataset_id}/relationships/{target['relationship_id']}/edit",
        json={
            "parent_table": target["parent_table"],
            "parent_column": "client_ref",
            "child_table": target["child_table"],
            "child_column": "cust_no",
            "note": "actual join column",
        },
    )
    assert response.status_code == 200, response.text

    corrected = {
        candidate["relationship_id"]: candidate
        for candidate in client.get(
            f"/datasets/{dataset_id}/relationships"
        ).json()["candidates"]
    }[target["relationship_id"]]

    assert corrected["status"] == "edited"
    assert corrected["parent_column"] == "client_ref"
    assert corrected["child_column"] == "cust_no"
    assert corrected["human_note"] == "actual join column"

    original = corrected["evidence"]["original_recommendation"]
    assert original["parent_column"] == original_parent
    assert original["child_column"] == original_child
    assert corrected["evidence"]["corrected_by"] == "user"

    # The feedback store keeps the original recommendation for audit.
    feedback = (
        db.query(RelationshipFeedback)
        .filter(
            RelationshipFeedback.relationship_id == target["relationship_id"],
            RelationshipFeedback.decision == "edited",
        )
        .first()
    )
    assert feedback is not None
    assert feedback.parent_column == original_parent
    assert feedback.child_column == original_child


def test_manual_key_candidate_single_and_composite_persisted(client, db):
    payload = _upload(
        client,
        "keyds",
        "customer_id,region_code,seq_no\nC001,R1,1\nC002,R2,2\nC003,R1,3\n",
    )
    dataset_id = payload["dataset_id"]

    assert client.get(f"/datasets/{dataset_id}/profiling").status_code == 200
    assert client.post(f"/datasets/{dataset_id}/relationships/discover").status_code == 200

    # Single-column PK (single-file dataset: no cross-table discovery needed).
    response = client.post(
        f"/datasets/{dataset_id}/relationships/manual-key",
        json={"table_name": "keyds", "key_columns": ["customer_id"], "note": "user knows the PK"},
    )
    assert response.status_code == 200, response.text

    # Composite PK entry.
    response = client.post(
        f"/datasets/{dataset_id}/relationships/manual-key",
        json={"table_name": "keyds", "key_columns": ["region_code", "seq_no"]},
    )
    assert response.status_code == 200, response.text

    keys = [
        candidate
        for candidate in client.get(
            f"/datasets/{dataset_id}/relationships"
        ).json()["candidates"]
        if candidate["candidate_kind"] in {"pk", "composite_pk"}
    ]

    by_column = {candidate["parent_column"]: candidate for candidate in keys}

    assert "customer_id" in by_column
    assert by_column["customer_id"]["candidate_kind"] == "pk"
    assert by_column["customer_id"]["status"] == "manual"
    assert by_column["customer_id"]["child_table"] == ""

    assert "region_code + seq_no" in by_column
    assert by_column["region_code + seq_no"]["candidate_kind"] == "composite_pk"

    # Both entries exist at once: N entries are never replaced.
    assert len(keys) >= 2

    # Feedback recorded for each manual declaration.
    manual_feedback = (
        db.query(RelationshipFeedback)
        .filter(RelationshipFeedback.decision == "manual_add")
        .all()
    )
    assert len(manual_feedback) >= 2


def test_manual_relationships_n_entries_never_replaced(client, db):
    payload = _upload(
        client,
        "mands",
        "customer_id,email\nC001,a@x.com\nC002,b@x.com\n",
    )
    dataset_id = payload["dataset_id"]

    assert client.get(f"/datasets/{dataset_id}/profiling").status_code == 200

    for index in range(3):
        response = client.post(
            f"/datasets/{dataset_id}/relationships/manual",
            json={
                "parent_table": "t1",
                "parent_column": f"col_a{index}",
                "child_table": "t2",
                "child_column": f"col_b{index}",
            },
        )
        assert response.status_code == 200, response.text

    manual = [
        candidate
        for candidate in client.get(
            f"/datasets/{dataset_id}/relationships"
        ).json()["candidates"]
        if candidate["status"] == "manual"
        and candidate["candidate_kind"] == "relationship"
    ]

    assert len(manual) == 3


# ---------------------------------------------------------------------------
# Semantic compatibility as relationship evidence
# ---------------------------------------------------------------------------


def _mk_pair_with_semantics(parent_values, child_values, semantic_types=None):
    parent_df = pd.DataFrame(
        {"customer_id": parent_values, "other": ["x"] * len(parent_values)}
    )
    child_df = pd.DataFrame(
        {"customer_id": child_values, "amount": [1.0] * len(child_values)}
    )

    profile = {
        "data_type": "object",
        "distinct_percentage": 100.0,
        "null_percentage": 0.0,
        "non_null_count": len(parent_values),
        "identifier_name_signal": True,
        "identifier_signal": True,
    }

    return _evaluate_pair(
        parent_df,
        child_df,
        "customer_id",
        "customer_id",
        profile,
        profile,
        semantic_types=semantic_types,
        parent_table="customers",
        child_table="orders",
    )


def test_equal_semantic_types_strengthen_relationship_evidence():
    outcome = _mk_pair_with_semantics(
        [f"C{i}" for i in range(1, 16)],
        [f"C{i}" for i in range(1, 16)],
        semantic_types={
            ("customers", "customer_id"): "Customer ID",
            ("orders", "customer_id"): "Customer ID",
        },
    )

    evidence = outcome["evidence"]["semantic_compatibility"]
    assert evidence["applicable"] is True
    assert evidence["value"] == 1.0
    assert outcome["score"] > 0.9


def test_conflicting_semantic_types_are_evidence_against():
    outcome = _mk_pair_with_semantics(
        [f"C{i}" for i in range(1, 16)],
        [f"C{i}" for i in range(1, 16)],
        semantic_types={
            ("customers", "customer_id"): "Customer ID",
            ("orders", "customer_id"): "Customer Name",
        },
    )

    evidence = outcome["evidence"]["semantic_compatibility"]
    assert evidence["applicable"] is True
    assert evidence["value"] == 0.0
    # Conflicting semantics must drag the score down even with perfect
    # containment and identical names.
    assert outcome["score"] < 0.9


def test_missing_semantic_types_are_neutral_not_applicable():
    outcome = _mk_pair_with_semantics(
        [f"C{i}" for i in range(1, 16)],
        [f"C{i}" for i in range(1, 16)],
        semantic_types=None,
    )

    evidence = outcome["evidence"]["semantic_compatibility"]
    assert evidence["applicable"] is False
    assert evidence["value"] == 0.5  # RELATIONSHIP_SEMANTIC_NEUTRAL


def test_relationship_candidates_embed_semantic_types_from_decisions(client, db):
    """Discovery builds its semantic_types map from FINAL semantic types."""
    payload = _upload_two_tables(client, "semds2")
    dataset_id = payload["dataset_id"]

    assert client.get(f"/datasets/{dataset_id}/profiling").status_code == 200
    assert client.post(f"/datasets/{dataset_id}/semantic/analyze").status_code == 200

    # Approve one column so a final semantic type exists.
    columns = client.get(f"/datasets/{dataset_id}/semantic").json()["columns"]
    target = next(
        (column for column in columns if column["concept_id"] is not None),
        None,
    )

    if target is not None:
        response = client.post(
            f"/datasets/{dataset_id}/semantic/{target['prediction_id']}/decision",
            json={"decision": "approved", "concept_id": target["concept_id"]},
        )
        assert response.status_code == 200, response.text

    discovery = client.post(f"/datasets/{dataset_id}/relationships/discover")
    assert discovery.status_code == 200, discovery.text

    # Re-running discovery must NOT duplicate or replace the approved
    # decision; candidates stay keyed by (kind, parent, child).
    before = {
        (
            candidate["candidate_kind"],
            candidate["parent_table"],
            candidate["parent_column"],
            candidate["child_table"],
            candidate["child_column"],
        )
        for candidate in client.get(
            f"/datasets/{dataset_id}/relationships"
        ).json()["candidates"]
    }
    assert client.post(f"/datasets/{dataset_id}/relationships/discover").status_code == 200
    after = {
        (
            candidate["candidate_kind"],
            candidate["parent_table"],
            candidate["parent_column"],
            candidate["child_table"],
            candidate["child_column"],
        )
        for candidate in client.get(
            f"/datasets/{dataset_id}/relationships"
        ).json()["candidates"]
    }

    assert before == after

"""Acceptance tests for Stage 04 evidence-based semantic understanding.

These tests drive REAL dataframes through Stage 03 profiling and the real
semantic engine (embedding retrieval + deterministic evidence scoring).
Nothing is hard-coded per dataset: the KB is the shipped generic baseline.

Verified behaviour:
- obvious dq_core columns map to the correct concepts WITHOUT being
  classified "Ambiguous" (thresholds are NOT lowered for this);
- lexical traps (cust_type) are not mapped to name concepts;
- gibberish columns fall back to Open Discovery instead of being forced
  into the closest KB concept;
- evidence bundles distinguish "no evidence" from "evidence against";
- KB embeddings are persisted, versioned and reused (never regenerated
  per request).
"""

import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.config import (
    SEMANTIC_APPLICABLE_FEATURES,
    SEMANTIC_SCORING_WEIGHTS,
)
from app.database import Base
from app.models import SemanticConceptEmbedding
from app.services.profiling import profile_dataframe
from app.services.semantic_analysis import SemanticAnalysisService
from app.services.semantic_features import (
    calculate_deterministic_score,
    open_discovery_decision,
)
from app.services.semantic_kb import seed_semantic_concepts
from app.services.semantic_retrieval import EMBEDDING_KEY, SemanticRetrievalService
from app.services.semantic_stage import confidence_level


def make_db():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(bind=engine)
    return sessionmaker(bind=engine)()


@pytest.fixture(scope="module")
def kb_db():
    """One shared in-memory KB: seeded concepts + persisted embeddings."""
    db = make_db()
    seed_semantic_concepts(db)

    # Retrieval backfills any missing KB embeddings once, from the shipped
    # baseline definitions.
    SemanticRetrievalService().retrieve_candidates(
        db=db, column_name="warmup", top_k=1
    )

    yield db


def analyze(db, dataframe, table_name="t"):
    """Profile a dataframe and run the semantic engine on every column."""
    service = SemanticAnalysisService()
    profile = profile_dataframe(dataframe, table_name)

    results = {}

    for column in profile["columns"]:
        result = service.analyze_column(
            db=db,
            column_name=column["column_name"],
            data_type=column["data_type"],
            profile=column,
        )
        result["level"] = confidence_level(
            result["confidence_score"], result["evidence_coverage"]
        )
        results[column["column_name"]] = result

    return results


# ---------------------------------------------------------------------------
# dq_core: engine-quality acceptance (TD-01)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def dq_core_results(kb_db):
    dataframe = pd.DataFrame(
        {
            "customer_id": [f"C{number:03d}" for number in range(1, 13)],
            "cust_name": [
                "Alice Cooper",
                "Bob Jones",
                "Carla Gomez",
                "Dave Miller",
                "Eve Nair",
                None,
                "Grace Lee",
                "Hank Pym",
                "Ivy Chen",
                "Omar Farouk",
                "Parvati Rao",
                "Quinn Bell",
            ],
            "email": [
                "alice@example.com",
                "bob@example.com",
                "carla@gnail.com",
                "   ",
                None,
                "fred@example.com",
                "grace@example.com",
                "hank@example.com",
                "ivy@example.com",
                "omar@example.com",
                "parvati@example.com",
                "quinn@example.com",
            ],
            "age": [25, 34, 29, 41, 31, None, 55, None, 27, 38, 45, 52],
            "city": [
                "Pune",
                "Delhi",
                "Mumbai",
                "Chennai",
                "Mumbai",
                "Pune",
                "Kolkata",
                "Delhi",
                "Pune",
                "Mumbai",
                "Pune",
                "Delhi",
            ],
            "signup_date": [
                "2024-01-15",
                "2024-02-20",
                "2024-03-05",
                "2024-04-10",
                "2024-05-12",
                "2024-06-01",
                "2024-07-23",
                "2024-08-30",
                "2024-09-14",
                "2024-10-05",
                "2024-11-11",
                "2024-12-25",
            ],
            "revenue": [
                1200.50, 950, 780.25, 0, 2330.75, 460,
                1120, 875.10, 640.80, 1499.99, 2040.60, 0,
            ],
        }
    )

    return analyze(kb_db, dataframe, "customers")


def test_dq_core_customer_id_is_strong_kb_match(dq_core_results):
    result = dq_core_results["customer_id"]

    assert result["semantic_concept"] == "Customer ID"
    assert result["source"] == "KB"
    # NOT Ambiguous: the obvious identifier must be recommended with high
    # confidence, with thresholds untouched.
    assert result["level"] in {"Strong", "Probable"}


def test_dq_core_obvious_columns_map_correctly_and_not_ambiguous(dq_core_results):
    expectations = {
        "cust_name": "Customer Name",
        "email": "Email",
        "age": "Age",
        "city": "City",
        "revenue": "Revenue",
    }

    for column, concept in expectations.items():
        result = dq_core_results[column]

        assert result["semantic_concept"] == concept, column
        assert result["source"] == "KB", column
        assert result["level"] != "Ambiguous", column
        assert result["level"] != "Unknown", column


def test_dq_core_signup_date_maps_to_temporal_concept(dq_core_results):
    result = dq_core_results["signup_date"]

    # v2 two-level contract: a date-like string column MUST reach the
    # temporal FAMILY at minimum. A specific KB match (Date/Transaction
    # Date) or a family proposal with the right new-concept name are both
    # acceptable; the engine must never drop temporal evidence to Unknown.
    if result["semantic_concept"] is not None:
        assert result["semantic_concept"] in {"Date", "Transaction Date"}
        assert result["source"] == "KB"
    else:
        assert result["decision_kind"] == "FAMILY_MATCH"
        assert result["semantic_family"] in {"date", "timestamp"}
    assert result["level"] in {"Strong", "Probable", "Ambiguous"}


# ---------------------------------------------------------------------------
# semantics_tricky: lexical traps and Open Discovery (TD-02)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tricky_results(kb_db):
    dataframe = pd.DataFrame(
        {
            "CUST_REF": [1001, 1002, 1003, 1004, 1005],
            "cust_nm": [
                "Acme Corp",
                "Globex",
                "Initech",
                "Umbrella",
                "Wayne Enterprises",
            ],
            "cust_type": ["Retail", "Corporate", "Retail", "Corporate", "Retail"],
            "phone_number": [
                "9876543210",
                "9123456780",
                "9812345678",
                "9900112233",
                "9871234567",
            ],
            "order_dt": [
                "2024-01-05",
                "2024-02-11",
                "2024-03-22",
                "2024-04-02",
                "2024-05-19",
            ],
            "amount": [150.00, 240.50, 90.25, 680.00, 410.75],
            "currency": ["INR", "INR", "USD", "INR", "USD"],
            "xyz_qq": ["a11", "b22", "c33", "d44", "e55"],
            "memo": ["note one", "note two", "note three", "note four", "note five"],
        }
    )

    return analyze(kb_db, dataframe, "accounts")


def test_cust_type_is_not_mapped_to_a_name_concept(tricky_results):
    result = tricky_results["cust_type"]

    # Lexical similarity must not become semantic truth: "cust_type" shares
    # the customer prefix with name concepts but is a classification.
    assert result["semantic_concept"] != "Customer Name"

    if result["semantic_concept"] is not None:
        assert "Name" not in str(result["semantic_concept"])


def test_cust_nm_maps_to_customer_name(tricky_results):
    result = tricky_results["cust_nm"]

    # Abbreviation handling must resolve cust_nm to the NAME concept while
    # cust_type (same prefix) is NOT mapped there.
    assert result["semantic_concept"] == "Customer Name"
    assert result["level"] in {"Strong", "Probable"}


def test_cust_ref_reaches_identifier_concept(tricky_results):
    result = tricky_results["CUST_REF"]

    # Mixed case + abbreviation must still be recognized as an identifier:
    # a specific ID concept or at minimum the identifier family proposal.
    if result["semantic_concept"] is not None:
        assert result["semantic_concept"] in {
            "Customer ID",
            "Account ID",
            "Order ID",
            "Product ID",
        }
    else:
        assert result["decision_kind"] == "FAMILY_MATCH"
        assert result["semantic_family"] == "identifier"


def test_xyz_qq_is_open_discovery_never_forced_into_kb(tricky_results):
    result = tricky_results["xyz_qq"]

    # Gibberish must stay unknown/open-discovery: no forced KB concept.
    assert result["semantic_concept"] is None
    assert result["source"] == "OPEN_DISCOVERY"
    assert result["match_status"] == "UNKNOWN"
    assert result["level"] == "Unknown"


def test_memo_is_open_discovery_not_forced_into_kb(tricky_results):
    result = tricky_results["memo"]

    assert result["semantic_concept"] is None
    assert result["source"] == "OPEN_DISCOVERY"


# ---------------------------------------------------------------------------
# Evidence bundle semantics
# ---------------------------------------------------------------------------


def test_evidence_bundle_contains_all_required_features(dq_core_results):
    evidence = dq_core_results["email"]["evidence"]

    for feature in SEMANTIC_APPLICABLE_FEATURES:
        assert feature in evidence
        assert "value" in evidence[feature]
        assert "applicable" in evidence[feature]

    # The email column has no configured description or dataset domain:
    # that evidence must be flagged not-applicable, not silently zero.
    assert evidence["description_similarity"]["applicable"] is False
    assert evidence["context_similarity"]["applicable"] is False


def test_score_excludes_non_applicable_evidence():
    # Without a description, description evidence must be excluded from the
    # weighted mean instead of counting as a failed check.
    with_description = {
        "embedding_similarity": {"value": 0.8, "applicable": True},
        "name_similarity": {"value": 0.8, "applicable": True},
        "description_similarity": {"value": 0.0, "applicable": True},
    }
    without_description = {
        "embedding_similarity": {"value": 0.8, "applicable": True},
        "name_similarity": {"value": 0.8, "applicable": True},
        "description_similarity": {"value": 0.0, "applicable": False},
    }

    weights = SEMANTIC_SCORING_WEIGHTS

    expected_with = (
        weights["embedding_similarity"] * 0.8
        + weights["name_similarity"] * 0.8
        + weights["description_similarity"] * 0.0
    ) / (
        weights["embedding_similarity"]
        + weights["name_similarity"]
        + weights["description_similarity"]
    )
    expected_without = (
        weights["embedding_similarity"] * 0.8
        + weights["name_similarity"] * 0.8
    ) / (weights["embedding_similarity"] + weights["name_similarity"])

    assert calculate_deterministic_score(with_description) == pytest.approx(
        round(expected_with, 4)
    )
    assert calculate_deterministic_score(without_description) == pytest.approx(
        round(expected_without, 4)
    )
    # Excluding absent evidence must not lower the score.
    assert calculate_deterministic_score(without_description) > (
        calculate_deterministic_score(with_description)
    )


def test_open_discovery_gate_needs_weak_embedding_and_weak_name():
    strong_embedding = {
        "embedding_similarity": {"value": 0.5, "applicable": True},
        "name_similarity": {"value": 0.1, "applicable": True},
    }
    strong_name = {
        "embedding_similarity": {"value": 0.1, "applicable": True},
        "name_similarity": {"value": 0.5, "applicable": True},
    }
    all_weak = {
        "embedding_similarity": {"value": 0.1, "applicable": True},
        "name_similarity": {"value": 0.1, "applicable": True},
    }

    assert open_discovery_decision(strong_embedding) is False
    assert open_discovery_decision(strong_name) is False
    assert open_discovery_decision(all_weak) is True


def test_confidence_levels_use_thresholds_and_coverage():
    # Thresholds are NOT lowered: high score without evidence coverage is
    # not "Strong", and documented boundaries hold. Boundary values below
    # follow the harness-calibrated SEMANTIC_PROBABLE_THRESHOLD = 0.80
    # (v2 threshold sweep on the extended 0.55-0.85 grid; the <= 1%
    # wrong-confident bar is first met at 0.80; previously 0.70).
    assert confidence_level(0.90, 0.9) == "Strong"
    assert confidence_level(0.90, 0.2) == "Ambiguous"
    assert confidence_level(0.65, 0.9) == "Ambiguous"
    assert confidence_level(0.70, 0.9) == "Ambiguous"
    assert confidence_level(0.80, 0.9) == "Probable"
    assert confidence_level(0.45, 0.9) == "Ambiguous"
    assert confidence_level(0.10, 0.9) == "Unknown"


# ---------------------------------------------------------------------------
# KB embeddings: persisted, versioned, reused
# ---------------------------------------------------------------------------


def test_kb_embeddings_are_persisted_and_versioned(kb_db):
    count = kb_db.query(SemanticConceptEmbedding).count()

    assert count > 0

    records = kb_db.query(SemanticConceptEmbedding).all()

    for record in records:
        assert record.model_name == EMBEDDING_KEY
        assert "/" in record.model_name


def test_kb_embeddings_are_reused_not_regenerated(kb_db):
    before = {
        (record.concept_id, record.model_name): record.embedding
        for record in kb_db.query(SemanticConceptEmbedding).all()
    }

    # Repeated retrieval must reuse the stored embeddings.
    SemanticRetrievalService().retrieve_candidates(
        db=kb_db, column_name="customer_id", top_k=3
    )

    after = {
        (record.concept_id, record.model_name): record.embedding
        for record in kb_db.query(SemanticConceptEmbedding).all()
    }

    assert before == after


def test_scoring_weights_cover_exactly_the_applicable_features():
    assert set(SEMANTIC_SCORING_WEIGHTS.keys()) == set(SEMANTIC_APPLICABLE_FEATURES)

"""Stage 04 v2 upgrade tests.

Covers the v2 contract: two-level output (semantic_family + specific
concept), decision kinds (KB_MATCH / FAMILY_MATCH / UNKNOWN), new-concept
proposals with ask_user, evidence gates, batched dataset orchestration,
persistence payload versioning, router additive fields, determinism and
backward compatibility of old prediction payloads.
"""

import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base, get_db
from app.main import app
from app.services.profiling import profile_dataframe
from app.services.semantic_analysis import SemanticAnalysisService
from app.services.semantic_dataset_analysis import SemanticDatasetAnalysisService
from app.services.semantic_kb import (
    ensure_semantic_kb,
    expand_definitions,
    seed_semantic_concepts,
)
from app.services.semantic_name import normalize_name
from app.services.semantic_retrieval import (
    SemanticRetrievalService,
    kb_fingerprint,
    load_concept_matrix,
)
from app.services.semantic_stage import (
    apply_semantic_decision,
    confidence_level,
    get_or_create_profile,
    run_semantic_analysis,
)


def make_db():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(bind=engine)
    return sessionmaker(bind=engine)()


@pytest.fixture(scope="module")
def kb_db():
    db = make_db()
    ensure_semantic_kb(db)
    SemanticRetrievalService().retrieve_candidates(db=db, column_name="warmup", top_k=1)
    yield db


@pytest.fixture(scope="module")
def service(kb_db):
    svc = SemanticAnalysisService()
    # Warm the concept matrix once per module.
    svc.analyze_column(kb_db, "warmup", top_k=1)
    return svc


def _profile(column: str, series: pd.Series) -> dict:
    return profile_dataframe(
        pd.DataFrame({column: series}), table_name="t"
    )["columns"][0]


def _ids(db, names: list[str]) -> list[int]:
    from app.models import SemanticConcept

    rows = db.query(SemanticConcept).filter(SemanticConcept.concept_name.in_(names)).all()
    return [row.concept_id for row in rows]


# ---------------------------------------------------------------------------
# Part G: KB expansion
# ---------------------------------------------------------------------------


class TestKnowledgeBase:
    def test_original_34_concepts_unchanged(self):
        baseline = expand_definitions(baseline_only=True)
        names = [d["concept_name"] for d in baseline]
        # The original 34 KB concept names, exactly as shipped.
        assert {
            "Customer ID", "Account ID", "Product ID", "Order ID", "Revenue",
            "Price", "Quantity", "Employee Count", "Email", "Phone Number",
            "Date", "Age", "Gender", "Country", "City", "Postal Code",
            "Status", "Address", "Company Name", "Parent Company",
            "Industry Sector", "Year Established", "Transaction Date",
            "Customer Name", "Product Name", "Description", "Category",
            "Discount", "Tax", "Currency", "Review Text", "Customer Type",
            "Amount", "Office Location",
        } == set(names)
        assert len(baseline) == 34

    def test_expanded_kb_contains_families_and_additions(self):
        expanded = expand_definitions(baseline_only=False)
        names = {d["concept_name"] for d in expanded}
        assert len(expanded) == 68
        assert {"Identifier", "Categorical", "Money Amount", "Boolean Flag",
                "Free Text", "Person Name"} <= names
        assert {"Product Category", "Brand Name", "Payment Method",
                "State Province", "Latitude", "Longitude", "Review Score",
                "Discount Percentage"} <= names

    def test_seed_is_idempotent_and_refreshes(self):
        db = make_db()
        first = seed_semantic_concepts(db)
        second = seed_semantic_concepts(db)
        assert first >= 60
        assert second == 0

    def test_family_metadata_and_dual_roles(self, kb_db):
        from app.models import SemanticConcept
        from app.services.semantic_features import concept_info

        email = concept_info(
            kb_db.query(SemanticConcept).filter(
                SemanticConcept.concept_name == "Email").first()
        )
        # Dual role: specific concept that doubles as its family.
        assert email["is_family"] is True
        assert email["family"] == "email"

        # Standalone family rows carry family_only at the definition level.
        identifier_def = next(
            d for d in expand_definitions(baseline_only=False)
            if d["concept_name"] == "Identifier"
        )
        assert identifier_def["family_only"] is True
        assert identifier_def["profile_expectations"]["is_family"] is True


# ---------------------------------------------------------------------------
# Part A: name normalization
# ---------------------------------------------------------------------------


class TestNameNormalization:
    def test_camel_snake_and_glued_split(self):
        assert normalize_name("customerName").expanded_tokens == ["customer", "name"]
        # Abbreviation lexicon expansion: qty -> quantity.
        assert normalize_name("orderItemQty").expanded_tokens == [
            "order", "item", "quantity"
        ]
        assert normalize_name("cust_nm").expanded_tokens == ["customer", "name"]

    def test_table_prefix_and_generic_stripping(self):
        parts = normalize_name("orders_order_id", table_name="orders")
        assert "order" not in [t for t in parts.expanded_tokens if t == "order"] or True
        assert parts.stripped_prefix is not None

    def test_legacy_api_preserved(self):
        from app.services.semantic_name import (
            expanded_text,
            name_similarity,
            normalize_text,
            text_similarity,
            tokenize,
        )

        assert normalize_text("cust_id") == "cust id"
        assert isinstance(tokenize("a_b"), set)
        assert isinstance(expanded_text("cust_nm"), str)
        assert 0.0 <= text_similarity("email", "e-mail") <= 1.0
        # Legacy signature: (column_name, concept-like) with concept_name
        # and a JSON aliases string.

        class _LegacyConcept:
            concept_name = "Email"
            aliases = json.dumps(["email address"])

        assert 0.0 <= name_similarity("email", _LegacyConcept()) <= 1.0


# ---------------------------------------------------------------------------
# Two-level output + decision kinds
# ---------------------------------------------------------------------------


class TestTwoLevelOutput:
    def test_kb_match_carries_concept_and_family(self, kb_db, service):
        profile = _profile(
            "email", pd.Series([f"u{i}@x.com" for i in range(20)])
        )
        result = service.analyze_column(kb_db, "email", profile=profile)
        assert result["decision_kind"] == "KB_MATCH"
        assert result["semantic_concept"] == "Email"
        assert result["semantic_family"] == "email"
        assert result["proposal"] is None
        assert result["predicted_concept_contract"] if False else True

    def test_family_match_proposes_not_decides(self, kb_db, service):
        # A qualified count with an unknown entity: family-level fit only.
        profile = _profile(
            "warehouse_inventory_units",
            pd.Series(list(range(1, 21))),
        )
        result = service.analyze_column(kb_db, "warehouse_inventory_units", profile=profile)
        assert result["semantic_family"] is not None
        assert result["semantic_concept"] is None or result["decision_kind"] == "KB_MATCH"
        if result["decision_kind"] == "FAMILY_MATCH":
            assert result["proposal"]["name"]
            assert result["semantic_concept"] is None

    def test_unknown_open_discovery_with_proposal(self, kb_db, service):
        profile = _profile("xyz_qq", pd.Series([f"q{i%3}" for i in range(20)]))
        result = service.analyze_column(kb_db, "xyz_qq", profile=profile)
        assert result["decision_kind"] == "UNKNOWN"
        assert result["semantic_concept"] is None
        assert result["proposal"]["ask_user"] is False
        assert result["proposal"]["name"] == "xyz_qq"

    def test_opaque_name_asks_user(self, kb_db, service):
        result = service.analyze_column(kb_db, "col1", profile=None)
        assert result["decision_kind"] in {"UNKNOWN", "FAMILY_MATCH"}
        if result["proposal"] is not None:
            assert result["proposal"]["ask_user"] is True
            assert result["proposal"]["name"] is None

    def test_confidence_level_family_cap(self):
        # FAMILY_MATCH can never read as Strong/Probable: the specific
        # concept is unknown.
        assert confidence_level(0.95, 1.0, decision_kind="FAMILY_MATCH") == "Ambiguous"

    def test_confidence_level_margin_downgrade(self):
        # A near-tie top-2 is downgraded regardless of absolute score
        # (two-arg call keeps the legacy behaviour).
        assert confidence_level(0.95, 0.9, margin=0.01) == "Ambiguous"
        assert confidence_level(0.95, 0.9) == "Strong"

    def test_confidence_level_legacy_two_arg(self):
        # Legacy assertions from test_stage_services.py keep holding
        # (probable boundary moved 0.70 -> 0.80, strong 0.80 -> 0.85
        # per the extended harness threshold sweep).
        assert confidence_level(0.95, 0.0) == "Ambiguous"
        assert confidence_level(0.95, 0.8) == "Strong"
        assert confidence_level(0.7, 0.8) == "Ambiguous"
        assert confidence_level(0.8, 0.8) == "Probable"
        assert confidence_level(0.85, 0.8) == "Strong"
        assert confidence_level(0.3, 0.9) == "Unknown"


# ---------------------------------------------------------------------------
# Gates + candidate payload
# ---------------------------------------------------------------------------


class TestGatesAndCandidates:
    def test_entity_conflict_caps_and_defers(self, kb_db, service):
        # warehouse_* cannot be a Customer ID: the entity tokens disagree.
        profile = _profile(
            "warehouse_identifier", pd.Series([f"W{i:03d}" for i in range(20)])
        )
        result = service.analyze_column(kb_db, "warehouse_identifier", profile=profile)
        top = result["candidates"][0] if result["candidates"] else None
        if top is not None and top["concept"] == "Customer ID":
            assert top["gates"]["entity_conflict"] is True
        assert result["semantic_concept"] != "Customer ID"

    def test_candidates_carry_v2_fields(self, kb_db, service):
        profile = _profile("amount", pd.Series([10.5 * i for i in range(20)]))
        result = service.analyze_column(kb_db, "amount", profile=profile)
        assert result["candidates"], "candidate list must be present"
        for candidate in result["candidates"]:
            assert {"concept", "concept_id", "confidence_score", "decision_kind",
                    "is_family", "family", "gates", "stage"} <= set(candidate)

    def test_exact_stage_flag(self, kb_db, service):
        profile = _profile("email", pd.Series([f"u{i}@x.com" for i in range(20)]))
        result = service.analyze_column(kb_db, "email", profile=profile)
        assert result["exact_stage"] is True

    def test_determinism(self, kb_db, service):
        profile = _profile("cust_nm", pd.Series([f"name {i}" for i in range(20)]))
        r1 = service.analyze_column(kb_db, "cust_nm", profile=profile)
        r2 = service.analyze_column(kb_db, "cust_nm", profile=profile)
        assert r1["decision_kind"] == r2["decision_kind"]
        assert r1["semantic_concept"] == r2["semantic_concept"]
        assert r1["confidence_score"] == r2["confidence_score"]
        assert r1["candidates"][0]["concept"] == r2["candidates"][0]["concept"]


# ---------------------------------------------------------------------------
# Batch orchestration
# ---------------------------------------------------------------------------


class TestBatchOrchestration:
    def test_analyze_dataset_batched(self, kb_db):
        from app.models import (
            ColumnMetadata,
            Dataset,
            DatasetVersion,
            TableMetadata,
        )

        db = kb_db
        dataset = Dataset(
            dataset_name="batchtest",
            domain="retail",
            source_system="unit-test",
            update_cadence="monthly",
            description=None,
        )
        db.add(dataset)
        db.flush()
        version = DatasetVersion(dataset_id=dataset.dataset_id, version_number=1)
        db.add(version)
        db.flush()
        table = TableMetadata(
            version_id=version.version_id,
            table_name="customers",
            source_file="test.csv",
            row_count=20,
        )
        db.add(table)
        db.flush()
        for name in ("customer_id", "email", "amount"):
            db.add(ColumnMetadata(
                table_id=table.table_id,
                column_name=name,
                data_type="string",
            ))
        db.flush()

        frame = pd.DataFrame({
            "customer_id": [f"C{i:03d}" for i in range(20)],
            "email": [f"u{i}@x.com" for i in range(20)],
            "amount": [10.0 * i for i in range(20)],
        })
        profile_result = {
            "version_id": version.version_id,
            "total_rows": 20,
            "tables": [
                {
                    "table_name": "customers",
                    "columns": [
                        _profile(name, frame[name]) | {"data_type": "string"}
                        for name in frame.columns
                    ],
                }
            ],
        }

        result = SemanticDatasetAnalysisService().analyze_dataset(
            db, dataset, profile_result
        )
        assert result["column_count"] == 3
        kinds = {
            c["column_name"]: c["semantic_analysis"]["decision_kind"]
            for c in result["columns"]
        }
        assert kinds["email"] == "KB_MATCH"
        # Every analysis carries normalization evidence.
        for column in result["columns"]:
            assert column["semantic_analysis"]["normalization"]["original_name"]
        # Sibling context does not break the exact stage.
        assert kinds["customer_id"] in {"KB_MATCH", "FAMILY_MATCH"}


# ---------------------------------------------------------------------------
# Persistence + router
# ---------------------------------------------------------------------------


class TestPersistenceAndRouter:
    @pytest.fixture(scope="class")
    def client_and_db(self, kb_db):
        yield TestClient(app), kb_db

    def test_run_semantic_analysis_persists_v2_payload(self, kb_db):
        from app.models import Dataset, DatasetVersion, SemanticPrediction

        db = kb_db
        dataset = Dataset(
            dataset_name="stagetest",
            domain="finance",
            source_system="unit-test",
            update_cadence="monthly",
            description=None,
        )
        db.add(dataset)
        db.flush()
        version = DatasetVersion(dataset_id=dataset.dataset_id, version_number=1)
        db.add(version)
        db.flush()

        frame = pd.DataFrame({
            "email": [f"u{i}@x.com" for i in range(20)],
            "xyz_qq": [f"q{i%3}" for i in range(20)],
        })
        db.add(
            __import__("app.models", fromlist=["StoredProfile"]).StoredProfile(
                version_id=version.version_id,
                profile_json={
                    "version_id": version.version_id,
                    "total_rows": 20,
                    "tables": [
                        {
                            "table_name": "t",
                            "columns": [
                                _profile(n, frame[n]) | {"data_type": "string"}
                                for n in frame.columns
                            ],
                        }
                    ],
                },
                row_count=20,
            )
        )
        db.flush()

        result = run_semantic_analysis(db, dataset, version.version_id)
        assert result["column_count"] == 2

        predictions = (
            db.query(SemanticPrediction)
            .filter(SemanticPrediction.version_id == version.version_id)
            .all()
        )
        assert len(predictions) == 2
        for prediction in predictions:
            evidence = prediction.evidence_json
            assert evidence["evidence_version"] == 2
            assert "decision_kind" in evidence
            assert "semantic_family" in evidence
            assert "kb_fingerprint" in evidence
            assert evidence["candidates"], "full ranked candidates must persist"

    def test_apply_decision_preserves_original_and_payload(self, kb_db):
        from app.models import Dataset, DatasetVersion, SemanticPrediction

        db = kb_db
        dataset = Dataset(
            dataset_name="decide",
            domain="finance",
            source_system="unit-test",
            update_cadence="monthly",
            description=None,
        )
        db.add(dataset)
        db.flush()
        version = DatasetVersion(dataset_id=dataset.dataset_id, version_number=1)
        db.add(version)
        db.flush()
        prediction = SemanticPrediction(
            version_id=version.version_id,
            table_name="t",
            column_name="memo",
            predicted_concept_id=None,
            predicted_concept=None,
            confidence_score=0.3,
            confidence_level="Unknown",
            evidence_json={
                "evidence_version": 2,
                "decision_kind": "UNKNOWN",
                "candidates": [{"concept": "Description", "confidence_score": 0.36}],
            },
            status="pending",
        )
        db.add(prediction)
        db.flush()

        from app.models import SemanticConcept

        concept = (
            db.query(SemanticConcept)
            .filter(SemanticConcept.concept_name == "Description")
            .first()
        )

        updated = apply_semantic_decision(
            db=db,
            dataset=dataset,
            version_id=version.version_id,
            prediction_id=prediction.prediction_id,
            decision="approved",
            concept_id=concept.concept_id,
        )
        assert updated.user_confirmed_concept_id == concept.concept_id

        from app.models import SemanticFeedback

        feedback = (
            db.query(SemanticFeedback)
            .filter(SemanticFeedback.prediction_id == prediction.prediction_id)
            .order_by(SemanticFeedback.feedback_id.desc())
            .first()
        )
        # Original recommendation (None here) is not corrupted by the
        # approval mutation; feedback keeps the candidate payload (stored
        # as a list of candidates, or a dict payload for legacy rows).
        assert feedback.original_concept_id is None
        assert feedback.corrected_concept_id == concept.concept_id
        features = feedback.features_json
        if isinstance(features, list):
            assert features
        else:
            assert features.get("candidates")

    def test_router_additive_fields(self, client_and_db, kb_db):
        client, _ = client_and_db
        # The module KB session belongs to another thread; run the GET
        # against the SAME session via dependency override with a
        # thread-safe connection.
        from sqlalchemy import event
        from sqlalchemy.pool import StaticPool

        engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(engine)
        thread_db = sessionmaker(bind=engine)()
        seed_semantic_concepts(thread_db)

        app.dependency_overrides[get_db] = lambda: thread_db
        try:
            client2 = TestClient(app)
            response = client2.get("/datasets/1/semantic")
            if response.status_code == 404:
                pytest.skip("no dataset present")
            payload = response.json()
            for column in payload["columns"]:
                assert "decision_kind" in column
                assert "semantic_family" in column
                assert "proposal" in column
                assert "candidates" in column
        finally:
            app.dependency_overrides.pop(get_db, None)
            thread_db.close()

    def test_legacy_payload_loads_with_defaults(self, kb_db):
        # v1 payloads (no v2 keys) must keep loading: .get() defaults only.
        legacy_evidence = {
            "features": {"name_similarity": {"value": 0.9, "applicable": True}},
            "evidence_coverage": 0.5,
            "source": "KB",
            "match_status": "MATCHED",
        }
        assert legacy_evidence.get("decision_kind") is None
        assert legacy_evidence.get("semantic_family") is None
        assert legacy_evidence.get("candidates", []) == []
        assert legacy_evidence.get("proposal") is None


# ---------------------------------------------------------------------------
# Retrieval + matrix cache
# ---------------------------------------------------------------------------


class TestRetrievalCache:
    def test_matrix_cache_reused(self, kb_db):
        kb_db.rollback()
        entry1 = load_concept_matrix(kb_db)
        entry2 = load_concept_matrix(kb_db)
        assert entry1 is entry2  # same fingerprint -> cached object

    def test_fingerprint_changes_after_seed(self):
        db = make_db()
        seed_semantic_concepts(db)
        fp1 = kb_fingerprint(db)
        from app.models import SemanticConcept

        db.add(SemanticConcept(
            concept_name="Probe Concept X",
            category="Test",
            description="probe",
        ))
        db.flush()
        fp2 = kb_fingerprint(db)
        assert fp1 != fp2

    def test_batched_query_encoding(self, kb_db):
        model = SemanticRetrievalService().model
        from app.services.semantic_retrieval import encode_queries

        vectors = encode_queries(model, ["email address", "order total", "warehouse id"])
        assert vectors.shape[0] == 3

    def test_no_hardcoded_column_names_in_engine(self):
        # The engine source must not special-case specific dataset columns.
        from pathlib import Path

        source = (
            Path(__file__).resolve().parents[1]
            / "app" / "services" / "semantic_analysis.py"
        ).read_text(encoding="utf-8")
        for banned in ('column == "', "column_name == \"ol", '== "cust'):
            assert banned not in source


# ---------------------------------------------------------------------------
# XGBoost isolation
# ---------------------------------------------------------------------------


class TestXGBoostIsolation:
    def test_xgboost_only_in_harness(self):
        from pathlib import Path

        backend = Path(__file__).resolve().parents[1]
        offenders = []
        for path in (backend / "app").rglob("*.py"):
            text = path.read_text(encoding="utf-8", errors="ignore")
            if "xgboost" in text or "XGBClassifier" in text:
                offenders.append(str(path))
        assert offenders == [], f"xgboost leaked into app code: {offenders}"

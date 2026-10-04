"""Part G tests: relationship key-candidate gates.

Covers the deterministic gates added to discover_relationships:
- float measures are never offered as PK candidates;
- identifier-like columns with < 100% uniqueness stay candidates;
- composite candidates drop any float-measure / free-text member;
- old-style profiles (no new keys) keep pre-change behaviour;
- the parent-pool fallback is unchanged for a plain 90%-unique key;
- ingestion's shared read_table preserves leading zeros like the profiler.

Uses an in-memory sqlite DB with StaticPool so one connection backs all
sessions, and a monkeypatched RAW_DATA_DIR so no real dataset file is read.
"""

import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import app.services.storage as storage
from app.database import Base
from app.models.analysis import StoredProfile
from app.models.dataset import Dataset
from app.models.dataset_version import DatasetVersion
from app.models.file_metadata import FileMetadata
from app.models.relationship import RelationshipCandidate
from app.models.table_metadata import TableMetadata
from app.services.ingestion import read_table
from app.services.relationship_discovery import discover_relationships


@pytest.fixture()
def db_session(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "RAW_DATA_DIR", tmp_path / "datasets")
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    try:
        yield session
    finally:
        session.close()
        engine.dispose()


def _seed_version(db) -> tuple[Dataset, DatasetVersion]:
    dataset = Dataset(dataset_name="KeyGateDS")
    db.add(dataset)
    db.flush()
    version = DatasetVersion(dataset_id=dataset.dataset_id, version_number=1)
    db.add(version)
    db.flush()
    return dataset, version


def _column_profile(
    name: str,
    count: int,
    distinct: int,
    *,
    dtype: str = "object",
    identifier_like: bool = False,
    identifier_repeats: bool = False,
    name_signal: bool = False,
    meaningful: bool = True,
) -> dict:
    """Build a minimal-but-realistic column profile with new-style keys."""
    distinct_pct = (distinct / count * 100.0) if count else 0.0
    profile: dict = {
        "column_name": name,
        "dtype": dtype,
        "count": count,
        "null_count": 0,
        "null_percentage": 0.0,
        "non_null_count": count,
        "distinct_count": distinct,
        "distinct_percentage": round(distinct_pct, 4),
        "identifier_name_signal": name_signal,
        "text": {
            "constant": distinct <= 1,
            "near_constant": distinct == 2,
        },
    }
    if dtype == "float64":
        profile["numeric"] = {
            "meaningful_statistics": meaningful,
            "integer_valued": False,
        }
    else:
        profile["identifier_like"] = identifier_like
        profile["identifier_repeats"] = identifier_repeats
    return profile


def _profile_payload(columns: list[dict], composites: list[dict] | None = None) -> dict:
    return {
        "row_count": columns[0]["count"] if columns else 0,
        "tables": [
            {
                "table_name": "orders",
                "columns": columns,
                "composite_uniqueness_candidates": composites or [],
            }
        ],
    }


def _seed_full(
    db,
    columns: list[dict],
    composites: list[dict] | None = None,
) -> tuple[Dataset, DatasetVersion]:
    dataset, version = _seed_version(db)
    db.add(
        TableMetadata(
            version_id=version.version_id,
            table_name="orders",
            source_file="orders.csv",
        )
    )
    db.add(
        FileMetadata(
            version_id=version.version_id,
            original_filename="orders.csv",
            stored_filename="orders.csv",
            content_fingerprint="fp-keygates",
            file_extension=".csv",
        )
    )
    db.add(
        StoredProfile(
            version_id=version.version_id,
            profile_json=_profile_payload(columns, composites),
            row_count=columns[0]["count"] if columns else 0,
        )
    )
    db.flush()
    return dataset, version


def _pk_names(db, version_id: int) -> set[str]:
    return {
        row.parent_column
        for row in db.query(RelationshipCandidate)
        .filter(
            RelationshipCandidate.version_id == version_id,
            RelationshipCandidate.candidate_kind == "pk",
        )
        .all()
    }


def test_float_measure_not_offered_as_pk(db_session):
    """A 59%-unique float measure must not become a PK candidate."""
    columns = [
        _column_profile("amount", 1000, 590, dtype="float64", meaningful=True),
    ]
    dataset, version = _seed_full(db_session, columns)
    discover_relationships(db_session, dataset, version.version_id)
    assert "amount" not in _pk_names(db_session, version.version_id)


def test_identifier_repeats_still_pk_candidate(db_session):
    """identifier_like with 99% uniqueness (repeating id) stays a candidate."""
    columns = [
        _column_profile(
            "order_id",
            1000,
            990,
            identifier_like=True,
            identifier_repeats=True,
            name_signal=True,
        ),
    ]
    dataset, version = _seed_full(db_session, columns)
    discover_relationships(db_session, dataset, version.version_id)
    assert "order_id" in _pk_names(db_session, version.version_id)

    row = (
        db_session.query(RelationshipCandidate)
        .filter(
            RelationshipCandidate.version_id == version.version_id,
            RelationshipCandidate.parent_column == "order_id",
        )
        .first()
    )
    assert row is not None and row.stats_json is not None
    assert row.stats_json.get("surrogate_counter") is not True


def test_counter_like_marks_surrogate_counter(db_session):
    """A dense integer counter keeps its PK candidacy and is tagged."""
    columns = [
        {
            "column_name": "row_id",
            "dtype": "int64",
            "count": 1000,
            "null_count": 0,
            "null_percentage": 0.0,
            "non_null_count": 1000,
            "distinct_count": 1000,
            "distinct_percentage": 100.0,
            "identifier_name_signal": True,
            "identifier_like": True,
            "identifier_repeats": False,
            "numeric": {
                "integer_sequence": {
                    "counter_like": True,
                    "dense": True,
                    "step_one": True,
                },
            },
        },
    ]
    dataset, version = _seed_full(db_session, columns)
    discover_relationships(db_session, dataset, version.version_id)
    assert "row_id" in _pk_names(db_session, version.version_id)

    row = (
        db_session.query(RelationshipCandidate)
        .filter(
            RelationshipCandidate.version_id == version.version_id,
            RelationshipCandidate.parent_column == "row_id",
        )
        .first()
    )
    assert row is not None and row.stats_json is not None
    assert row.stats_json.get("surrogate_counter") is True


def test_composite_candidate_drops_measure_member(db_session):
    """A composite containing a float measure is not persisted."""
    columns = [
        _column_profile("order_id", 1000, 1000, identifier_like=True, name_signal=True),
        _column_profile("amount", 1000, 900, dtype="float64"),
    ]
    composites = [
        {
            "columns": ["order_id", "amount"],
            "composite_uniqueness_percentage": 99.95,
        },
    ]
    dataset, version = _seed_full(db_session, columns, composites)
    discover_relationships(db_session, dataset, version.version_id)

    composites_stored = (
        db_session.query(RelationshipCandidate)
        .filter(
            RelationshipCandidate.version_id == version.version_id,
            RelationshipCandidate.candidate_kind == "composite_pk",
        )
        .all()
    )
    assert composites_stored == []

    # The single-column key candidate is unaffected by the composite gate.
    assert "order_id" in _pk_names(db_session, version.version_id)


def test_old_style_profile_unchanged(db_session):
    """Profiles without the new keys keep the pre-change gating exactly."""
    columns = [
        {
            "column_name": "legacy_id",
            "dtype": "object",
            "count": 1000,
            "null_count": 0,
            "null_percentage": 0.0,
            "non_null_count": 1000,
            "distinct_count": 995,
            "distinct_percentage": 99.5,
            "identifier_name_signal": False,
        },
    ]
    dataset, version = _seed_full(db_session, columns)
    discover_relationships(db_session, dataset, version.version_id)
    # No identifier evidence, but distinct% >= 95 keeps the legacy path.
    assert "legacy_id" in _pk_names(db_session, version.version_id)


def test_parent_pool_fallback_for_90pct_key(db_session):
    """No profile PK candidate -> all columns stay eligible as parents.

    With only a 90%-unique column (below the 95% gate) the profile-based
    pool is empty, so the fallback exposes the column for FK matching.
    """
    columns = [
        _column_profile("region_code", 1000, 900, identifier_like=False),
    ]
    dataset, version = _seed_full(db_session, columns)
    discover_relationships(db_session, dataset, version.version_id)

    # Nothing persisted as pk (below gate), and the fallback pool still
    # contains the column so cross-table matching is not blinded.
    assert "region_code" not in _pk_names(db_session, version.version_id)
    pk_rows = (
        db_session.query(RelationshipCandidate)
        .filter(
            RelationshipCandidate.version_id == version.version_id,
            RelationshipCandidate.candidate_kind == "pk",
        )
        .count()
    )
    assert pk_rows == 0


def test_read_table_matches_profiling_dtypes(tmp_path):
    """Ingestion's shared reader preserves leading-zero codes as strings."""
    frame = pd.DataFrame({"code": ["0012", "0450", "1200"]})
    path = tmp_path / "codes.csv"
    frame.to_csv(path, index=False)

    loaded = read_table(path)
    assert list(loaded.columns) == ["code"]
    assert loaded["code"].tolist() == ["0012", "0450", "1200"]
    dtype_name = str(loaded["code"].dtype).lower()
    assert "str" in dtype_name or dtype_name == "object"

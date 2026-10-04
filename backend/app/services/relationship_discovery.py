"""Relationship discovery (deterministic PK/FK candidate evidence).

Relationship discovery is NOT referential integrity validation. It produces
RANKED CANDIDATES with per-feature deterministic evidence; only human-approved
candidates may later become authoritative RI rules.

Core principles:
- Every feature is evidence, never proof. The score is a documented weighted
  mean of APPLICABLE evidence (same convention as semantic scoring).
- LOW CONTAINMENT DOES NOT ELIMINATE a candidate: containment measures how
  much of the child data matches the parent, not whether the relationship
  exists. Low containment on otherwise-strong evidence is reported as a
  candidate relationship WITH LIKELY ORPHANS (potential RI violations) - the
  exact distinction between discovery and validation.
- PK candidates: identifier-like, high-uniqueness, low-null columns. PK
  status is evidence-based, never assumed from the column name alone.
- Human decisions (approve/reject/edit/missed/manual) go to the Feedback
  Store and update candidate status; they do not silently rewrite the KB
  or auto-create rules.
"""

import json

import pandas as pd
from sqlalchemy.orm import Session

from app.config import (
    RELATIONSHIP_COMPOSITE_UNIQUENESS,
    RELATIONSHIP_MAX_PAIRS,
    RELATIONSHIP_MIN_CANDIDATE_SCORE,
    RELATIONSHIP_MIN_NAME_SIMILARITY,
    RELATIONSHIP_MIN_NAME_SUPPORT,
    RELATIONSHIP_MIN_PARENT_UNIQUENESS,
    RELATIONSHIP_SEMANTIC_NEUTRAL,
    RELATIONSHIP_SCORING_WEIGHTS,
    RELATIONSHIP_TOP_PER_CHILD,
)
from app.models import (
    ColumnMetadata,
    Dataset,
    DatasetVersion,
    RelationshipCandidate,
    RelationshipFeedback,
    SemanticPrediction,
    StoredProfile,
    TableMetadata,
)
from app.services.semantic_features import text_similarity
from app.services import storage as storage_module

# Datatype families considered compatible for a join.
_COMPATIBLE_TYPE_FAMILIES = {
    frozenset({"int", "bigint", "smallint", "float", "double", "decimal", "numeric"}),
    frozenset({"object", "string", "str", "text"}),
    frozenset({"datetime64", "datetime", "date", "timestamp"}),
    frozenset({"bool", "boolean"}),
}


def _type_family(data_type: str | None) -> str | None:
    if not data_type:
        return None

    value = data_type.lower()

    if any(token in value for token in ("int", "float", "double", "decimal", "numeric", "real")):
        return "numeric"

    if any(token in value for token in ("datetime", "timestamp")):
        return "temporal"

    if value == "date":
        return "temporal"

    if any(token in value for token in ("bool",)):
        return "boolean"

    if any(token in value for token in ("object", "string", "str", "text")):
        return "string"

    return value


def _family_compatible(first: str | None, second: str | None) -> bool | None:
    """Datatype family compatibility; None when either type is unknown."""
    if not first or not second:
        return None

    if first == second:
        return True

    return first in {"numeric", "string", "temporal", "boolean"} and second in {
        "numeric",
        "string",
        "temporal",
        "boolean",
    }


def _table_dataframe(
    db: Session,
    dataset_id: int,
    version: DatasetVersion,
    table: TableMetadata,
) -> pd.DataFrame | None:
    """Load a table's raw dataframe (bounded by what ingestion stored)."""
    stored_filename = _stored_filename_for_table(db, version.version_id, table)

    if stored_filename is None:
        return None

    path = (
        storage_module.RAW_DATA_DIR
        / str(dataset_id)
        / f"v{version.version_number}"
        / stored_filename
    )

    if path.exists():
        # Shared with ingestion/profiling so every stage sees identical
        # dtypes for the same file (including leading-zero-safe codes).
        from app.services.ingestion import read_table

        return read_table(path)

    return None


def _stored_filename_for_table(
    db: Session,
    version_id: int,
    table: TableMetadata,
) -> str | None:
    from app.models import FileMetadata

    file_metadata = (
        db.query(FileMetadata)
        .filter(FileMetadata.version_id == version_id)
        .all()
    )

    for metadata in file_metadata:
        original = metadata.original_filename or ""

        # Folder uploads keep a relative path in original_filename while the
        # table stores the bare filename; match on the path tail too.
        if original == table.source_file or original.replace("\\", "/").endswith(
            "/" + table.source_file
        ):
            return metadata.stored_filename

    return None


def _is_float_measure_profile(profile: dict) -> bool:
    """True when a profile describes a float measure (fractional values).

    Reads ONLY the new Part D profiling keys; old stored profiles without
    them yield False (behaviour unchanged).
    """
    numeric = profile.get("numeric")
    if not isinstance(numeric, dict):
        return False
    return bool(numeric.get("integer_valued") is False)


def _is_free_text_profile(profile: dict) -> bool:
    """True when a profile describes free text (mean_length > 64).

    Old stored profiles without text.mean_length yield False.
    """
    text = profile.get("text")
    if not isinstance(text, dict):
        return False
    mean_length = text.get("mean_length")
    return isinstance(mean_length, (int, float)) and mean_length > 64


def _is_counter_like_profile(profile: dict) -> bool:
    """True when numeric.integer_sequence.counter_like is set (new profiles)."""
    numeric = profile.get("numeric")
    if not isinstance(numeric, dict):
        return False
    sequence = numeric.get("integer_sequence")
    return isinstance(sequence, dict) and bool(sequence.get("counter_like"))


def _parent_pk_evidence(
    profile: dict,
) -> tuple[float, bool, dict]:
    """PK evidence for a parent column from its profile.

    Returns (score, applicable, stats). Identifier-like + complete +
    (near-)unique columns are strong PK candidates. Uniqueness on very small
    samples is weaker evidence.
    """
    distinct_percentage = profile.get("distinct_percentage")
    null_percentage = profile.get("null_percentage")
    non_null_count = profile.get("non_null_count")
    identifier_name_signal = bool(profile.get("identifier_name_signal", False))
    identifier_signal = bool(profile.get("identifier_signal", False))

    stats = {
        "distinct_percentage": distinct_percentage,
        "null_percentage": null_percentage,
        "non_null_count": non_null_count,
        "identifier_name_signal": identifier_name_signal,
        "identifier_signal": identifier_signal,
    }

    if distinct_percentage is None or null_percentage is None:
        return 0.0, False, stats

    if non_null_count is None:
        non_null_count = profile.get("row_count", 0) - (profile.get("null_count") or 0)

    uniqueness = float(distinct_percentage) / 100.0
    completeness = 1.0 - (float(null_percentage) / 100.0)

    # Small-sample dampening: uniqueness over few rows is weak evidence.
    sample_factor = min(1.0, max(0.3, non_null_count / 10.0))

    score = (
        0.55 * uniqueness + 0.25 * completeness + 0.20 * float(identifier_name_signal)
    ) * sample_factor

    if identifier_signal:
        score = min(1.0, score + 0.1)

    return round(score, 4), True, stats


def _evaluate_pair(
    parent_df: pd.DataFrame,
    child_df: pd.DataFrame,
    parent_column: str,
    child_column: str,
    parent_profile: dict | None,
    child_profile: dict | None,
    semantic_types: dict[tuple[str, str], str] | None = None,
    parent_table: str | None = None,
    child_table: str | None = None,
) -> dict | None:
    """Compute the full evidence bundle for one (parent, child) pair.

    semantic_types maps (table_name, column_name) -> FINAL semantic type
    (user-approved/edited when present, otherwise the system
    recommendation). When both sides carry a final type, semantic
    compatibility becomes first-class relationship evidence: equal types are
    strong positive evidence, conflicting types are evidence AGAINST —
    name similarity alone must not drive a relationship.
    """
    evidence: dict[str, dict] = {}
    stats: dict = {}

    parent_series = parent_df[parent_column]
    child_series = child_df[child_column]

    # Normalize values for set comparison: numeric values must compare
    # across dtypes (int 1 vs float 1.0 vs string "1"), so numbers are
    # canonicalized before stringification.
    def _normalize_values(series: pd.Series) -> set[str]:
        values: set[str] = set()

        for value in series.dropna():
            if isinstance(value, bool):
                values.add("1" if value else "0")
            elif isinstance(value, (int, float)):
                number = float(value)

                values.add(
                    str(int(number)) if number.is_integer() else repr(number)
                )
            else:
                values.add(str(value).strip())

        return values

    parent_values = _normalize_values(parent_series)
    child_values = _normalize_values(child_series)

    # ------------------------------------------------------------------
    # Name similarity (child column vs parent column and table names).
    # ------------------------------------------------------------------
    name_sim = text_similarity(child_column, parent_column)
    evidence["name_similarity"] = {"value": round(name_sim, 4), "applicable": True}

    # ------------------------------------------------------------------
    # Datatype compatibility (profile datatypes; computable = applicable).
    # ------------------------------------------------------------------
    parent_dtype = (parent_profile or {}).get("data_type")
    child_dtype = (child_profile or {}).get("data_type")

    parent_family = _type_family(parent_dtype)
    child_family = _type_family(child_dtype)
    compatible = _family_compatible(parent_family, child_family)

    if compatible is None:
        evidence["datatype_compatibility"] = {"value": 0.0, "applicable": False}
    else:
        evidence["datatype_compatibility"] = {
            "value": 1.0 if compatible else 0.0,
            "applicable": True,
        }

    # ------------------------------------------------------------------
    # Semantic compatibility from FINAL semantic types (user decisions
    # preferred). Not applicable (neutral) when either side has no type.
    # ------------------------------------------------------------------
    parent_semantic = (semantic_types or {}).get(
        ((parent_table or "").strip().lower(), (parent_column or "").strip().lower())
    )
    child_semantic = (semantic_types or {}).get(
        ((child_table or "").strip().lower(), (child_column or "").strip().lower())
    )

    if parent_semantic and child_semantic:
        if parent_semantic.strip().lower() == child_semantic.strip().lower():
            semantic_value = 1.0
        else:
            semantic_value = 0.0
        stats["semantic_types"] = {
            "parent": parent_semantic,
            "child": child_semantic,
            "match": semantic_value == 1.0,
        }
        evidence["semantic_compatibility"] = {
            "value": semantic_value,
            "applicable": True,
        }
    else:
        # No semantic mapping on at least one side: neutral, non-applicable
        # evidence so columns without Stage 04 decisions are not penalized
        # and semantic absence is not treated as semantic conflict.
        evidence["semantic_compatibility"] = {
            "value": RELATIONSHIP_SEMANTIC_NEUTRAL,
            "applicable": False,
        }
        if parent_semantic or child_semantic:
            stats["semantic_types"] = {
                "parent": parent_semantic,
                "child": child_semantic,
                "match": None,
            }

    # ------------------------------------------------------------------
    # Parent uniqueness (PK-side evidence).
    # ------------------------------------------------------------------
    if parent_profile is not None:
        pk_score, pk_applicable, pk_stats = _parent_pk_evidence(parent_profile)
        evidence["parent_uniqueness"] = {
            "value": pk_score,
            "applicable": pk_applicable,
        }
        stats["parent_pk"] = pk_stats
    else:
        distinct = parent_series.nunique(dropna=True)
        non_null = int(parent_series.notna().sum())
        uniqueness = distinct / non_null if non_null else 0.0
        evidence["parent_uniqueness"] = {"value": round(uniqueness, 4), "applicable": True}
        stats["parent_pk"] = {"uniqueness_from_data": round(uniqueness, 4)}

    # ------------------------------------------------------------------
    # Value containment + overlap (child -> parent direction).
    # ------------------------------------------------------------------
    if child_values:
        contained = sum(1 for value in child_values if value in parent_values)
        containment = contained / len(child_values)
    else:
        containment = 0.0

    if parent_values and child_values:
        overlap = len(parent_values & child_values) / min(
            len(parent_values), len(child_values)
        )
    else:
        overlap = 0.0

    stats["child_distinct"] = len(child_values)
    stats["parent_distinct"] = len(parent_values)
    stats["contained_distinct"] = (
        sum(1 for value in child_values if value in parent_values)
        if child_values
        else 0
    )
    stats["containment"] = round(containment, 4)
    stats["overlap"] = round(overlap, 4)

    # Dense-integer-range dampening. Two unrelated dense sequences of small
    # integers (1..N on both sides) trivially "contain" each other, so raw
    # containment/overlap overstate the evidence. When BOTH sides cover
    # their dense 1..max integer range almost completely, the value evidence
    # is dampened (not zeroed) toward a weak neutral signal. Sparse or
    # string keys are never affected. This fights the classic false positive
    # "ProductID contains CustomerID" without any pair-specific rules.
    both_dense = False

    try:
        parent_numeric = sorted(float(v) for v in parent_values)
        child_numeric = sorted(float(v) for v in child_values)

        def _dense(values: list[float]) -> bool:
            if not values:
                return False

            lo, hi = values[0], values[-1]

            return lo >= 0 and hi <= 100000 and hi - lo + 1 > 0 and (
                len(set(values)) / (hi - lo + 1)
            ) >= 0.95

        both_dense = _dense(parent_numeric) and _dense(child_numeric)
    except (TypeError, ValueError):
        both_dense = False

    if both_dense:
        containment = 0.5 * containment
        overlap = 0.5 * overlap
        stats["dense_integer_warning"] = (
            "Both columns are dense integer sequences; containment/overlap "
            "evidence dampened (any two dense 1..N ranges trivially overlap)."
        )

    evidence["value_containment"] = {"value": round(containment, 4), "applicable": True}
    evidence["value_overlap"] = {"value": round(overlap, 4), "applicable": True}

    # Orphan accounting: containment below 100% means unmatched child values.
    orphan_count = len(child_values) - (
        sum(1 for value in child_values if value in parent_values)
        if child_values
        else 0
    )
    orphan_rate = orphan_count / len(child_values) if child_values else 0.0
    stats["orphan_distinct_count"] = orphan_count
    stats["orphan_distinct_rate"] = round(orphan_rate, 4)

    # LOW CONTAINMENT IS NOT DISQUALIFYING: it lowers the containment
    # evidence value (bounded below so strong lexical/structural matches
    # with orphans remain visible) but never zeroes out the candidate. The
    # orphan statistics are surfaced separately for RI validation later.
    evidence["value_containment"] = {"value": round(containment, 4), "applicable": True}
    evidence["value_overlap"] = {"value": round(overlap, 4), "applicable": True}

    # ------------------------------------------------------------------
    # Cardinality compatibility: FK side should not exceed PK side.
    # ------------------------------------------------------------------
    parent_distinct = stats["parent_distinct"]
    child_distinct = stats["child_distinct"]

    if parent_distinct and child_distinct:
        ratio = child_distinct / parent_distinct
        cardinality = 1.0 if ratio <= 1.0 else max(0.0, 1.0 - (ratio - 1.0))
    else:
        cardinality = None

    if cardinality is None:
        evidence["cardinality_compatibility"] = {"value": 0.0, "applicable": False}
    else:
        evidence["cardinality_compatibility"] = {
            "value": round(cardinality, 4),
            "applicable": True,
        }

    # ------------------------------------------------------------------
    # Child completeness: FK columns are typically mostly non-null.
    # ------------------------------------------------------------------
    child_null_percentage = (child_profile or {}).get("null_percentage")

    if child_null_percentage is None:
        null_share = child_series.isna().mean()
        child_null_percentage = float(null_share) * 100 if pd.notna(null_share) else None

    if child_null_percentage is not None:
        completeness = 1.0 - (float(child_null_percentage) / 100.0)
        evidence["child_completeness"] = {
            "value": round(max(0.0, completeness), 4),
            "applicable": True,
        }
    else:
        evidence["child_completeness"] = {"value": 0.0, "applicable": False}

    # ------------------------------------------------------------------
    # Structural evidence: both sides look identifier-like.
    # ------------------------------------------------------------------
    parent_identifier = bool((parent_profile or {}).get("identifier_name_signal", False))
    child_identifier = bool((child_profile or {}).get("identifier_name_signal", False))
    structural = (
        0.5 * float(parent_identifier) + 0.5 * float(child_identifier)
    )
    evidence["structural_evidence"] = {"value": round(structural, 4), "applicable": True}

    stats["evidence"] = evidence

    # Deterministic weighted mean over applicable evidence.
    numerator = 0.0
    denominator = 0.0

    for feature, weight in RELATIONSHIP_SCORING_WEIGHTS.items():
        item = evidence.get(feature)

        if item is None or not item.get("applicable", False):
            continue

        numerator += weight * float(item.get("value", 0.0))
        denominator += weight

    score = round(numerator / denominator, 4) if denominator > 0 else 0.0

    return {"score": score, "evidence": evidence, "stats": stats}


def _iter_table_profiles(profile_result: dict):
    """Yield (table_name, {column_name: profile}) from a stored profile."""
    for table_profile in profile_result.get("tables", []):
        table_name = table_profile.get("table_name")

        if not table_name:
            continue

        columns = {
            column.get("column_name"): column
            for column in table_profile.get("columns", [])
            if column.get("column_name")
        }

        yield table_name, columns


def discover_relationships(
    db: Session,
    dataset: Dataset,
    version_id: int,
) -> dict:
    """Discover and persist ranked relationship candidates for a version.

    Existing pending candidates are replaced; approved/rejected/missed/manual
    decisions are kept.
    """
    version = (
        db.query(DatasetVersion)
        .filter(
            DatasetVersion.dataset_id == dataset.dataset_id,
            DatasetVersion.version_id == version_id,
        )
        .first()
    )

    if version is None:
        raise ValueError(f"Version {version_id} not found for this dataset.")

    stored = (
        db.query(StoredProfile)
        .filter(StoredProfile.version_id == version.version_id)
        .first()
    )

    if stored is None:
        raise ValueError("Profiling must run before relationship discovery.")

    profile_result = stored.profile_json

    tables = (
        db.query(TableMetadata)
        .filter(TableMetadata.version_id == version.version_id)
        .all()
    )

    table_by_name = {table.table_name.strip().lower(): table for table in tables}

    # Candidate PK columns per table: profile identifier evidence.
    profiles: dict[str, dict[str, dict]] = dict(_iter_table_profiles(profile_result))

    # Keep pending decisions across re-runs.
    existing = (
        db.query(RelationshipCandidate)
        .filter(RelationshipCandidate.version_id == version.version_id)
        .all()
    )

    kept: dict[tuple[str, str, str, str, str], RelationshipCandidate] = {}
    for candidate in existing:
        if candidate.status in {"approved", "rejected", "missed", "manual", "edited"}:
            # Rows predating candidate_kind carry NULL: normalize to the
            # relationship kind so legacy user decisions keep their identity
            # and are never duplicated by a re-run.
            kind = candidate.candidate_kind or "relationship"
            key = (
                kind,
                candidate.parent_table,
                candidate.parent_column,
                candidate.child_table,
                candidate.child_column,
            )
            kept[key] = candidate
        else:
            db.delete(candidate)

    db.flush()

    # Final semantic types from Stage 04: user-defined/confirmed decisions
    # take priority over raw system recommendations. Used for semantic
    # compatibility evidence — never as proof by itself.
    semantic_types: dict[tuple[str, str], str] = {}

    for prediction in (
        db.query(SemanticPrediction)
        .filter(SemanticPrediction.version_id == version.version_id)
        .all()
    ):
        final_type = (
            prediction.user_defined_type
            or (
                prediction.predicted_concept
                if prediction.user_confirmed_concept_id
                else None
            )
            or prediction.predicted_concept
        )

        if final_type:
            semantic_types[
                (
                    prediction.table_name.strip().lower(),
                    prediction.column_name.strip().lower(),
                )
            ] = final_type

    # Load dataframes once per table.
    dataframes: dict[str, pd.DataFrame] = {}
    for table_name_lower, table in table_by_name.items():
        dataframe = _table_dataframe(
            db, dataset.dataset_id, version, table
        )
        if dataframe is not None:
            dataframes[table_name_lower] = dataframe

    # Enumerate PK candidate columns (parent side).
    pk_candidates: dict[str, list[str]] = {}

    for table_name, columns in profiles.items():
        pk_columns = []

        for column_name, profile in columns.items():
            pk_score, applicable, _ = _parent_pk_evidence(profile)

            if applicable and pk_score >= 0.55:
                pk_columns.append(column_name)

        if pk_columns:
            pk_candidates[table_name] = pk_columns

    # Fallback: when no profile-based PK candidates exist, allow all columns
    # of each table as potential parents (evidence still gates the score).
    parent_pool: dict[str, list[str]] = pk_candidates or {
        table_name: list(columns.keys()) for table_name, columns in profiles.items()
    }

    # ------------------------------------------------------------------
    # KEY CANDIDATES (single- and multi-table). Persisted as candidates of
    # kind "pk" / "composite_pk" so single-file datasets surface key
    # evidence too. Derived from profiling evidence only; never claimed as
    # confirmed constraints.
    # ------------------------------------------------------------------
    key_results: list[RelationshipCandidate] = []

    for table_name, columns in profiles.items():
        table_meta = table_by_name.get(table_name.strip().lower())

        if table_meta is None:
            continue

        for column_name, profile in columns.items():
            key = ("pk", table_meta.table_name, column_name, "", "")
            existing_key = kept.get(key)

            if existing_key is not None:
                key_results.append(existing_key)
                continue

            pk_score, applicable, pk_stats = _parent_pk_evidence(profile)

            if not applicable or pk_score < 0.55:
                continue

            # Key-candidate gates (deterministic evidence, not verdicts):
            # a PK candidate must be at least this distinct and must not be
            # a float measure. Profiles lacking the new keys behave exactly
            # as before (.get -> None -> gates pass).
            distinct_percentage = profile.get("distinct_percentage")
            if (
                distinct_percentage is not None
                and float(distinct_percentage) < RELATIONSHIP_MIN_PARENT_UNIQUENESS
            ):
                continue

            if _is_float_measure_profile(profile):
                continue

            pk_stats = dict(pk_stats)
            if _is_counter_like_profile(profile):
                pk_stats["surrogate_counter"] = True

            candidate = RelationshipCandidate(
                dataset_id=dataset.dataset_id,
                version_id=version.version_id,
                parent_table=table_meta.table_name,
                parent_column=column_name,
                child_table="",
                child_column="",
                parent_role="pk_candidate",
                candidate_kind="pk",
                score=pk_score,
                evidence_json={"parent_uniqueness": {"value": pk_score, "applicable": True}},
                stats_json=pk_stats,
                status="pending",
            )
            db.add(candidate)
            key_results.append(candidate)

        # Composite key candidates from table-level profiling evidence.
        table_profile = next(
            (
                tp
                for tp in profile_result.get("tables", [])
                if tp.get("table_name") == table_name
            ),
            {},
        )

        for composite in table_profile.get("composite_uniqueness_candidates", []):
            combo = composite.get("columns", [])

            if len(combo) < 2:
                continue

            # Composite key gate: no member may be a float measure or free
            # text (new profile keys; old profiles pass unchanged).
            columns_by_name = {
                column_profile.get("column_name"): column_profile
                for column_profile in table_profile.get("columns", [])
            }

            if any(
                _is_float_measure_profile(columns_by_name.get(member, {}))
                or _is_free_text_profile(columns_by_name.get(member, {}))
                for member in combo
            ):
                continue

            uniqueness = float(composite.get("composite_uniqueness_percentage", 0.0))

            if uniqueness < RELATIONSHIP_COMPOSITE_UNIQUENESS:
                continue

            combo_label = " + ".join(combo)
            key = (
                "composite_pk",
                table_meta.table_name,
                combo_label,
                "",
                "",
            )
            existing_key = kept.get(key)

            if existing_key is not None:
                key_results.append(existing_key)
                continue

            candidate = RelationshipCandidate(
                dataset_id=dataset.dataset_id,
                version_id=version.version_id,
                parent_table=table_meta.table_name,
                parent_column=combo_label,
                child_table="",
                child_column="",
                parent_role="composite_pk_candidate",
                candidate_kind="composite_pk",
                score=round(uniqueness / 100.0, 4),
                evidence_json={
                    "composite_uniqueness": {
                        "value": round(uniqueness / 100.0, 4),
                        "applicable": True,
                    }
                },
                stats_json=dict(composite),
                status="pending",
            )
            db.add(candidate)
            key_results.append(candidate)

    db.flush()

    # Enumerate (parent, child) pairs and evaluate.
    evaluated = 0
    results = []

    for child_table_name, child_columns in profiles.items():
        child_table_meta = table_by_name.get(child_table_name.strip().lower())
        child_df = dataframes.get(child_table_name.strip().lower())

        if child_df is None or child_table_meta is None:
            continue

        for parent_table_name, parent_columns in profiles.items():
            if parent_table_name == child_table_name:
                # Self-references are valid but skipped in this iteration
                # bound; they can be added manually.
                continue

            parent_table_meta = table_by_name.get(parent_table_name.strip().lower())
            parent_df = dataframes.get(parent_table_name.strip().lower())

            if parent_df is None or parent_table_meta is None:
                continue

            for parent_column in parent_pool.get(parent_table_name, []):
                if parent_column not in parent_df.columns:
                    continue

                parent_profile = parent_columns.get(parent_column)

                for child_column, child_profile in child_columns.items():
                    if child_column not in child_df.columns:
                        continue

                    if evaluated >= RELATIONSHIP_MAX_PAIRS:
                        break

                    # Lexical prefilter unless the parent is a strong PK.
                    name_sim = text_similarity(child_column, parent_column)
                    parent_pk_profile = (
                        profiles.get(parent_table_name, {}).get(parent_column, {})
                    )
                    pk_score, pk_applicable, _ = _parent_pk_evidence(parent_pk_profile)

                    if (
                        name_sim < RELATIONSHIP_MIN_NAME_SIMILARITY
                        and not (pk_applicable and pk_score >= 0.75)
                    ):
                        continue

                    evaluated += 1

                    outcome = _evaluate_pair(
                        parent_df,
                        child_df,
                        parent_column,
                        child_column,
                        parent_profile,
                        child_profile,
                        semantic_types=semantic_types,
                        parent_table=parent_table_meta.table_name,
                        child_table=child_table_meta.table_name,
                    )

                    if outcome is None:
                        continue

                    if outcome["score"] < RELATIONSHIP_MIN_CANDIDATE_SCORE:
                        continue

                    # Affirmative-evidence floor: with NO value support at all
                    # (zero containment and zero overlap), the pair needs
                    # clear name agreement to stay a candidate. Identical
                    # identifier-style names with disjoint values (e.g.
                    # CustomerID vs CategoryID) are coincidence, not orphan
                    # evidence; a genuinely renamed FK with matching values
                    # never reaches this gate.
                    evidence = outcome["evidence"]
                    name_value = float(
                        evidence.get("name_similarity", {}).get("value", 0.0)
                    )
                    containment_value = float(
                        evidence.get("value_containment", {}).get("value", 0.0)
                    )
                    overlap_value = float(
                        evidence.get("value_overlap", {}).get("value", 0.0)
                    )

                    if (
                        containment_value == 0.0
                        and overlap_value == 0.0
                        and name_value < RELATIONSHIP_MIN_NAME_SUPPORT
                    ):
                        continue

                    key = (
                        "relationship",
                        parent_table_meta.table_name,
                        parent_column,
                        child_table_meta.table_name,
                        child_column,
                    )

                    existing_candidate = kept.get(key)

                    if existing_candidate is not None:
                        existing_candidate.score = outcome["score"]
                        existing_candidate.evidence_json = outcome["evidence"]
                        existing_candidate.stats_json = outcome["stats"]
                        results.append(existing_candidate)
                        continue

                    candidate = RelationshipCandidate(
                        dataset_id=dataset.dataset_id,
                        version_id=version.version_id,
                        parent_table=parent_table_meta.table_name,
                        parent_column=parent_column,
                        child_table=child_table_meta.table_name,
                        child_column=child_column,
                        parent_role="pk_candidate",
                        candidate_kind="relationship",
                        score=outcome["score"],
                        evidence_json=outcome["evidence"],
                        stats_json=outcome["stats"],
                        status="pending",
                    )

                    db.add(candidate)
                    db.flush()
                    results.append(candidate)

        # Keep at most the top candidates per child column.
        by_child: dict[str, list[RelationshipCandidate]] = {}
        for candidate in results:
            if candidate.child_table != child_table_meta.table_name:
                continue
            by_child.setdefault(candidate.child_column, []).append(candidate)

        for child_column, candidates in by_child.items():
            candidates.sort(key=lambda item: item.score, reverse=True)

            for extra in candidates[RELATIONSHIP_TOP_PER_CHILD:]:
                if extra.status == "pending":
                    db.delete(extra)
                    results.remove(extra)

    db.flush()

    all_results = key_results + results

    return {
        "dataset_id": dataset.dataset_id,
        "version_id": version.version_id,
        "evaluated_pairs": evaluated,
        "candidate_count": len(
            [item for item in all_results if item.status == "pending"]
        ),
        "candidates": [_candidate_response(item) for item in all_results],
    }


def _candidate_response(candidate: RelationshipCandidate) -> dict:
    stats = candidate.stats_json or {}

    return {
        "relationship_id": candidate.relationship_id,
        "candidate_kind": candidate.candidate_kind,
        "parent_table": candidate.parent_table,
        "parent_column": candidate.parent_column,
        "child_table": candidate.child_table,
        "child_column": candidate.child_column,
        "parent_role": candidate.parent_role,
        "score": candidate.score,
        "status": candidate.status,
        "human_note": candidate.human_note,
        "evidence": candidate.evidence_json,
        "containment": stats.get("containment"),
        "orphan_distinct_count": stats.get("orphan_distinct_count"),
        "orphan_distinct_rate": stats.get("orphan_distinct_rate"),
        "stats": stats,
    }


def apply_relationship_decision(
    db: Session,
    dataset: Dataset,
    version_id: int,
    relationship_id: int,
    decision: str,
    note: str | None = None,
) -> RelationshipCandidate:
    """Record a human relationship decision (feedback store + status)."""
    candidate = (
        db.query(RelationshipCandidate)
        .filter(
            RelationshipCandidate.relationship_id == relationship_id,
            RelationshipCandidate.version_id == version_id,
        )
        .first()
    )

    if candidate is None:
        raise ValueError(
            f"Relationship candidate {relationship_id} not found for this version."
        )

    if decision not in {"approved", "rejected", "missed", "edited"}:
        raise ValueError(f"Unsupported relationship decision: {decision}")

    candidate.status = decision
    candidate.human_note = note

    db.add(
        RelationshipFeedback(
            dataset_id=dataset.dataset_id,
            version_id=version_id,
            relationship_id=candidate.relationship_id,
            parent_table=candidate.parent_table,
            parent_column=candidate.parent_column,
            child_table=candidate.child_table,
            child_column=candidate.child_column,
            decision=decision,
            evidence_json=candidate.evidence_json or {},
        )
    )

    db.flush()
    return candidate


def add_manual_relationship(
    db: Session,
    dataset: Dataset,
    version_id: int,
    parent_table: str,
    parent_column: str,
    child_table: str,
    child_column: str,
    note: str | None = None,
) -> RelationshipCandidate:
    """Add a human-specified relationship (marked manual, still evidence-
    checked for transparency but never silently discarded)."""
    candidate = RelationshipCandidate(
        dataset_id=dataset.dataset_id,
        version_id=version_id,
        parent_table=parent_table,
        parent_column=parent_column,
        child_table=child_table,
        child_column=child_column,
        parent_role="pk_candidate",
        candidate_kind="relationship",
        score=1.0,
        evidence_json={"source": "manual", "note": note},
        stats_json={},
        status="manual",
        human_note=note,
    )

    db.add(candidate)
    db.flush()

    db.add(
        RelationshipFeedback(
            dataset_id=dataset.dataset_id,
            version_id=version_id,
            relationship_id=candidate.relationship_id,
            parent_table=parent_table,
            parent_column=parent_column,
            child_table=child_table,
            child_column=child_column,
            decision="manual_add",
            evidence_json=candidate.evidence_json or {},
        )
    )

    db.flush()
    return candidate


def add_manual_key_candidate(
    db: Session,
    dataset: Dataset,
    version_id: int,
    table_name: str,
    key_columns: list[str],
    note: str | None = None,
) -> RelationshipCandidate:
    """Add a user-declared PK / composite-PK candidate for one table.

    Supports single-file datasets where no cross-table discovery is
    possible. Each entry is stored separately (N entries allowed); existing
    candidates are never replaced.
    """
    normalized = [column.strip() for column in key_columns if column.strip()]

    if not normalized:
        raise ValueError("A key candidate needs at least one column.")

    kind = "pk" if len(normalized) == 1 else "composite_pk"
    label = " + ".join(normalized)

    candidate = RelationshipCandidate(
        dataset_id=dataset.dataset_id,
        version_id=version_id,
        parent_table=table_name.strip(),
        parent_column=label,
        child_table="",
        child_column="",
        parent_role="pk_candidate" if kind == "pk" else "composite_pk_candidate",
        candidate_kind=kind,
        score=1.0,
        evidence_json={"source": "manual", "note": note},
        stats_json={},
        status="manual",
        human_note=note,
    )

    db.add(candidate)
    db.flush()

    db.add(
        RelationshipFeedback(
            dataset_id=dataset.dataset_id,
            version_id=version_id,
            relationship_id=candidate.relationship_id,
            parent_table=candidate.parent_table,
            parent_column=candidate.parent_column,
            child_table="",
            child_column="",
            decision="manual_add",
            evidence_json={
                "source": "manual",
                "candidate_kind": kind,
                "key_columns": normalized,
            },
        )
    )

    db.flush()
    return candidate


def edit_relationship_candidate(
    db: Session,
    dataset: Dataset,
    version_id: int,
    relationship_id: int,
    parent_table: str,
    parent_column: str,
    child_table: str,
    child_column: str,
    note: str | None = None,
) -> RelationshipCandidate:
    """Apply a user correction to a recommended relationship.

    The corrected values REPLACE the displayed parent/child columns while
    the original recommendation stays available in the feedback history
    (decision="edited" records the pre-edit state). Status becomes
    "edited": the corrected relationship is the user's final decision and
    is never silently overwritten by a later discovery run (kept map).
    """
    candidate = (
        db.query(RelationshipCandidate)
        .filter(
            RelationshipCandidate.relationship_id == relationship_id,
            RelationshipCandidate.version_id == version_id,
        )
        .first()
    )

    if candidate is None:
        raise ValueError(
            f"Relationship candidate {relationship_id} not found for this version."
        )

    original = {
        "parent_table": candidate.parent_table,
        "parent_column": candidate.parent_column,
        "child_table": candidate.child_table,
        "child_column": candidate.child_column,
        "score": candidate.score,
        "evidence": candidate.evidence_json,
    }

    candidate.parent_table = parent_table.strip()
    candidate.parent_column = parent_column.strip()
    candidate.child_table = child_table.strip()
    candidate.child_column = child_column.strip()
    candidate.status = "edited"
    candidate.human_note = note
    candidate.evidence_json = {
        **(candidate.evidence_json or {}),
        "original_recommendation": original,
        "corrected_by": "user",
    }

    db.add(
        RelationshipFeedback(
            dataset_id=dataset.dataset_id,
            version_id=version_id,
            relationship_id=candidate.relationship_id,
            parent_table=original["parent_table"],
            parent_column=original["parent_column"],
            child_table=original["child_table"],
            child_column=original["child_column"],
            decision="edited",
            evidence_json={
                "original": original,
                "corrected": {
                    "parent_table": candidate.parent_table,
                    "parent_column": candidate.parent_column,
                    "child_table": candidate.child_table,
                    "child_column": candidate.child_column,
                },
            },
        )
    )

    db.flush()
    return candidate


def list_relationship_candidates(
    db: Session,
    version_id: int,
) -> list[RelationshipCandidate]:
    return (
        db.query(RelationshipCandidate)
        .filter(RelationshipCandidate.version_id == version_id)
        .order_by(RelationshipCandidate.score.desc())
        .all()
    )

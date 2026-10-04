"""Stage 06 first layer: evidence-based metric applicability.

Deterministic. Reads ONLY existing outputs:
  - stored profiling (profiling.py output shape)
  - semantic predictions with human final types (Stage 04)
  - relationship candidates incl. approved/edited/manual ones (Stage 05)
  - prior approved rules of the same version (Stage 06 history)

It never re-computes statistics and never recommends anything. Its output is
an applicability record per column with status APPLICABLE /
NOT_APPLICABLE / NEEDS_REVIEW plus explicit evidence, so every downstream
rule recommendation can explain why a metric was even considered.
"""

from __future__ import annotations

from sqlalchemy.orm import Session

from app.models import RelationshipCandidate, Rule, SemanticPrediction, StoredProfile

METRICS = (
    "completeness",
    "uniqueness",
    "validity",
    "accuracy",
    "consistency",
    "referential_integrity",
    "timeliness",
)

APPLICABLE = "APPLICABLE"
NOT_APPLICABLE = "NOT_APPLICABLE"
NEEDS_REVIEW = "NEEDS_REVIEW"

# Semantic concept names that carry identifier evidence. These mirror the
# knowledge-base concepts seeded for the semantic stage.
IDENTIFIER_CONCEPTS = {
    "Customer ID",
    "Account ID",
    "Product ID",
    "Order ID",
    "Employee ID",
    "Loyalty Member ID",
    "Identifier",
}

EMAIL_CONCEPTS = {"Email", "Email Address"}
PHONE_CONCEPTS = {"Phone", "Phone Number", "Telephone"}
DATETIME_CONCEPTS = {
    "Created At",
    "Updated At",
    "Modified At",
    "Date",
    "Datetime",
    "Timestamp",
    "Order Date",
    "Start Date",
    "End Date",
    "Birth Date",
}
GEO_CONCEPTS = {"Country", "State", "City", "Region", "Province", "Postal Code", "Zip Code"}


def _column_profile(profile_result: dict, table_name: str, column_name: str) -> dict | None:
    for table in profile_result.get("tables", []):
        if table.get("table_name") == table_name:
            for column in table.get("columns", []):
                if column.get("column_name") == column_name:
                    return column
    return None


def _table_profile(profile_result: dict, table_name: str) -> dict | None:
    for table in profile_result.get("tables", []):
        if table.get("table_name") == table_name:
            return table
    return None


def _final_semantic_type(prediction: SemanticPrediction | None) -> str | None:
    """Human final semantic type (Stage 04): user-defined > approved concept."""
    if prediction is None:
        return None
    if prediction.user_defined_type:
        return prediction.user_defined_type
    if prediction.status in {"approved", "edited"} and prediction.predicted_concept:
        return prediction.predicted_concept
    return None


def _semantic_status(prediction: SemanticPrediction | None) -> str | None:
    return prediction.status if prediction else None


class MetricApplicabilityEngine:
    """Deterministic applicability matrix for one dataset version."""

    def __init__(self, db: Session, dataset_id: int, version_id: int):
        self.db = db
        self.dataset_id = dataset_id
        self.version_id = version_id

        stored_profile = (
            db.query(StoredProfile)
            .filter(StoredProfile.version_id == version_id)
            .first()
        )
        self.profile_result: dict = stored_profile.profile_json if stored_profile else {}

        predictions = (
            db.query(SemanticPrediction)
            .filter(SemanticPrediction.version_id == version_id)
            .all()
        )
        self.predictions: dict[tuple[str, str], SemanticPrediction] = {
            (p.table_name, p.column_name): p for p in predictions
        }

        candidates = (
            db.query(RelationshipCandidate)
            .filter(RelationshipCandidate.version_id == version_id)
            .all()
        )
        self.key_candidates = [
            c for c in candidates if c.candidate_kind in {"pk", "composite_pk"}
        ]
        self.approved_relationships = [
            c
            for c in candidates
            if c.candidate_kind == "relationship"
            and c.status in {"approved", "edited", "manual"}
        ]
        # A key candidate the human accepted is strong evidence; pending
        # candidates remain evidence-only (needs review for uniqueness).
        self.approved_keys = [
            c for c in self.key_candidates if c.status in {"approved", "edited", "manual"}
        ]
        self.pending_keys = [
            c for c in self.key_candidates if c.status not in {"approved", "edited", "manual", "rejected"}
        ]
        self.pending_relationships = [
            c
            for c in candidates
            if c.candidate_kind == "relationship" and c.status == "pending"
        ]

        self.approved_rules = (
            db.query(Rule)
            .filter(
                Rule.version_id == version_id,
                Rule.status == "approved",
            )
            .all()
        )

    # ------------------------------------------------------------------
    # Evidence helpers (all reuse stored profiling values)
    # ------------------------------------------------------------------

    def _column_profile_lookup(self, table_name: str, column_name: str) -> dict | None:
        """Public lookup for downstream consumers (feedback features)."""
        return _column_profile(self.profile_result, table_name, column_name)

    def _evidence(self, table_name: str, column_name: str) -> dict:
        column = _column_profile(self.profile_result, table_name, column_name)
        if column is None:
            return {"column_found": False}

        prediction = self.predictions.get((table_name, column_name))
        semantic_type = _final_semantic_type(prediction)

        key_members = self._key_membership(table_name, column_name)

        return {
            "column_found": True,
            "semantic_type": semantic_type,
            "semantic_status": _semantic_status(prediction),
            "semantic_confidence": prediction.confidence_score if prediction else None,
            "data_type": column.get("data_type"),
            "null_percentage": column.get("null_percentage"),
            "empty_string_count": column.get("empty_string_count", 0),
            "whitespace_only_count": column.get("whitespace_only_count", 0),
            "distinct_percentage": column.get("distinct_percentage"),
            "identifier_signal": column.get("identifier_signal"),
            "identifier_name_signal": column.get("identifier_name_signal"),
            "numeric": bool(column.get("numeric")),
            "datetime": bool(column.get("datetime")),
            "categorical": bool(column.get("categorical")),
            "patterns": (column.get("text", {}) or {}).get("patterns", {}) or {},
            "is_true_datetime": str(column.get("data_type", "")).startswith("datetime"),
            "key_member_of": key_members,
        }

    def _key_membership(self, table_name: str, column_name: str) -> list[dict]:
        members = []
        for candidate in self.key_candidates:
            # Key candidates are stored on the parent side (parent_table).
            if candidate.parent_table != table_name:
                continue
            columns = self._key_columns(candidate)
            if column_name in columns:
                members.append(
                    {
                        "relationship_id": candidate.relationship_id,
                        "columns": columns,
                        "status": candidate.status,
                        "composite": len(columns) > 1,
                    }
                )
        return members

    @staticmethod
    def _key_columns(candidate: RelationshipCandidate) -> list[str]:
        """Key candidates store their columns in evidence_json.key_columns."""
        evidence = candidate.evidence_json or {}
        columns = evidence.get("key_columns")
        if isinstance(columns, list) and columns:
            return [str(c) for c in columns]
        # Single-column key candidates fall back to the parent column.
        return [candidate.parent_column] if candidate.parent_column else []

    # ------------------------------------------------------------------
    # Per-metric applicability
    # ------------------------------------------------------------------

    def _completeness(self, ev: dict) -> tuple[str, list[str]]:
        if not ev.get("column_found"):
            return NOT_APPLICABLE, ["Column not found in profile."]
        evidence = [
            f"nulls {ev.get('null_percentage', 0):.2f}%, "
            f"empty strings {ev.get('empty_string_count', 0)}, "
            f"whitespace-only {ev.get('whitespace_only_count', 0)} (profile)"
        ]
        # Technically measurable for every column; whether the column is
        # business-required is a user decision (column metric context).
        return APPLICABLE, ["Measurable for every column (NULL vs EMPTY vs WHITESPACE are distinct)."] + evidence

    def _uniqueness(self, table_name: str, column_name: str, ev: dict) -> tuple[str, list[str]]:
        if not ev.get("column_found"):
            return NOT_APPLICABLE, ["Column not found in profile."]

        evidence: list[str] = []
        semantic_type = ev.get("semantic_type")
        is_identifier_type = semantic_type in IDENTIFIER_CONCEPTS
        identifier_signal = bool(ev.get("identifier_signal"))
        approved_key = any(m["status"] in {"approved", "edited", "manual"} for m in ev["key_member_of"])
        pending_key = any(m["status"] not in {"approved", "edited", "manual", "rejected"} for m in ev["key_member_of"])
        distinct_pct = ev.get("distinct_percentage") or 0.0

        if approved_key:
            return APPLICABLE, [
                "Human-approved key candidate includes this column.",
                f"Observed distinct ratio: {distinct_pct:.1f}% (profile).",
            ]

        if is_identifier_type and identifier_signal:
            return APPLICABLE, [
                f"Final semantic type '{semantic_type}' is an identifier concept (Stage 04).",
                "Profiler identifier signal is strong (name + completeness + uniqueness).",
                f"Observed distinct ratio: {distinct_pct:.1f}% (profile).",
            ]

        if is_identifier_type or identifier_signal or pending_key:
            reasons = []
            if is_identifier_type:
                reasons.append(f"Semantic type '{semantic_type}' suggests an identifier.")
            if identifier_signal:
                reasons.append("Profiler identifier signal present.")
            if pending_key:
                reasons.append("Pending key candidate includes this column.")
            if distinct_pct < 99.0:
                reasons.append(
                    f"Observed distinct ratio is only {distinct_pct:.1f}% - confirm before treating as key."
                )
            return NEEDS_REVIEW, reasons

        return NOT_APPLICABLE, [
            "No identifier evidence: semantic type is not an identifier, profiler identifier signal absent, no key candidate."
        ]

    def _validity(self, ev: dict) -> tuple[str, list[str]]:
        if not ev.get("column_found"):
            return NOT_APPLICABLE, ["Column not found in profile."]

        evidence: list[str] = []
        semantic_type = ev.get("semantic_type")
        patterns = ev.get("patterns", {})

        if semantic_type in EMAIL_CONCEPTS:
            evidence.append(f"Semantic type '{semantic_type}' implies a format domain (email).")
        if ev.get("numeric"):
            evidence.append("Numeric dtype observed in profile (numeric representation check possible).")
        if ev.get("datetime") or ev.get("is_true_datetime"):
            evidence.append("Datetime evidence in profile (parseable-date check possible).")
        if ev.get("categorical"):
            evidence.append("Low-cardinality categorical profile observed (allowed-value domain possible).")
        if patterns.get("email_like", 0) > 0:
            evidence.append("Email-like pattern evidence in profile.")
        if patterns.get("date_like", 0) > 0:
            evidence.append("Date-like pattern evidence in profile.")

        if evidence:
            return APPLICABLE, evidence

        # No domain evidence at all: validity may still apply via a
        # user-provided business rule, but nothing is recommended.
        return NEEDS_REVIEW, [
            "No format/domain evidence in profile or semantic type. Validity applies only if you add a business rule."
        ]

    def _accuracy(self, ev: dict) -> tuple[str, list[str]]:
        if not ev.get("column_found"):
            return NOT_APPLICABLE, ["Column not found in profile."]

        semantic_type = ev.get("semantic_type")
        if semantic_type in EMAIL_CONCEPTS or semantic_type in GEO_CONCEPTS or semantic_type in PHONE_CONCEPTS:
            return NEEDS_REVIEW, [
                f"Semantic type '{semantic_type}' could be verified against an authoritative/reference source, "
                "but accuracy requires you to configure that source. Never claimed from the uploaded dataset alone."
            ]

        return NOT_APPLICABLE, [
            "Accuracy requires an authoritative source or explicit user configuration; none is associated with this column."
        ]

    def _consistency(self, table_name: str, column_name: str, ev: dict) -> tuple[str, list[str]]:
        if not ev.get("column_found"):
            return NOT_APPLICABLE, ["Column not found in profile."]

        evidence: list[str] = []

        # Cross-table hierarchy: approved relationships between two columns
        # of THIS table (e.g. country -> state inside the same table) or a
        # functional-dependency-shaped relationship.
        for rel in self.approved_relationships:
            if rel.child_table == table_name and rel.parent_table == table_name:
                evidence.append(
                    f"Approved relationship {rel.parent_column} -> {rel.child_column} within the table "
                    "supports a hierarchy/dependency check."
                )

        semantic_type = ev.get("semantic_type")
        if semantic_type in GEO_CONCEPTS:
            evidence.append(
                f"Semantic type '{semantic_type}' can participate in a geographic hierarchy "
                "(needs a reference hierarchy or approved relationship)."
            )

        if ev.get("datetime") or ev.get("is_true_datetime"):
            evidence.append("Datetime column can participate in date-order checks with a second date column.")

        if evidence:
            return NEEDS_REVIEW, evidence + [
                "Consistency rules need a second column and approval; statistical similarity alone is never used."
            ]

        return NOT_APPLICABLE, [
            "No cross-column evidence (approved intra-table relationship, hierarchy concept, or second date column)."
        ]

    def _referential_integrity(self, table_name: str, column_name: str, ev: dict) -> tuple[str, list[str]]:
        parents = [
            rel
            for rel in self.approved_relationships
            if rel.child_table == table_name and rel.child_column == column_name
        ]
        if parents:
            rel = parents[0]
            return APPLICABLE, [
                f"Human-approved relationship: {rel.parent_table}.{rel.parent_column} <- "
                f"{rel.child_table}.{rel.child_column} (Stage 05).",
                "RI rules are generated only from approved relationships.",
            ]

        # Pending candidate: possible RI but not authoritative yet.
        pending = [
            rel
            for rel in self.pending_relationships
            if rel.child_table == table_name and rel.child_column == column_name
        ]
        if pending:
            return NEEDS_REVIEW, [
                "A pending relationship candidate exists; approve it in Stage 05 before RI rules are generated."
            ]

        return NOT_APPLICABLE, [
            "No approved relationship targets this column. RI is never generated from name similarity alone."
        ]

    def _timeliness(self, ev: dict) -> tuple[str, list[str]]:
        if not ev.get("column_found"):
            return NOT_APPLICABLE, ["Column not found in profile."]

        semantic_type = ev.get("semantic_type") or ""
        freshness_named = any(
            token in semantic_type.lower()
            for token in ("updated", "created", "modified", "refresh")
        )
        is_datetime = bool(ev.get("datetime")) or ev.get("is_true_datetime")

        if is_datetime and freshness_named:
            return NEEDS_REVIEW, [
                f"Timestamp column '{semantic_type}' may represent record freshness, "
                "but a timeliness SLA must be supplied by you (never invented)."
            ]

        if is_datetime:
            return NEEDS_REVIEW, [
                "Datetime column - timeliness is possible only if this column carries a freshness SLA you define."
            ]

        return NOT_APPLICABLE, ["Not a datetime/timestamp column; no freshness semantics."]

    # ------------------------------------------------------------------

    def applicability_for_column(self, table_name: str, column_name: str) -> dict:
        ev = self._evidence(table_name, column_name)

        checks = {
            "completeness": lambda: self._completeness(ev),
            "uniqueness": lambda: self._uniqueness(table_name, column_name, ev),
            "validity": lambda: self._validity(ev),
            "accuracy": lambda: self._accuracy(ev),
            "consistency": lambda: self._consistency(table_name, column_name, ev),
            "referential_integrity": lambda: self._referential_integrity(table_name, column_name, ev),
            "timeliness": lambda: self._timeliness(ev),
        }

        metrics: dict[str, dict] = {}
        for metric, check in checks.items():
            status, evidence = check()
            metrics[metric] = {"status": status, "evidence": evidence}

        return {
            "table_name": table_name,
            "column_name": column_name,
            "semantic_type": ev.get("semantic_type"),
            "semantic_status": ev.get("semantic_status"),
            "semantic_confidence": ev.get("semantic_confidence"),
            "data_type": ev.get("data_type"),
            "metrics": metrics,
        }

    def applicability_matrix(self) -> dict:
        """Full matrix for the version: one record per profiled column."""
        columns_out: list[dict] = []

        for table in self.profile_result.get("tables", []):
            table_name = table.get("table_name")
            for column in table.get("columns", []):
                columns_out.append(
                    self.applicability_for_column(table_name, column.get("column_name"))
                )

        # Attach pending-relationship lookup lazily (used by RI checks).
        applicable_metrics_count = sum(
            1
            for record in columns_out
            for m in record["metrics"].values()
            if m["status"] == APPLICABLE
        )

        return {
            "version_id": self.version_id,
            "column_count": len(columns_out),
            "applicable_metric_count": applicable_metrics_count,
            "columns": columns_out,
        }


def compute_applicability_matrix(db: Session, version_id: int) -> dict:
    engine = MetricApplicabilityEngine(db, dataset_id, version_id)
    return engine.applicability_matrix()

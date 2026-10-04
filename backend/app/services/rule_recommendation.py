"""Stage 06: metric applicability + deterministic rule candidate generation.

Pipeline position:
    profiling -> semantic -> relationships -> APPLICABILITY -> candidates
    -> parameterization -> deterministic ranking -> validation -> human review

Principles enforced here:
  - Only existing outputs are reused (profile, semantic final types,
    relationship candidates, approved rules). Statistics are never recomputed.
  - Candidates are generated ONLY for metrics the applicability layer marked
    APPLICABLE (or NEEDS_REVIEW when the spec asks for a candidate the user
    must complete - e.g. timeliness SLA, observed numeric ranges).
  - Observed min/max are candidates only; they never silently become
    authoritative business bounds.
  - RI rules are generated only from human-approved relationships.
  - Nothing here is authoritative: everything is born status="recommended"
    and requires human approval.
"""

from __future__ import annotations

from sqlalchemy.orm import Session

from app.models import Rule, StoredProfile
from app.services.metric_applicability import (
    APPLICABLE,
    IDENTIFIER_CONCEPTS,
    EMAIL_CONCEPTS,
    MetricApplicabilityEngine,
    NEEDS_REVIEW,
)
from app.services.rule_templates import METRIC_LABELS
from app.services.rule_validation import validate_rule_dict

# ---------------------------------------------------------------------------
# Deterministic, configurable ranking weights.
# Deliberately transparent and equal-ish: no arbitrary business weighting.
# Each signal contributes when its evidence is present; the score is a count
# of supporting signals with bounded weights, not a black-box probability.
# ---------------------------------------------------------------------------
RANK_WEIGHTS = {
    "approved_key": 0.30,
    "identifier_semantic": 0.25,
    "identifier_signal": 0.20,
    "observed_uniqueness": 0.15,
    "completeness_high": 0.10,
    "pattern_evidence": 0.10,
    "approved_relationship": 0.30,
    "user_context": 0.20,
}


def _confidence_from_score(score: float) -> str:
    if score >= 0.7:
        return "high"
    if score >= 0.4:
        return "medium"
    return "low"


def _find_column_profile(profile_result: dict, table_name: str, column_name: str) -> dict | None:
    for table in profile_result.get("tables", []):
        if table.get("table_name") == table_name:
            for column in table.get("columns", []):
                if column.get("column_name") == column_name:
                    return column
    return None


class RuleCandidateGenerator:
    """Generates structured rule candidates from applicability + evidence."""

    def __init__(self, db: Session, dataset_id: int, version_id: int):
        self.db = db
        self.dataset_id = dataset_id
        self.version_id = version_id
        self.engine = MetricApplicabilityEngine(db, dataset_id, version_id)
        self.profile_result = self.engine.profile_result

        # User business context: column/metric requiredness overrides.
        from app.models import ColumnMetricContext

        contexts = (
            db.query(ColumnMetricContext)
            .filter(ColumnMetricContext.version_id == version_id)
            .all()
        )
        self.contexts = {
            (c.table_name, c.column_name, c.metric): c for c in contexts
        }

    def _required(self, table: str, column: str, metric: str) -> bool:
        context = self.contexts.get((table, column, metric))
        if context is None or context.required is None:
            return True
        return bool(context.required)

    # ------------------------------------------------------------------

    def generate(self) -> list[dict]:
        matrix = self.engine.applicability_matrix()
        candidates: list[dict] = []

        for record in matrix["columns"]:
            table = record["table_name"]
            column = record["column_name"]
            column_profile = _find_column_profile(self.profile_result, table, column)
            if column_profile is None:
                continue

            metrics = record["metrics"]

            if metrics["completeness"]["status"] == APPLICABLE:
                candidates.extend(self._completeness_candidates(record, column_profile))
            if metrics["uniqueness"]["status"] == APPLICABLE:
                candidates.extend(self._uniqueness_candidates(record, column_profile))
            if metrics["validity"]["status"] in {APPLICABLE, NEEDS_REVIEW}:
                candidates.extend(self._validity_candidates(record, column_profile))
            if metrics["referential_integrity"]["status"] == APPLICABLE:
                candidates.extend(self._ri_candidates(record))
            if metrics["consistency"]["status"] in {APPLICABLE, NEEDS_REVIEW}:
                candidates.extend(self._consistency_candidates(record))
            if metrics["timeliness"]["status"] == NEEDS_REVIEW:
                candidates.extend(self._timeliness_candidates(record))

        candidates.extend(self._composite_uniqueness_candidates())
        candidates.extend(self._date_order_candidates())

        return self._rank(candidates)

    # ------------------------------------------------------------------
    # Generators per metric
    # ------------------------------------------------------------------

    def _completeness_candidates(self, record: dict, column_profile: dict) -> list[dict]:
        table = record["table_name"]
        column = record["column_name"]
        candidates = []
        required = self._required(table, column, "completeness")

        empty_count = column_profile.get("empty_string_count", 0) or 0
        ws_count = column_profile.get("whitespace_only_count", 0) or 0
        null_pct = column_profile.get("null_percentage", 0.0) or 0.0

        if not required:
            # Business-requiredness override: do NOT recommend NOT_NULL.
            # The metric stays measurable; only requiredness rules are suppressed.
            candidates.append(
                self._candidate(
                    metric="completeness",
                    template=None,
                    table=table,
                    columns=[column],
                    rule_name=f"{column} is optional (business context: not required)",
                    parameters={},
                    evidence=record["metrics"]["completeness"]["evidence"]
                    + ["Business context marks this column as not required; requiredness rules suppressed."],
                    features={"null_percentage": null_pct, "required": False},
                    confidence="high",
                    skip=True,  # informational, never a rule row
                )
            )
            return candidates

        evidence = list(record["metrics"]["completeness"]["evidence"])
        features = {
            "null_percentage": null_pct,
            "empty_string_count": empty_count,
            "whitespace_only_count": ws_count,
        }

        candidates.append(
            self._candidate(
                metric="completeness",
                template="NOT_NULL",
                table=table,
                columns=[column],
                rule_name=f"{column} must not be NULL",
                parameters={},
                evidence=evidence,
                features=features,
                confidence="high" if null_pct < 5 else "medium",
            )
        )

        # NOT_EMPTY / NOT_WHITESPACE only when the profile shows such values
        # exist (or for text columns, where the distinction is meaningful).
        is_text = (column_profile.get("text") or {}) or column_profile.get("data_type") == "object"
        if empty_count > 0 or is_text:
            candidates.append(
                self._candidate(
                    metric="completeness",
                    template="NOT_EMPTY",
                    table=table,
                    columns=[column],
                    rule_name=f"{column} must not be empty (\"\" differs from NULL)",
                    parameters={},
                    evidence=evidence + [f"Profile shows {empty_count} empty-string value(s)."],
                    features=features,
                    confidence="medium",
                )
            )
        if ws_count > 0 or is_text:
            candidates.append(
                self._candidate(
                    metric="completeness",
                    template="NOT_WHITESPACE",
                    table=table,
                    columns=[column],
                    rule_name=f"{column} must not be whitespace-only (\"  \" differs from NULL and \"\")",
                    parameters={},
                    evidence=evidence + [f"Profile shows {ws_count} whitespace-only value(s)."],
                    features=features,
                    confidence="medium",
                )
            )

        return candidates

    def _uniqueness_candidates(self, record: dict, column_profile: dict) -> list[dict]:
        table = record["table_name"]
        column = record["column_name"]
        distinct_pct = column_profile.get("distinct_percentage", 0.0) or 0.0

        features = {
            "distinct_percentage": distinct_pct,
            "identifier_signal": bool(column_profile.get("identifier_signal")),
            "semantic_type": record.get("semantic_type"),
        }

        score = 0.0
        if any(m["status"] in {"approved", "edited", "manual"} for m in record.get("_key_members", [])):
            score += RANK_WEIGHTS["approved_key"]
        if record.get("semantic_type") in IDENTIFIER_CONCEPTS:
            score += RANK_WEIGHTS["identifier_semantic"]
        if column_profile.get("identifier_signal"):
            score += RANK_WEIGHTS["identifier_signal"]
        if distinct_pct >= 99.0:
            score += RANK_WEIGHTS["observed_uniqueness"]

        evidence = list(record["metrics"]["uniqueness"]["evidence"])
        evidence.append(
            f"Ranking evidence: approved key / identifier semantic type / identifier signal / observed uniqueness "
            f"= {score:.2f} (deterministic weights, reviewable)."
        )

        return [
            self._candidate(
                metric="uniqueness",
                template="UNIQUE",
                table=table,
                columns=[column],
                rule_name=f"{column} values must be unique",
                parameters={"ignore_nulls": True, "normalize_whitespace": True},
                evidence=evidence,
                features=features,
                confidence=_confidence_from_score(score),
                score=score,
            )
        ]

    def _validity_candidates(self, record: dict, column_profile: dict) -> list[dict]:
        candidates = []
        table = record["table_name"]
        column = record["column_name"]
        semantic_type = record.get("semantic_type")
        patterns = (column_profile.get("text", {}) or {}).get("patterns", {}) or {}

        # Email syntax from semantic type + observed pattern evidence.
        if semantic_type in EMAIL_CONCEPTS and patterns.get("email_like", 0) > 0:
            candidates.append(
                self._candidate(
                    metric="validity",
                    template="REGEX",
                    table=table,
                    columns=[column],
                    rule_name=f"{column} must be a valid email address",
                    parameters={"pattern_name": "email_syntax"},
                    evidence=list(record["metrics"]["validity"]["evidence"]),
                    features={"email_like_count": patterns.get("email_like", 0)},
                    confidence="high",
                    score=RANK_WEIGHTS["pattern_evidence"] + 0.5,
                    legacy_check="email_syntax",
                )
            )

        # Numeric representation validity.
        if column_profile.get("numeric"):
            candidates.append(
                self._candidate(
                    metric="validity",
                    template="NUMERIC_TYPE",
                    table=table,
                    columns=[column],
                    rule_name=f"{column} must be numeric",
                    parameters={},
                    evidence=list(record["metrics"]["validity"]["evidence"]),
                    features={"data_type": column_profile.get("data_type")},
                    confidence="high",
                    score=RANK_WEIGHTS["pattern_evidence"],
                )
            )

        # Observed-range candidate: ALWAYS labeled as a candidate from the
        # observed distribution - never silently an authoritative bound.
        numeric = column_profile.get("numeric") or {}
        if numeric and numeric.get("min") is not None and numeric.get("max") is not None:
            candidates.append(
                self._candidate(
                    metric="validity",
                    template="NUMERIC_RANGE",
                    table=table,
                    columns=[column],
                    rule_name=(
                        f"{column} potential range constraint (observed "
                        f"{numeric['min']}..{numeric['max']} - candidate only, confirm business bounds)"
                    ),
                    parameters={"min": numeric["min"], "max": numeric["max"]},
                    evidence=[
                        "Potential numeric range rule detected from the observed distribution.",
                        f"Observed min={numeric['min']}, max={numeric['max']} (profile).",
                        "These observed bounds are a CANDIDATE; accepting is a business decision.",
                    ],
                    features={
                        "observed_min": numeric["min"],
                        "observed_max": numeric["max"],
                        "origin": "observed_distribution",
                    },
                    confidence="low",
                    score=0.2,
                    origin="observed_distribution",
                )
            )

        # Date parseability.
        if column_profile.get("datetime") or str(column_profile.get("data_type", "")).startswith("datetime"):
            candidates.append(
                self._candidate(
                    metric="validity",
                    template="DATE_FORMAT",
                    table=table,
                    columns=[column],
                    rule_name=f"{column} must be a parseable date/timestamp",
                    parameters={},
                    evidence=list(record["metrics"]["validity"]["evidence"]),
                    features={"datetime": True},
                    confidence="medium",
                    score=RANK_WEIGHTS["pattern_evidence"],
                )
            )

        # Allowed-values candidate for low-cardinality categoricals.
        categorical = column_profile.get("categorical") or {}
        top_values = categorical.get("top_values") or []
        if top_values and len(top_values) <= 25:
            values = [str(v.get("value")) for v in top_values]
            candidates.append(
                self._candidate(
                    metric="validity",
                    template="ALLOWED_VALUES",
                    table=table,
                    columns=[column],
                    rule_name=(
                        f"{column} allowed-values candidate (observed "
                        f"{len(values)} distinct values - candidate only)"
                    ),
                    parameters={"allowed_values": values},
                    evidence=[
                        "Low-cardinality categorical profile observed.",
                        "Observed values are a CANDIDATE domain; confirm the business value list before approval.",
                    ],
                    features={"observed_values": values, "origin": "observed_distribution"},
                    confidence="low",
                    score=0.2,
                    origin="observed_distribution",
                )
            )

        return candidates

    def _ri_candidates(self, record: dict) -> list[dict]:
        candidates = []
        table = record["table_name"]
        column = record["column_name"]

        for rel in self.engine.approved_relationships:
            if rel.child_table == table and rel.child_column == column:
                candidates.append(
                    self._candidate(
                        metric="referential_integrity",
                        template="FOREIGN_KEY_EXISTS",
                        table=table,
                        columns=[column],
                        rule_name=(
                            f"{column} must exist in {rel.parent_table}.{rel.parent_column}"
                        ),
                        parameters={
                            "parent_table": rel.parent_table,
                            "parent_column": rel.parent_column,
                            "child_table": table,
                            "child_column": column,
                        },
                        evidence=list(record["metrics"]["referential_integrity"]["evidence"]),
                        features={
                            "relationship_id": rel.relationship_id,
                            "relationship_score": rel.score,
                            "containment": (rel.stats_json or {}).get("containment"),
                        },
                        confidence="high",
                        score=RANK_WEIGHTS["approved_relationship"],
                    )
                )

        return candidates

    def _consistency_candidates(self, record: dict) -> list[dict]:
        """Intra-table dependency candidates from APPROVED relationships only."""
        candidates = []
        table = record["table_name"]
        column = record["column_name"]

        for rel in self.engine.approved_relationships:
            if rel.parent_table == table and rel.child_table == table and rel.parent_column != rel.child_column:
                if record["column_name"] == rel.child_column:
                    candidates.append(
                        self._candidate(
                            metric="consistency",
                            template="FUNCTIONAL_DEPENDENCY",
                            table=table,
                            columns=[rel.parent_column, rel.child_column],
                            rule_name=(
                                f"{rel.parent_column} determines {rel.child_column} "
                                "(dependency candidate from approved relationship)"
                            ),
                            parameters={
                                "determinant_column": rel.parent_column,
                                "dependent_column": rel.child_column,
                            },
                            evidence=list(record["metrics"]["consistency"]["evidence"]),
                            features={"relationship_id": rel.relationship_id},
                            confidence="medium",
                            score=0.3,
                        )
                    )

        return candidates

    def _timeliness_candidates(self, record: dict) -> list[dict]:
        """Timeliness candidates ALWAYS need a user SLA; generated with empty
        parameters so validation is INVALID until the user edits the rule
        with a threshold. This makes the 'ask the user' step explicit."""
        table = record["table_name"]
        column = record["column_name"]

        return [
            self._candidate(
                metric="timeliness",
                template="FRESHNESS_THRESHOLD",
                table=table,
                columns=[column],
                rule_name=(
                    f"{column} freshness SLA (define a threshold - the system never invents one)"
                ),
                parameters={},
                evidence=list(record["metrics"]["timeliness"]["evidence"]),
                features={"needs_user_sla": True},
                confidence="low",
                score=0.1,
                origin="needs_user_sla",
            )
        ]

    def _composite_uniqueness_candidates(self) -> list[dict]:
        """Reuse profiling composite candidates - never recomputed here."""
        candidates = []
        for table in self.profile_result.get("tables", []):
            table_name = table.get("table_name")
            for combo in table.get("composite_uniqueness_candidates", []) or []:
                columns = list(combo.get("columns") or [])
                if len(columns) < 2:
                    continue
                candidates.append(
                    self._candidate(
                        metric="uniqueness",
                        template="COMPOSITE_UNIQUE",
                        table=table_name,
                        columns=columns,
                        rule_name=(
                            f"Combination ({', '.join(columns)}) should be unique "
                            f"({combo.get('composite_uniqueness_percentage', 0):.1f}% observed - candidate)"
                        ),
                        parameters={"ignore_nulls": True, "normalize_whitespace": True},
                        evidence=[
                            "Composite uniqueness candidate from profiling (existing candidate list reused).",
                            f"Observed composite uniqueness: {combo.get('composite_uniqueness_percentage', 0):.1f}% "
                            f"over {combo.get('evaluated_row_count', 0)} rows.",
                            "The user decides; this is never auto-declared a key.",
                        ],
                        features={
                            "composite_uniqueness_percentage": combo.get("composite_uniqueness_percentage"),
                            "evaluated_row_count": combo.get("evaluated_row_count"),
                        },
                        confidence="medium",
                        score=0.35,
                    )
                )
        return candidates

    def _date_order_candidates(self) -> list[dict]:
        """Date-order candidates when a table has two date-like columns whose
        names suggest ordering (start/end, created/closed). Deterministic
        name-based pairing; always a candidate requiring review."""
        pairs = [
            ("start", "end"),
            ("begin", "end"),
            ("created", "closed"),
            ("opened", "closed"),
            ("effective", "expiry"),
            ("effective", "expiration"),
        ]
        candidates = []

        for table in self.profile_result.get("tables", []):
            table_name = table.get("table_name")
            date_columns = []
            for column in table.get("columns", []):
                if column.get("datetime") or str(column.get("data_type", "")).startswith("datetime"):
                    date_columns.append(str(column.get("column_name")))

            lowered = {c: c.lower() for c in date_columns}
            for left_prefix, right_prefix in pairs:
                left = next((c for c, low in lowered.items() if left_prefix in low), None)
                right = next((c for c, low in lowered.items() if right_prefix in low), None)
                if left and right and left != right:
                    candidates.append(
                        self._candidate(
                            metric="consistency",
                            template="DATE_ORDER",
                            table=table_name,
                            columns=[left, right],
                            rule_name=f"{left} must be before {right}",
                            parameters={"left_column": left, "right_column": right, "allow_equal": False},
                            evidence=[
                                f"Both {left} and {right} are datetime columns (profile).",
                                "Names suggest an ordering relation; confirm the business rule.",
                            ],
                            features={"left": left, "right": right},
                            confidence="medium",
                            score=0.3,
                        )
                    )

        return candidates

    # ------------------------------------------------------------------
    # Candidate assembly + deterministic ranking
    # ------------------------------------------------------------------

    def _candidate(
        self,
        *,
        metric: str,
        template: str | None,
        table: str,
        columns: list[str],
        rule_name: str,
        parameters: dict,
        evidence: list[str],
        features: dict,
        confidence: str,
        score: float = 0.0,
        origin: str | None = None,
        legacy_check: str | None = None,
        skip: bool = False,
    ) -> dict:
        if skip:
            return {"skip": True}

        rule_json: dict = {
            "rule_template": template,
            "metric": metric,
            "table": table,
            "column": columns[0] if len(columns) == 1 else None,
            "columns": columns if len(columns) > 1 else None,
            "parameters": parameters,
        }
        if origin:
            rule_json["origin"] = origin
        if legacy_check:
            # Backwards-compatible field so older validation paths recognize
            # the email rule.
            rule_json["check"] = legacy_check

        return {
            "rule_name": rule_name,
            "rule_json": rule_json,
            "metric": metric,
            "metric_label": METRIC_LABELS.get(metric, metric),
            "rule_template": template,
            "table_name": table,
            "target_columns": columns,
            "evidence": evidence,
            "features": features,
            "confidence": confidence,
            "score": round(score, 4),
            "risk": "low",
            "origin": origin,
        }

    def _rank(self, candidates: list[dict]) -> list[dict]:
        """Deterministic order: metric, then score desc, then name."""
        real = [c for c in candidates if not c.get("skip")]
        real.sort(
            key=lambda c: (
                c["table_name"],
                c["target_columns"][0] if c["target_columns"] else "",
                c["metric"],
                -c["score"],
                c["rule_name"],
            )
        )
        return real


def recommend_rules_for_version(
    db: Session,
    dataset_id: int,
    version_id: int,
) -> list[dict]:
    """Generate ranked rule candidates for one dataset version."""
    stored_profile = (
        db.query(StoredProfile)
        .filter(StoredProfile.version_id == version_id)
        .first()
    )

    if stored_profile is None:
        raise ValueError("Profiling must run before rule recommendation.")

    generator = RuleCandidateGenerator(db, dataset_id, version_id)
    return generator.generate()


def save_recommendations(
    db: Session,
    dataset_id: int,
    version_id: int,
    recommendations: list[dict],
) -> list[Rule]:
    """Persist recommendations as Rule rows (status=recommended)."""
    saved = []

    for recommendation in recommendations:
        rule_json = recommendation["rule_json"]

        validation = validate_rule_dict(
            db,
            dataset_id,
            version_id,
            rule_json,
        )

        rule = Rule(
            dataset_id=dataset_id,
            version_id=version_id,
            table_name=recommendation["table_name"],
            rule_name=recommendation["rule_name"],
            rule_json=rule_json,
            metric=recommendation["metric"],
            source="system",
            risk=recommendation.get("risk", "low"),
            validation_status=validation["status"],
            validation_issues_json={"issues": validation["issues"]},
            status="recommended",
        )

        db.add(rule)
        saved.append(rule)

    db.flush()
    return saved

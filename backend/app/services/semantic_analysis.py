"""Stage 04 semantic analysis: two-level decisions with evidence.

Decision outcomes (v2):

- ``KB_MATCH``: a specific Knowledge Base concept wins (exact/alias stage or
  retrieval + evidence), gated by structural conflict checks.
- ``FAMILY_MATCH``: no specific concept fits, but a generic family
  (identifier, boolean flag, money, ...) does - a normalized new-concept
  proposal is returned for human review.
- ``UNKNOWN``: nothing fits. Opaque names without a description ask the user
  directly (no suggestion); readable names receive a name-derived proposal.

``match_status``/``source`` keep their historical values (MATCHED/UNKNOWN,
KB/OPEN_DISCOVERY); ``decision_kind`` and ``semantic_family`` are NEW keys.
"""

from sqlalchemy.orm import Session

from app.config import (
    SEMANTIC_ALTERNATIVE_LIMIT,
    SEMANTIC_TOP_K,
)
from app.models import SemanticConcept
from app.services.semantic_features import (
    calculate_deterministic_score,
    collect_gates,
    concept_info,
    evidence_coverage,
    feature_payload,
    generate_features,
)
from app.services.semantic_name import normalize_name
from app.services.semantic_retrieval import (
    SemanticRetrievalService,
)


# Score/gate caps applied by the decision rules (kept in sync with
# app.config thresholds: below probable -> at most Ambiguous, below
# ambiguous -> never confident). Imported from config so the harness can
# re-tune the thresholds in one place.
from app.config import (
    SEMANTIC_PROBABLE_THRESHOLD as _PROBABLE_THRESHOLD,
    SEMANTIC_AMBIGUOUS_THRESHOLD as _AMBIGUOUS_THRESHOLD,
)

DECISION_THRESHOLD = _PROBABLE_THRESHOLD
# A FAMILY_MATCH is a PROPOSAL for human review, not a confident KB match:
# the family layer accepts candidates at the ambiguous threshold.
FAMILY_THRESHOLD = _AMBIGUOUS_THRESHOLD
_AMBIGUOUS_CAP = _PROBABLE_THRESHOLD - 0.01  # never "Probable"
_UNKNOWN_CAP = _AMBIGUOUS_THRESHOLD - 0.01  # never confident


def build_proposal(
    name_parts,
    info: dict | None,
    profile: dict | None,
    decision_kind: str,
) -> dict:
    """Build the new-concept proposal for FAMILY_MATCH / UNKNOWN columns.

    The proposal name comes from the ordered, abbreviation-expanded tokens
    (after prefix stripping) - never a guessed role word ("Phone" proposes
    "phone", never "phone_identifier"). Opaque names without a description
    produce ``ask_user`` with NO name.
    """
    if name_parts.opaque and not (info and info.get("description")):
        return {
            "name": None,
            "family": info.get("family") if info else None,
            "source": "profile" if profile else "name",
            "ask_user": True,
            "group_key": None,
        }

    tokens = [
        token
        for token in name_parts.expanded_tokens
        if len(token) > 1 and not token.isdigit()
    ]

    if not tokens:
        return {
            "name": None,
            "family": info.get("family") if info else None,
            "source": "profile" if profile else "name",
            "ask_user": True,
            "group_key": None,
        }

    name = "_".join(tokens)
    group_key = name.lower()

    return {
        "name": name,
        "family": info.get("family") if info else None,
        "source": "name",
        "ask_user": False,
        "group_key": group_key,
    }


def _apply_caps(
    score: float,
    gates: dict,
) -> float:
    """Apply the structural-gate score caps to a candidate score.

    entity_conflict: can never be a confident match (below the ambiguous
    threshold). profile_veto / head_conflict: at most Ambiguous (below the
    probable threshold). Caps are applied BEFORE the decision kind is
    derived so a capped score can never read as a KB match.
    """
    if gates.get("entity_conflict"):
        return min(score, _UNKNOWN_CAP)
    if gates.get("profile_veto") or gates.get("head_conflict"):
        return min(score, _AMBIGUOUS_CAP)
    return score


def _decide_kind(
    gates: dict,
    score: float,
    threshold: float,
    is_family: bool,
) -> str:
    """Map (gates, score) to a decision kind for a candidate."""
    if gates.get("entity_conflict"):
        return "FAMILY_MATCH" if is_family else "UNKNOWN"
    if score >= threshold and not gates.get("profile_veto"):
        return "KB_MATCH"
    if is_family:
        # A family that clears the threshold is a FAMILY_MATCH even under a
        # head conflict (head conflicts cap SPECIFIC concepts at Ambiguous,
        # they do not block the generic fallback layer).
        return "FAMILY_MATCH" if score >= threshold else "UNKNOWN"
    return "UNKNOWN"


def _empty_result(column_name: str) -> dict:
    return {
        "column": column_name,
        "semantic_concept": None,
        "concept_id": None,
        "confidence_score": 0.0,
        "evidence": {},
        "evidence_coverage": 0.0,
        "alternatives": [],
        "source": "OPEN_DISCOVERY",
        "match_status": "UNKNOWN",
        "decision_kind": "UNKNOWN",
        "semantic_family": None,
        "proposal": build_proposal(
            normalize_name(column_name), None, None, "UNKNOWN"
        ),
        "candidates": [],
        "decision_rule": "no_candidates",
        "margin": 0.0,
        "confidence_level": "Unknown",
    }


class SemanticAnalysisService:
    """
    Performs Stage 04 semantic understanding by combining
    embedding retrieval with deterministic evidence scoring.

    v2: two-level output. A specific KB concept is recommended only when it
    survives the structural gates; otherwise the generic family layer
    proposes a new concept, or the column is reported as unknown. The full
    ranked candidate list with gates is returned for evidence logging.
    """

    def __init__(self):
        self.retrieval_service = SemanticRetrievalService()

    # ------------------------------------------------------------------
    # Evidence assembly
    # ------------------------------------------------------------------

    def _load_concepts(self, db: Session) -> dict[int, SemanticConcept]:
        return {
            concept.concept_id: concept
            for concept in db.query(SemanticConcept).all()
        }

    def _score_candidates(
        self,
        db: Session,
        column_name: str,
        name_parts,
        column_description: str | None,
        data_type: str | None,
        profile: dict | None,
        dataset_domain: str | None,
        table_context: str | None,
        sibling_names: list[str] | None,
        relationship_hints: list[str] | None,
        candidates: list[dict],
        concepts: dict[int, SemanticConcept],
        stage: str,
    ) -> list[dict]:
        """Score retrieved candidates and apply caps; returns ranked list."""
        ranked: list[dict] = []

        for candidate in candidates:
            concept = concepts.get(candidate["concept_id"])
            if concept is None:
                continue

            info = concept_info(concept)
            features = generate_features(
                column_name=column_name,
                concept=concept,
                embedding_similarity=candidate["similarity"],
                column_description=column_description,
                data_type=data_type,
                profile=profile,
                dataset_domain=dataset_domain,
                table_context=table_context,
                sibling_names=sibling_names,
                relationship_hints=relationship_hints,
                name_parts=name_parts,
                concept_data=info,
            )

            gates = collect_gates(features)
            score = calculate_deterministic_score(features)
            is_family = bool(info["is_family"])
            family_only = bool(info.get("family_only", False))
            capped = _apply_caps(score, gates)
            decision_kind = _decide_kind(
                gates, capped, DECISION_THRESHOLD, is_family
            )
            coverage = evidence_coverage(features)

            ranked.append(
                {
                    "concept_id": concept.concept_id,
                    "concept": concept.concept_name,
                    "category": concept.category,
                    "is_family": is_family,
                    "family_only": family_only,
                    "family": info.get("family"),
                    "role": info.get("role"),
                    "confidence_score": capped,
                    "raw_score": score,
                    "evidence": feature_payload(features),
                    "evidence_coverage": coverage,
                    "gates": gates,
                    "decision_kind": decision_kind,
                    "stage": stage,
                }
            )

        ranked.sort(
            key=lambda item: item["confidence_score"],
            reverse=True,
        )
        return ranked

    def _exact_candidates(
        self,
        concepts: dict[int, SemanticConcept],
        name_parts,
        profile: dict | None,
    ) -> tuple[list[dict], list[dict]]:
        """Exact/alias stage (Part B): concepts whose name/alias key EQUALS
        the column's expanded-token key.

        Returns (exact_candidates, conflicts) where conflicts lists concept
        names whose key matched but whose profile contradicts the exact
        reading (caller records exact_but_profile_conflict and continues).
        """
        from app.services.semantic_features import data_type_compatibility, observed_profile_role

        column_key = name_parts.key
        if not column_key:
            return [], []

        exact: list[dict] = []
        conflicts: list[dict] = []

        for concept_id, concept in concepts.items():
            info = concept_info(concept)
            if column_key not in info["keys"]:
                continue

            contradiction = False

            if profile:
                # Datatype must not be applicable-negative.
                datatype_value, datatype_applicable = (
                    data_type_compatibility(
                        str(profile.get("data_type", "")) or None,
                        concept,
                        profile=profile,
                        info=info,
                    )
                )
                if datatype_applicable and datatype_value == 0.0:
                    contradiction = True

                # Observed role must be compatible with the concept role.
                observed = observed_profile_role(profile)
                concept_role = info.get("role")
                if concept_role:
                    allowed = {
                        "identifier": {"identifier", "code", "count", "text"},
                        "code": {"code", "identifier", "categorical", "text"},
                        "measure": {"measure", "count"},
                        "count": {"count", "measure", "identifier", "code"},
                        "flag": {"flag", "categorical"},
                        "text": {"text", "categorical", "identifier", "code"},
                        "categorical": {"categorical", "text", "flag", "code"},
                        "date": {"date", "time"},
                        "time": {"time", "date"},
                    }.get(str(concept_role), None)
                    if allowed is not None and not (observed & allowed):
                        contradiction = True

                # Constant / near-constant columns are not identifiers.
                if (
                    concept_role in ("identifier", "code")
                    and (
                        (profile.get("numeric") or {}).get("constant")
                        or (profile.get("text") or {}).get("constant")
                        or (profile.get("numeric") or {}).get("near_constant")
                        or (profile.get("text") or {}).get("near_constant")
                    )
                ):
                    contradiction = True

            entry = {
                "concept_id": concept_id,
                "concept": concept.concept_name,
                "category": concept.category,
                "similarity": 1.0,
                "lexical": 1.0,
            }
            if contradiction:
                conflicts.append(entry)
            else:
                exact.append(entry)

        return exact, conflicts

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def analyze_column(
        self,
        db: Session,
        column_name: str,
        column_description: str | None = None,
        data_type: str | None = None,
        profile: dict | None = None,
        dataset_domain: str | None = None,
        table_context: str | None = None,
        top_k: int | None = None,
        sibling_names: list[str] | None = None,
        relationship_hints: list[str] | None = None,
        query_embedding=None,
        skip_exact_stage: bool = False,
    ) -> dict:
        """Analyze the semantic meaning of one incoming column."""
        top_k = top_k or SEMANTIC_TOP_K
        name_parts = normalize_name(column_name)

        concepts = self._load_concepts(db)

        # --------------------------------------------------------------
        # Part B: exact/alias stage. A full key match with a
        # non-contradicting profile is an immediate KB_MATCH.
        # --------------------------------------------------------------
        exact_but_profile_conflict = False
        if not skip_exact_stage and not name_parts.opaque:
            exact_candidates, conflicts = self._exact_candidates(
                concepts, name_parts, profile
            )
            exact_but_profile_conflict = bool(conflicts)

            if exact_candidates:
                ranked = self._score_candidates(
                    db,
                    column_name,
                    name_parts,
                    column_description,
                    data_type,
                    profile,
                    dataset_domain,
                    table_context,
                    sibling_names,
                    relationship_hints,
                    exact_candidates,
                    concepts,
                    stage="exact",
                )
                # Part B: an exact match with STRONG evidence (high score,
                # no conflict gate, clear of the runner-up) is decided
                # WITHOUT retrieval. Weaker exact readings continue to the
                # retrieval pass instead of short-circuiting.
                if ranked:
                    best = ranked[0]
                    margin = (
                        best["confidence_score"] - ranked[1]["confidence_score"]
                        if len(ranked) > 1
                        else best["confidence_score"]
                    )
                    if (
                        best["confidence_score"] >= DECISION_THRESHOLD
                        and not any(best["gates"].values())
                        and margin >= 0.05
                        and best["decision_kind"] == "KB_MATCH"
                    ):
                        return self._finalize(
                            column_name=column_name,
                            name_parts=name_parts,
                            ranked=ranked,
                            ranked_from_retrieval=False,
                            exact_but_profile_conflict=False,
                            top_k=top_k,
                        )

        # --------------------------------------------------------------
        # Retrieval path (hybrid union of embedding + lexical candidates)
        # --------------------------------------------------------------
        # Score EVERY concept (68 rows is cheap with the matrix cache):
        # top-k truncation before evidence scoring would let the encoder
        # decide, and the family fallback must see ALL family rows.
        candidates = self.retrieval_service.retrieve_candidates(
            db=db,
            column_name=column_name,
            column_description=column_description,
            dataset_domain=dataset_domain,
            table_context=table_context,
            top_k=len(concepts),
            sibling_names=sibling_names,
            query_embedding=query_embedding,
        )

        ranked = self._score_candidates(
            db,
            column_name,
            name_parts,
            column_description,
            data_type,
            profile,
            dataset_domain,
            table_context,
            sibling_names,
            relationship_hints,
            candidates,
            concepts,
            stage="retrieval",
        )

        return self._finalize(
            column_name=column_name,
            name_parts=name_parts,
            ranked=ranked,
            ranked_from_retrieval=True,
            exact_but_profile_conflict=exact_but_profile_conflict,
            top_k=top_k,
        )

    # ------------------------------------------------------------------
    # Decision + proposal assembly
    # ------------------------------------------------------------------

    def _finalize(
        self,
        column_name: str,
        name_parts,
        ranked: list[dict],
        ranked_from_retrieval: bool,
        exact_but_profile_conflict: bool,
        top_k: int,
    ) -> dict:
        if not ranked:
            result = _empty_result(column_name)
            result["exact_but_profile_conflict"] = exact_but_profile_conflict
            return result

        # Dual-role rows (a specific concept that doubles as its family,
        # e.g. "Email") stay specific-eligible; ONLY family_only rows are
        # excluded from pass 1.
        specific = [
            item for item in ranked if not item.get("family_only", False)
        ]
        families = [item for item in ranked if item["is_family"]]

        # ---------------- pass 1: specific concepts ----------------
        best_specific = specific[0] if specific else None
        best_family = families[0] if families else None

        margin = 0.0
        if len(ranked) >= 2:
            margin = ranked[0]["confidence_score"] - ranked[1]["confidence_score"]

        chosen = None
        decision_kind = "UNKNOWN"
        decision_rule = None

        if best_specific is not None:
            if best_specific["decision_kind"] == "KB_MATCH":
                chosen = best_specific
                decision_kind = "KB_MATCH"
                decision_rule = "specific_kb_match"
            elif best_specific["decision_kind"] == "FAMILY_MATCH":
                # Entity-conflicted or vetoed specific: not a KB match, but
                # its family metadata may still fit.
                chosen = None
                decision_rule = "specific_blocked_by_gate"
            else:
                decision_rule = "specific_below_threshold"

        # ---------------- pass 2: family fallback ----------------
        if chosen is None:
            # Families ranked by capped score; a specific concept that was
            # only entity-conflicted may carry a family whose entity also
            # does not match (warehouse_x -> Identifier would be wrong), so
            # skip family rows whose evidence ALSO conflicts.
            family_candidates = [
                item
                for item in ranked
                if item["is_family"]
                and not item["gates"].get("entity_conflict")
            ]
            if family_candidates:
                candidate = family_candidates[0]
                if candidate["confidence_score"] >= FAMILY_THRESHOLD:
                    chosen = candidate
                    decision_kind = "FAMILY_MATCH"
                    decision_rule = "family_fallback"

        if chosen is None:
            # ---------------- UNKNOWN ----------------
            leader = ranked[0]
            proposal = build_proposal(
                name_parts, None, None, "UNKNOWN"
            )
            return {
                "column": column_name,
                "semantic_concept": None,
                "concept_id": None,
                "confidence_score": leader["confidence_score"],
                "evidence": leader["evidence"],
                "evidence_coverage": leader["evidence_coverage"],
                "alternatives": [
                    {
                        "concept": item["concept"],
                        "score": item["confidence_score"],
                        "concept_id": item["concept_id"],
                    }
                    for item in ranked[1 : 1 + SEMANTIC_ALTERNATIVE_LIMIT]
                ],
                "source": "OPEN_DISCOVERY",
                "match_status": "UNKNOWN",
                "decision_kind": "UNKNOWN",
                "semantic_family": None,
                "proposal": proposal,
                "candidates": ranked[:top_k],
                "decision_rule": decision_rule or "unknown",
                "margin": round(margin, 4),
                "exact_but_profile_conflict": exact_but_profile_conflict,
            }

        # ---------------- KB_MATCH / FAMILY_MATCH ----------------
        alternatives = [
            {
                "concept": item["concept"],
                "score": item["confidence_score"],
                "concept_id": item["concept_id"],
            }
            for item in ranked
            if item is not chosen
        ][:SEMANTIC_ALTERNATIVE_LIMIT]

        if decision_kind == "KB_MATCH":
            proposal = None
            family = chosen["family"]
            source = "KB"
            match_status = "MATCHED"
            if chosen["stage"] == "exact":
                match_status = "MATCHED"
        else:
            proposal = build_proposal(
                name_parts,
                {
                    "family": chosen["family"],
                    "description": chosen["concept"],
                },
                None,
                "FAMILY_MATCH",
            )
            family = chosen["family"]
            source = "OPEN_DISCOVERY"
            match_status = "UNKNOWN"

        return {
            "column": column_name,
            "semantic_concept": (
                chosen["concept"] if decision_kind == "KB_MATCH" else None
            ),
            "concept_id": (
                chosen["concept_id"] if decision_kind == "KB_MATCH" else None
            ),
            "confidence_score": chosen["confidence_score"],
            "evidence": chosen["evidence"],
            "evidence_coverage": chosen["evidence_coverage"],
            "alternatives": alternatives,
            "source": source,
            "match_status": match_status,
            "decision_kind": decision_kind,
            "semantic_family": family,
            "proposal": proposal,
            "candidates": ranked[:top_k],
            "decision_rule": decision_rule,
            "margin": round(margin, 4),
            "exact_stage": chosen["stage"] == "exact",
            "exact_but_profile_conflict": exact_but_profile_conflict,
        }

"""Deterministic semantic evidence features.

Every feature measures ONE piece of evidence about a (column, concept)
candidate. Similarity is evidence, never proof: the deterministic score is a
documented, reproducible weighted mean of applicable evidence, and no single
feature (lexical, embedding, datatype or profile) can decide a mapping alone.

Applicability convention
------------------------
A feature that CANNOT be computed for a candidate (no column description
supplied, no dataset domain/table context configured, concept defines no
expected data types or profile expectations) is reported as
``applicable = False`` with ``value = 0.0``. It is then EXCLUDED from the
weighted score and from evidence coverage instead of silently counting as a
failed check. A feature that CAN be computed and simply disagrees
(incompatible datatype, failed profile expectation, dissimilar description)
is ``applicable = True`` and counts - including against the candidate.

The score is a weighted mean configurable in app.config; it is NOT a
calibrated probability. Confidence levels (Strong/Probable/Ambiguous/Unknown)
are assigned in semantic_stage.confidence_level using the score, evidence
coverage and the top-1/top-2 margin.

Gates (v2)
----------
Name evidence carries structural conflict flags that the decision layer
applies as caps:

- ``entity_conflict``: the column has entity tokens (what it is about) and
  the concept asserts entity tokens too, but none of them appear in the
  concept's name/aliases - e.g. ``warehouse_identifier`` vs "Customer ID".
  A conflicted candidate can never be a KB match.
- ``head_conflict``: both column and concept have a role word (``id``,
  ``date``, ``amount`` ...) and none of the concept's variants match it -
  e.g. ``customer_type`` vs "Customer Name". A conflicted candidate is at
  most Ambiguous.
- ``profile_veto``: the observed data role (measure / identifier / flag / ...)
  is incompatible with the concept's family role - e.g. a float measure
  against an identifier concept. Caps the candidate below Probable.
"""

import json
import re
from difflib import SequenceMatcher

from app.config import (
    SEMANTIC_APPLICABLE_FEATURES,
    SEMANTIC_KB_MIN_EMBEDDING_SIMILARITY,
    SEMANTIC_MIN_UNIQUE_SAMPLE,
    SEMANTIC_PATTERN_MIN_COVERAGE,
    SEMANTIC_SCORING_WEIGHTS,
)
from app.models import SemanticConcept
from app.services.semantic_name import (
    NameParts,
    alias_similarity,
    expand_tokens,
    expanded_text,
    name_similarity,
    normalize_name,
    normalize_text,
    text_similarity,
    tokenize,
)

_TEMPORAL_DTYPE_TOKENS = ("datetime", "timestamp")

__all__ = [
    "alias_similarity",
    "calculate_deterministic_score",
    "collect_gates",
    "concept_info",
    "data_type_compatibility",
    "description_similarity",
    "evidence_coverage",
    "expand_tokens",
    "expanded_text",
    "feature_payload",
    "generate_features",
    "name_similarity",
    "normalize_name",
    "normalize_text",
    "open_discovery_decision",
    "profile_compatibility",
    "text_similarity",
    "tokenize",
]

# ---------------------------------------------------------------------------
# Concept metadata (parsed once per concept, reused across columns)
# ---------------------------------------------------------------------------


def _as_string_list(raw) -> list[str]:
    """Parse aliases/expected-types from a JSON string OR a list."""
    if isinstance(raw, (list, tuple)):
        return [str(value) for value in raw]
    try:
        loaded = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return []
    if isinstance(loaded, (list, tuple)):
        return [str(value) for value in loaded]
    return []


def concept_info(concept: SemanticConcept | dict) -> dict:
    """Parse a concept row (ORM or definition dict) into reusable metadata."""
    if isinstance(concept, dict):
        concept_name = concept["concept_name"]
        category = concept.get("category", "")
        description = concept.get("description", "")
        aliases_raw = concept.get("aliases") or "[]"
        expected_raw = concept.get("expected_data_types") or "[]"
        expectations_raw = concept.get("profile_expectations") or "{}"
    else:
        concept_name = concept.concept_name
        category = concept.category
        description = concept.description
        aliases_raw = concept.aliases or "[]"
        expected_raw = concept.expected_data_types or "[]"
        expectations_raw = concept.profile_expectations or "{}"

    aliases = _as_string_list(aliases_raw)
    expected_types = _as_string_list(expected_raw)

    if isinstance(expectations_raw, dict):
        expectations = dict(expectations_raw)
    else:
        try:
            expectations = dict(json.loads(expectations_raw))
        except (json.JSONDecodeError, TypeError):
            expectations = {}

    name_parts = normalize_name(concept_name)
    variants = [name_parts]
    for alias in aliases:
        variants.append(normalize_name(alias))

    if "entities" in expectations:
        # Explicit declaration is authoritative: an empty list means the
        # concept asserts NO entity (generic concepts such as Date), so no
        # entity conflict can ever fire against it. Derived alias entities
        # (e.g. "calendar" from the "calendar date" alias) would otherwise
        # pollute the pool and wrongly block qualified columns such as
        # "signup_date".
        entity_pool = {
            str(entity) for entity in expectations["entities"]
        }
    else:
        entity_pool = set()
        for variant in variants:
            entity_pool.update(variant.entity_tokens)

    token_pool: set[str] = set()
    for variant in variants:
        token_pool.update(variant.expanded_tokens)

    heads = {variant.head for variant in variants if variant.head}
    for head_token in expectations.get("head_tokens", []):
        heads.add(str(head_token))

    keys = {variant.key for variant in variants if variant.key}

    return {
        "concept_name": concept_name,
        "category": category,
        "description": description,
        "aliases": aliases,
        "expected_types": expected_types,
        "expectations": expectations,
        "name_parts": name_parts,
        "variants": variants,
        "entity_pool": entity_pool,
        "token_pool": token_pool,
        "heads": heads,
        "keys": keys,
        "is_family": bool(expectations.get("is_family", False)),
        "family": expectations.get("family"),
        "role": expectations.get("role"),
    }


# ---------------------------------------------------------------------------
# Name evidence (entity/head aware)
# ---------------------------------------------------------------------------

_HEAD_SYNONYMS: tuple[tuple[str, ...], ...] = (
    ("id", "identifier"),
    ("no", "number"),
    ("cd", "code"),
    ("percentage", "percent"),
    ("time", "timestamp"),
    ("rating", "score"),
)


def _heads_match(first: str | None, second: str | None) -> float:
    """1.0 equal/synonymous heads, 0.5 when one side has no head, else 0.0."""
    if first is None or second is None:
        return 0.5
    if first == second:
        return 1.0
    for group in _HEAD_SYNONYMS:
        if first in group and second in group:
            return 1.0
    return 0.0


def _is_head_synonym(first: str | None, second: str | None) -> bool:
    """True when the two role words are equal or documented synonyms."""
    if first is None or second is None:
        return False
    if first == second:
        return True
    return any(
        first in group and second in group for group in _HEAD_SYNONYMS
    )


def _entity_overlap(
    column_entities: list[str], variant_entities: list[str]
) -> float:
    """F1 overlap between entity tokens; neutral 0.5 when one side is empty."""
    if not column_entities and not variant_entities:
        return 0.6  # both purely role-based names ("amount" vs "Amount")
    if not column_entities or not variant_entities:
        return 0.5  # one side is a qualified role word, the other is not

    first = set(column_entities)
    second = set(variant_entities)
    intersection = len(first & second)
    if not intersection:
        return 0.0

    precision = intersection / len(first)
    recall = intersection / len(second)
    return 2 * precision * recall / (precision + recall)


#: Tokens that may legitimately appear in an entity position without being
#: an entity (boolean prefix "is", noise "field", generic "prefix").
_NON_ENTITY_TOKENS: frozenset[str] = frozenset(
    {"is", "has", "was", "prefix", "num", "no", "dt"}
)


from app.services.semantic_name import _is_known_word


def _column_has_known_token(name_parts: NameParts) -> bool:
    """True when at least one column token is a recognised word.

    Gibberish names (xyz_qq) deserve far weaker name evidence than real
    vocabulary: unrecognized tokens get no lexical credit.
    """
    for token in name_parts.expanded_tokens:
        if _is_known_word(token) or token in _HEAD_SYNONYM_TOKENS:
            return True
    return False


_HEAD_SYNONYM_TOKENS: frozenset[str] = frozenset(
    token for group in _HEAD_SYNONYMS for token in group
)


def name_evidence(name_parts: NameParts, info: dict) -> dict:
    """Entity/head-aware name similarity with conflict flags.

    Returns {value, entity_conflict, head_conflict, entity_overlap,
    head_match}. The value is the strongest evidence over the concept name
    and all aliases. Names whose tokens are ALL unrecognized (gibberish) are
    capped at a low value: lexical similarity must not manufacture evidence.
    """
    column_key = name_parts.key
    best_value = 0.0
    best_variant = None

    for variant in info["variants"]:
        if column_key and column_key == variant.key:
            best_value = 1.0
            best_variant = variant
            break

        # Contained column name (e.g. "amount" inside "tax amount") is
        # weaker than a full match.
        containment = 0.0
        if column_key and variant.key and column_key in variant.key:
            containment = 0.5

        overlap = _entity_overlap(
            name_parts.entity_tokens, variant.entity_tokens
        )
        head = _heads_match(name_parts.head, variant.head)
        value = 0.6 * overlap + 0.4 * head

        # Legacy sequence similarity keeps partial-lexical names competitive
        # (e.g. "cust_nm" readings) without allowing unrelated tokens to win.
        # When the column's LAST (most specific) token is not part of the
        # concept variant at all, the sequence path is halved: shared-entity
        # names with a different specific token ("customer city" vs
        # "customer name") must not read as name evidence.
        sequence = SequenceMatcher(None, column_key, variant.key).ratio()
        if column_key and variant.key:
            column_last = column_key.split()[-1]
            if column_last not in variant.key.split():
                sequence *= 0.5
        value = max(value, 0.6 * _token_f1_sets(
            set(name_parts.expanded_tokens), set(variant.expanded_tokens)
        ) + 0.4 * sequence)

        value = max(value, containment)

        # Head-position coverage: a SPECIFIC concept must account for the
        # column's LAST content token ("customer city" ends in "city" so
        # City covers it while Customer Name does not; "customer state" is
        # covered by the state concept, not the name concept). Families keep
        # the full value — their whole purpose is generic fallback.
        if not info.get("is_family") and name_parts.expanded_tokens:
            variant_tokens = set(variant.expanded_tokens) | set(
                variant.entity_tokens
            )
            if name_parts.expanded_tokens[-1] not in variant_tokens:
                value = min(value, 0.4)

        if value > best_value:
            best_value = value
            best_variant = variant

    if best_value < 1.0 and not _column_has_known_token(name_parts):
        best_value = min(best_value, 0.2)

    # --- gates -----------------------------------------------------------
    def _column_entities() -> list[str]:
        return [
            token
            for token in name_parts.entity_tokens
            if token not in _NON_ENTITY_TOKENS and _is_known_word(token)
        ]

    concept_entities = {
        entity
        for entity in info["entity_pool"]
        if entity not in _NON_ENTITY_TOKENS and _is_known_word(entity)
    }
    column_entity_tokens = _column_entities()
    entity_conflict = bool(
        column_entity_tokens
        and concept_entities
        and not (set(column_entity_tokens) & concept_entities)
    )

    head_match_value = 0.0
    if best_variant is not None:
        head_match_value = _heads_match(
            name_parts.head, best_variant.head
        )
    elif info["heads"]:
        head_match_value = max(
            (
                _heads_match(name_parts.head, head)
                for head in info["heads"]
            ),
            default=0.0,
        )

    # Head conflict is decided against the CONCEPT-level head set (not the
    # best variant): a column whose role word (id/code/date/...) disagrees
    # with every head the concept declares is structurally incompatible even
    # when an entity token overlaps (review_id vs Review Text). Variants
    # without their own head do not neutralize this check.
    head_conflict = bool(
        name_parts.head is not None
        and info["heads"]
        and not any(
            _is_head_synonym(name_parts.head, head)
            or _heads_match(name_parts.head, head) == 1.0
            for head in info["heads"]
        )
    )

    return {
        "value": round(min(1.0, best_value), 4),
        "entity_conflict": entity_conflict,
        "head_conflict": head_conflict,
        "entity_overlap": round(
            _entity_overlap(
                name_parts.entity_tokens,
                best_variant.entity_tokens if best_variant else [],
            ),
            4,
        ),
        "head_match": round(head_match_value, 4),
    }


def _token_f1_sets(first: set[str], second: set[str]) -> float:
    if not first or not second:
        return 0.0
    intersection = len(first & second)
    if not intersection:
        return 0.0
    precision = intersection / len(first)
    recall = intersection / len(second)
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# Description / datatype evidence
# ---------------------------------------------------------------------------


def description_similarity(
    column_description: str | None,
    concept: SemanticConcept,
    info: dict | None = None,
) -> tuple[float, bool]:
    """Compare an incoming column description with the concept text.

    Returns (value, applicable). Without a column description the evidence
    simply does not exist and is excluded from scoring. When a description
    EXISTS it is scored against the concept description, name AND aliases
    (strongest signal wins).
    """
    if not column_description:
        return 0.0, False

    if info is None:
        info = concept_info(concept)

    best = max(
        text_similarity(column_description, info["description"]),
        text_similarity(column_description, info["concept_name"]),
    )
    for alias in info["aliases"]:
        best = max(best, text_similarity(column_description, alias))

    return round(best, 4), True


def normalize_data_type(data_type: str | None) -> str:
    """Normalize common dataframe data type names."""
    if not data_type:
        return ""

    value = data_type.lower()

    if any(token in value for token in ("int", "integer", "long")):
        return "int"

    if any(
        token in value
        for token in ("float", "double", "decimal", "numeric", "real")
    ):
        return "float"

    if any(token in value for token in _TEMPORAL_DTYPE_TOKENS):
        return "datetime"

    if value == "date":
        return "date"

    if any(token in value for token in ("bool", "boolean")):
        return "bool"

    if any(token in value for token in ("object", "string", "str", "text")):
        return "string"

    return value


def _actual_data_type(profile: dict, data_type: str | None) -> str:
    """Resolve the effective datatype, honouring Stage 03 datetime detection.

    A string column that profiling detected as strongly date-like
    (>= 95% parseable ISO dates) is evidence of a date/datetime column.
    """
    normalized = normalize_data_type(data_type)

    datetime_profile = profile.get("datetime") if isinstance(profile, dict) else None
    if (
        normalized in ("string", "object", "")
        and isinstance(datetime_profile, dict)
        and float(datetime_profile.get("format_valid_percentage") or 0) >= 95.0
    ):
        return "datetime"

    return normalized


def data_type_compatibility(
    data_type: str | None,
    concept: SemanticConcept,
    profile: dict | None = None,
    info: dict | None = None,
) -> tuple[float, bool]:
    """Compare the incoming column datatype with the concept's expected types.

    Returns (value, applicable). An incompatible but computable comparison is
    applicable evidence with value 0.0 (evidence against); a concept without
    expected types, or an unknown column datatype, is not applicable.
    """
    if info is None:
        info = concept_info(concept)

    expected_types = info["expected_types"]
    if not expected_types:
        return 0.0, False

    actual = _actual_data_type(profile or {}, data_type)

    if not actual:
        return 0.0, False

    normalized_expected = {
        normalize_data_type(value) for value in expected_types
    }

    if actual in normalized_expected:
        return 1.0, True

    # Logically impossible combinations count as applicable negative evidence:
    # numeric data cannot be free text; datetime data cannot be a number.
    if actual in ("int", "float") and normalized_expected <= {"string"}:
        return 0.0, True

    if actual == "datetime" and normalized_expected <= {"int", "float"}:
        return 0.0, True

    # Comparable but simply not matching (e.g. string vs identifiers): weak
    # negative evidence, still applicable.
    return 0.0, True


# ---------------------------------------------------------------------------
# Observed profile role + veto
# ---------------------------------------------------------------------------


def observed_profile_role(profile: dict) -> set[str]:
    """Plausible structural roles for the OBSERVED column data.

    Returns a SET: integer columns can legitimately be counts or measures,
    so role compatibility is an intersection test, not an equality test.
    """
    roles: set[str] = set()
    data_type = str(profile.get("data_type", "")).lower()

    numeric_profile = profile.get("numeric") or {}
    datetime_profile = profile.get("datetime") or {}
    text_profile = profile.get("text") or {}

    is_numeric = data_type in {
        "int",
        "int64",
        "int32",
        "float",
        "float64",
        "float32",
        "decimal",
        "numeric",
    } or bool(numeric_profile)
    is_text = data_type in {"str", "string", "object", "text"} or bool(
        text_profile
    )
    is_temporal = data_type in {
        "date",
        "datetime",
        "datetime64",
        "datetime64[ns]",
        "timestamp",
    } or bool(datetime_profile)

    if is_numeric:
        integer_valued = bool(
            numeric_profile.get("integer_valued", data_type.startswith("int"))
        )
        code_like = bool(numeric_profile.get("code_like", False))
        counter_like = bool(
            (numeric_profile.get("integer_sequence") or {}).get(
                "counter_like", False
            )
        )
        identifier_like = bool(profile.get("identifier_like", False)) or bool(
            profile.get("identifier_signal", False)
        )
        boolean_like = bool(numeric_profile.get("constant", False)) or bool(
            numeric_profile.get("near_constant", False)
        )

        if code_like or identifier_like:
            roles.update({"identifier", "code"})
        if counter_like:
            roles.update({"identifier", "code", "count"})
        if integer_valued:
            roles.update({"count", "measure"})
        else:
            roles.add("measure")
        if boolean_like:
            # A constant/near-constant numeric column may be a flag.
            roles.add("flag")
        if not roles:
            roles.update({"count", "measure"})

    if data_type in {"bool", "boolean"}:
        roles.add("flag")

    if is_text:
        distinct_percentage = profile.get("distinct_percentage")
        mean_length = text_profile.get("mean_length")
        if mean_length is not None and float(mean_length) > 64:
            roles.add("text")
        elif distinct_percentage is not None and float(distinct_percentage) < 50.0:
            roles.update({"categorical", "text"})
        else:
            roles.update({"text", "categorical"})
        if profile.get("identifier_like") or profile.get("identifier_signal"):
            roles.update({"identifier", "code"})

    if is_temporal:
        has_time = bool(datetime_profile.get("has_time_component", False))
        roles.add("time" if has_time else "date")

    if not roles:
        # All-null or unknown-shape columns carry no role evidence; nothing
        # may be vetoed on this basis.
        roles.update(
            {
                "identifier",
                "code",
                "measure",
                "count",
                "flag",
                "text",
                "categorical",
                "date",
                "time",
            }
        )

    return roles


_ROLE_COMPATIBILITY: dict[str, set[str]] = {
    "identifier": {"identifier", "code", "count", "text"},
    "code": {"code", "identifier", "categorical", "text"},
    "measure": {"measure", "count", "flag"},
    "count": {"count", "measure", "identifier", "code"},
    "flag": {"flag", "categorical", "count", "code", "identifier"},
    # An observed identifier-shaped column (unique/patterned values) is
    # structurally incompatible with free-text/categorical concepts: real
    # names/categories repeat, key columns do not.
    "text": {"text", "categorical"},
    "categorical": {"categorical", "text", "flag"},
    "date": {"date", "time"},
    "time": {"time", "date"},
    "other": {
        "identifier",
        "code",
        "measure",
        "count",
        "flag",
        "text",
        "categorical",
        "date",
        "time",
    },
}


def role_veto(profile: dict, info: dict) -> tuple[bool, set[str], str | None]:
    """Profile-as-VETO: observed role vs the concept's family role.

    Returns (veto, observed_roles, concept_role). Incompatible roles (e.g. a
    float measure vs an identifier concept) veto the candidate; unknown
    roles never veto.
    """
    concept_role = info.get("role")
    if not concept_role:
        return False, set(), None

    observed = observed_profile_role(profile or {})
    allowed = _ROLE_COMPATIBILITY.get(str(concept_role), set())

    if not allowed:
        return False, observed, str(concept_role)

    veto = not bool(observed & allowed)
    return veto, observed, str(concept_role)


# ---------------------------------------------------------------------------
# Profile compatibility (coverage-based checks)
# ---------------------------------------------------------------------------


def _non_null_count(profile: dict) -> int:
    if profile.get("non_null_count") is not None:
        return int(profile.get("non_null_count"))
    row_count = profile.get("row_count", 0) or 0
    return int(row_count) - int(profile.get("null_count") or 0)


def _profile_checks(
    profile: dict,
    expectations: dict,
    column_name: str | None,
    column_description: str | None,
    concept: SemanticConcept,
) -> list[dict]:
    """Evaluate one evidence check per applicable profile expectation.

    Each entry is {expectation, actual, matched}. Checks evaluate OBSERVED
    profile facts only. None of them decides a mapping by itself. Metadata
    keys inside profile_expectations (family, rule_family, value_vocabulary,
    ...) are ignored here.
    """
    checks: list[dict] = []

    data_type = str(profile.get("data_type", "")).lower()
    null_percentage = profile.get("null_percentage")
    distinct_percentage = profile.get("distinct_percentage")
    non_null_count = _non_null_count(profile)

    identifier_name_signal = bool(profile.get("identifier_name_signal", False))
    identifier_signal = bool(profile.get("identifier_signal", False))

    text_profile = profile.get("text", {}) or {}
    patterns = text_profile.get("patterns", {}) or {}
    numeric_profile = profile.get("numeric", {}) or {}
    datetime_profile = profile.get("datetime", {}) or {}
    shape_profile = profile.get("shape", {}) or text_profile.get("shape", {}) or {}

    is_numeric = data_type in {
        "int",
        "int64",
        "int32",
        "float",
        "float64",
        "float32",
        "decimal",
        "numeric",
    } or bool(numeric_profile)
    is_text = data_type in {"str", "string", "object", "text"} or bool(
        text_profile
    )
    is_temporal = data_type in {
        "date",
        "datetime",
        "datetime64",
        "datetime64[ns]",
        "timestamp",
    } or bool(datetime_profile)

    def add(expectation: str, actual: bool, expected: bool) -> None:
        checks.append(
            {
                "expectation": expectation,
                "actual": actual,
                "matched": actual == expected,
            }
        )

    if "identifier_like" in expectations:
        # Identifier evidence requires the identifier NAME/PROFILE signal from
        # Stage 03. A naturally-unique measure (e.g. revenue on 5 rows) is not
        # an identifier; on very small samples uniqueness is not meaningful.
        actual_identifier_like = bool(
            profile.get("identifier_like", False)
            or profile.get("identifier_repeats", False)
            or identifier_name_signal
            or identifier_signal
        )
        if not actual_identifier_like:
            unique_profile = (
                distinct_percentage is not None
                and float(distinct_percentage) >= 99.0
                and non_null_count >= SEMANTIC_MIN_UNIQUE_SAMPLE
            )
            complete_profile = (
                null_percentage is not None
                and float(null_percentage) == 0.0
            )
            semantic_evidence = max(
                name_similarity(column_name or "", concept),
                text_similarity(column_description or "", concept.description),
            ) >= 0.50
            actual_identifier_like = (
                unique_profile and complete_profile and semantic_evidence
            )

        add(
            "identifier_like",
            actual_identifier_like,
            bool(expectations["identifier_like"]),
        )

    if "unique" in expectations:
        # Informational only (v2): uniqueness belongs to an approved primary
        # key, never to every identifier concept. Never a failed check.
        actual_unique = bool(
            distinct_percentage is not None
            and float(distinct_percentage) >= 99.0
            and non_null_count >= SEMANTIC_MIN_UNIQUE_SAMPLE
        )
        checks.append(
            {
                "expectation": "uniqueness_hint",
                "actual": actual_unique,
                "matched": True,
                "informational": True,
            }
        )

    if "non_null" in expectations:
        actual_non_null = bool(
            null_percentage is not None and float(null_percentage) == 0.0
        )
        add("non_null", actual_non_null, bool(expectations["non_null"]))

    if "numeric" in expectations:
        add("numeric", is_numeric, bool(expectations["numeric"]))

    if "text" in expectations:
        add("text", is_text, bool(expectations["text"]))

    if "temporal" in expectations:
        add("temporal", is_temporal, bool(expectations["temporal"]))

    if "categorical" in expectations:
        actual_categorical = bool(
            is_text
            and distinct_percentage is not None
            and float(distinct_percentage) < 50.0
        )
        add("categorical", actual_categorical, bool(expectations["categorical"]))

    if "non_negative" in expectations:
        actual_non_negative = False
        if numeric_profile:
            minimum = numeric_profile.get("min")
            if minimum is not None:
                actual_non_negative = float(minimum) >= 0
        add("non_negative", actual_non_negative, bool(expectations["non_negative"]))

    if "continuous" in expectations:
        actual_continuous = bool(
            is_numeric
            and distinct_percentage is not None
            and float(distinct_percentage) >= 50.0
        )
        add("continuous", actual_continuous, bool(expectations["continuous"]))

    if "bounded" in expectations:
        actual_bounded = False
        if numeric_profile:
            minimum = numeric_profile.get("min")
            maximum = numeric_profile.get("max")
            if minimum is not None and maximum is not None:
                actual_bounded = float(minimum) >= 0 and float(maximum) <= 150
        add("bounded", actual_bounded, bool(expectations["bounded"]))

    if "integer_valued" in expectations:
        actual_integer = bool(
            numeric_profile.get("integer_valued", data_type.startswith("int"))
            if numeric_profile
            else data_type.startswith("int")
        )
        add("integer_valued", actual_integer, bool(expectations["integer_valued"]))

    if "boolean_like" in expectations:
        actual_boolean = data_type in {"bool", "boolean"}
        if not actual_boolean:
            values = _observed_values(profile)
            if values:
                lowered = {str(value).strip().lower() for value in values}
                actual_boolean = lowered <= {
                    "0",
                    "1",
                    "true",
                    "false",
                    "yes",
                    "no",
                    "y",
                    "n",
                }
        if not actual_boolean and numeric_profile:
            # Integer-encoded flag: the numeric block's observed DISTINCT
            # values are the vocabulary (categorical top-values only cover
            # low-cardinality TEXT columns in Stage 03).
            observed = numeric_profile.get("observed_values") or []
            if not observed:
                distinct_count = profile.get("distinct_count")
                if distinct_count is not None and int(distinct_count) <= 3:
                    actual_boolean = True  # 0/1/(2) columns: two states
        add("boolean_like", actual_boolean, bool(expectations["boolean_like"]))

    if "year" in expectations:
        actual_year = False
        if numeric_profile:
            minimum = numeric_profile.get("min")
            maximum = numeric_profile.get("max")
            if minimum is not None and maximum is not None:
                actual_year = (
                    1900 <= float(minimum) <= 2100
                    and 1900 <= float(maximum) <= 2100
                )
        add("year", actual_year, bool(expectations["year"]))

    if "long_text" in expectations:
        actual_long_text = False
        max_length = text_profile.get("max_length")
        mean_length = text_profile.get("mean_length")
        if max_length is not None:
            actual_long_text = float(max_length) >= 100
        elif mean_length is not None:
            actual_long_text = float(mean_length) >= 50
        add("long_text", actual_long_text, bool(expectations["long_text"]))

    # ------------------------------------------------------------------
    # Pattern expectations: coverage of NON-NULL rows, never "count > 0".
    # ------------------------------------------------------------------
    pattern_expectations = {
        "email_pattern": "email_like",
        "phone_pattern": "phone_like",
        "postal_pattern": "postal_like",
        "currency_code": "currency_like",
    }

    for expectation_key, profile_key in pattern_expectations.items():
        if expectation_key not in expectations:
            continue
        pattern_count = int(patterns.get(profile_key, 0) or 0)
        coverage = pattern_count / non_null_count if non_null_count else 0.0
        add(
            expectation_key,
            coverage >= SEMANTIC_PATTERN_MIN_COVERAGE,
            bool(expectations[expectation_key]),
        )

    if "url_pattern" in expectations:
        values = _observed_values(profile)
        if values:
            url_like = [
                value
                for value in values
                if "://" in str(value) or str(value).startswith("/")
            ]
            coverage = len(url_like) / len(values)
        else:
            coverage = 0.0
        add(
            "url_pattern",
            coverage >= SEMANTIC_PATTERN_MIN_COVERAGE,
            bool(expectations["url_pattern"]),
        )

    if "time_of_day" in expectations:
        actual_time_of_day = False
        if numeric_profile:
            minimum = numeric_profile.get("min")
            maximum = numeric_profile.get("max")
            if minimum is not None and maximum is not None:
                actual_time_of_day = (
                    0 <= float(minimum) <= 23 and 0 <= float(maximum) <= 23
                )
        if not actual_time_of_day and bool(datetime_profile):
            actual_time_of_day = bool(datetime_profile.get("has_time_component", False))
        if not actual_time_of_day:
            values = _observed_values(profile)
            if values:
                time_like = sum(
                    1
                    for value in values
                    if re.fullmatch(r"\d{1,2}:\d{2}(:\d{2})?", str(value).strip())
                )
                actual_time_of_day = time_like / len(values) >= SEMANTIC_PATTERN_MIN_COVERAGE
        add(
            "time_of_day",
            actual_time_of_day,
            bool(expectations["time_of_day"]),
        )

    if "domain_range" in expectations:
        actual_in_range = False
        declared = expectations["domain_range"]
        if (
            isinstance(declared, (list, tuple))
            and len(declared) == 2
            and numeric_profile
        ):
            minimum = numeric_profile.get("min")
            maximum = numeric_profile.get("max")
            if minimum is not None and maximum is not None:
                actual_in_range = (
                    float(minimum) >= float(declared[0])
                    and float(maximum) <= float(declared[1])
                )
        add("domain_range", actual_in_range, True)

    return checks


def _observed_values(profile: dict, limit: int = 50) -> list:
    """Observed top values from the categorical profile block (if any)."""
    categorical = profile.get("categorical") or {}
    top_values = categorical.get("top_values") or []
    values = [
        entry.get("value")
        for entry in top_values[:limit]
        if isinstance(entry, dict) and entry.get("value") is not None
    ]
    return values


def profile_compatibility(
    profile: dict | None,
    concept: SemanticConcept,
    column_name: str | None = None,
    column_description: str | None = None,
    info: dict | None = None,
) -> tuple[float, bool, dict]:
    """Fraction of the concept's applicable profile checks that the profile passes.

    Returns (value, applicable, payload). A concept without profile
    expectations is not applicable (excluded from scoring); a concept WITH
    expectations always yields applicable evidence, even when checks fail.
    The payload carries the profile veto flag and observed role evidence.
    """
    if not profile:
        return 0.0, False, {}

    if info is None:
        info = concept_info(concept)

    expectations = info["expectations"]
    if not expectations:
        return 0.0, False, {}

    checks = _profile_checks(
        profile,
        expectations,
        column_name,
        column_description,
        concept,
    )

    if not checks:
        return 0.0, False, {}

    veto, observed_roles, concept_role = role_veto(profile, info)

    matches = sum(1 for check in checks if check["matched"])

    payload = {
        "profile_veto": veto,
        "observed_roles": sorted(observed_roles),
        "concept_role": concept_role,
        "checks": checks,
    }

    return round(matches / len(checks), 4), True, payload


# ---------------------------------------------------------------------------
# Value evidence (NEW)
# ---------------------------------------------------------------------------


def value_evidence(
    profile: dict | None,
    concept: SemanticConcept,
    info: dict | None = None,
) -> tuple[float, bool]:
    """Observed values vs the concept's declared value vocabulary/regex.

    Returns (value, applicable). Applicable only when the concept declares a
    value vocabulary or value regex AND the profile carries observed top
    values (categorical block). The value is the share of observed rows whose
    value matches - high share SUPPORTS the concept, low share counts
    against it.
    """
    if not profile:
        return 0.0, False

    if info is None:
        info = concept_info(concept)

    expectations = info["expectations"]
    vocabulary = expectations.get("value_vocabulary")
    value_regex = expectations.get("value_regex")

    if not vocabulary and not value_regex:
        return 0.0, False

    values = _observed_values(profile)
    if not values:
        return 0.0, False

    matched = 0
    for value in values:
        text = str(value).strip().lower()
        if vocabulary and text in {str(entry).strip().lower() for entry in vocabulary}:
            matched += 1
        elif value_regex and re.fullmatch(value_regex, str(value).strip()):
            matched += 1

    coverage = matched / len(values)
    return round(coverage, 4), True


# ---------------------------------------------------------------------------
# Context evidence (NEW: siblings + domain keywords)
# ---------------------------------------------------------------------------


def context_similarity(
    dataset_domain: str | None,
    table_context: str | None,
    concept: SemanticConcept,
    sibling_names: list[str] | None = None,
    info: dict | None = None,
) -> tuple[float, bool]:
    """Contextual compatibility: siblings vs context keywords, domain vs
    domain keywords; falls back to the legacy category similarity when the
    concept declares no keyword metadata.

    Returns (value, applicable). Without any context the evidence does not
    exist and is excluded from scoring.
    """
    if info is None:
        info = concept_info(concept)

    expectations = info["expectations"]
    context_keywords = expectations.get("context_keywords") or []
    domain_keywords = expectations.get("domain_keywords") or []

    parts: list[float] = []

    if context_keywords and sibling_names:
        keyword_set = {
            token
            for keyword in context_keywords
            for token in str(keyword).lower().split()
        }
        sibling_tokens: set[str] = set()
        for sibling in sibling_names[:24]:
            sibling_tokens.update(normalize_name(str(sibling)).expanded_tokens)
        if sibling_tokens:
            parts.append(_token_f1_sets(sibling_tokens, set(context_keywords)))

    if domain_keywords and dataset_domain:
        domain_tokens = set(normalize_name(dataset_domain).expanded_tokens)
        keyword_set = {
            token
            for keyword in domain_keywords
            for token in str(keyword).lower().split()
        }
        if domain_tokens and keyword_set:
            parts.append(_token_f1_sets(domain_tokens, keyword_set))

    if parts:
        return round(max(parts), 4), True

    # Legacy fallback: string similarity between configured context text and
    # the concept category (kept for backward compatibility).
    context_parts = [
        dataset_domain or "",
        table_context or "",
    ]
    context_text = " ".join(part for part in context_parts if part)

    if not context_text:
        return 0.0, False

    return text_similarity(context_text, concept.category), True


# ---------------------------------------------------------------------------
# Relationship evidence (NEW, optional second pass)
# ---------------------------------------------------------------------------


def relationship_evidence(
    relationship_hints: list[str] | None,
    concept: SemanticConcept,
    info: dict | None = None,
) -> tuple[float, bool]:
    """Approved-relationship support: a parent column's confirmed concept
    agrees with this candidate.

    ``relationship_hints`` carries the confirmed concept names of the
    approved parent columns for THIS column (built by the dataset-level
    service from the read-only relationship store). None = no relationship
    data was collected (feature not applicable); an empty list means the
    column HAS approved relationships but none carry a confirmed concept.
    """
    if relationship_hints is None:
        return 0.0, False

    if info is None:
        info = concept_info(concept)

    if not relationship_hints:
        return 0.0, True

    if info["concept_name"] in relationship_hints:
        return 1.0, True

    return 0.0, True


# ---------------------------------------------------------------------------
# Feature bundle
# ---------------------------------------------------------------------------


def generate_features(
    column_name: str,
    concept: SemanticConcept,
    embedding_similarity: float,
    column_description: str | None = None,
    data_type: str | None = None,
    profile: dict | None = None,
    dataset_domain: str | None = None,
    table_context: str | None = None,
    sibling_names: list[str] | None = None,
    relationship_hints: list[str] | None = None,
    name_parts: NameParts | None = None,
    concept_data: dict | None = None,
) -> dict:
    """Generate the complete evidence bundle for one column/concept candidate.

    Every feature carries {value, applicable} so consumers (score, coverage,
    UI) can distinguish "no evidence" from "evidence against". Optional
    errors in single features are recorded as "<feature>_error" and never
    abort the bundle.
    """
    info = concept_data if concept_data is not None else concept_info(concept)
    parts = name_parts if name_parts is not None else normalize_name(column_name)

    features: dict = {
        "embedding_similarity": {
            "value": float(embedding_similarity),
            "applicable": True,
        },
    }

    try:
        evidence = name_evidence(parts, info)
        features["name_similarity"] = {
            "value": float(evidence["value"]),
            "applicable": True,
            "payload": {
                "entity_conflict": evidence["entity_conflict"],
                "head_conflict": evidence["head_conflict"],
                "entity_overlap": evidence["entity_overlap"],
                "head_match": evidence["head_match"],
            },
        }
    except Exception as error:  # pragma: no cover - defensive
        features["name_similarity"] = {"value": 0.0, "applicable": True}
        features["name_similarity_error"] = {"value": type(error).__name__, "applicable": True}

    try:
        description_value, description_applicable = description_similarity(
            column_description, concept, info=info
        )
        features["description_similarity"] = {
            "value": float(description_value),
            "applicable": description_applicable,
        }
    except Exception as error:  # pragma: no cover - defensive
        features["description_similarity"] = {"value": 0.0, "applicable": False}
        features["description_similarity_error"] = {"value": type(error).__name__, "applicable": True}

    try:
        datatype_value, datatype_applicable = data_type_compatibility(
            data_type, concept, profile=profile, info=info
        )
        features["datatype_compatibility"] = {
            "value": float(datatype_value),
            "applicable": datatype_applicable,
        }
    except Exception as error:  # pragma: no cover - defensive
        features["datatype_compatibility"] = {"value": 0.0, "applicable": False}
        features["datatype_compatibility_error"] = {"value": type(error).__name__, "applicable": True}

    try:
        profile_value, profile_applicable, profile_payload = profile_compatibility(
            profile, concept, column_name, column_description, info=info
        )
        features["profile_compatibility"] = {
            "value": float(profile_value),
            "applicable": profile_applicable,
            "payload": profile_payload,
        }
    except Exception as error:  # pragma: no cover - defensive
        features["profile_compatibility"] = {"value": 0.0, "applicable": False}
        features["profile_compatibility_error"] = {"value": type(error).__name__, "applicable": True}

    try:
        value_value, value_applicable = value_evidence(profile, concept, info=info)
        features["value_evidence"] = {
            "value": float(value_value),
            "applicable": value_applicable,
        }
    except Exception as error:  # pragma: no cover - defensive
        features["value_evidence"] = {"value": 0.0, "applicable": False}
        features["value_evidence_error"] = {"value": type(error).__name__, "applicable": True}

    try:
        context_value, context_applicable = context_similarity(
            dataset_domain,
            table_context,
            concept,
            sibling_names=sibling_names,
            info=info,
        )
        features["context_similarity"] = {
            "value": float(context_value),
            "applicable": context_applicable,
        }
    except Exception as error:  # pragma: no cover - defensive
        features["context_similarity"] = {"value": 0.0, "applicable": False}
        features["context_similarity_error"] = {"value": type(error).__name__, "applicable": True}

    try:
        relationship_value, relationship_applicable = relationship_evidence(
            relationship_hints, concept, info=info
        )
        features["relationship_evidence"] = {
            "value": float(relationship_value),
            "applicable": relationship_applicable,
        }
    except Exception as error:  # pragma: no cover - defensive
        features["relationship_evidence"] = {"value": 0.0, "applicable": False}
        features["relationship_evidence_error"] = {"value": type(error).__name__, "applicable": True}

    return features


def calculate_deterministic_score(features: dict) -> float:
    """Calculate the deterministic semantic score.

    The score is the weighted mean of APPLICABLE evidence features (weights
    from app.config.SEMANTIC_SCORING_WEIGHTS). Non-applicable evidence is
    excluded from both numerator and denominator, so a column without a
    description or dataset context is not penalized for evidence that was
    never available. The result is a reproducible weighted evidence score,
    NOT a calibrated probability.
    """
    numerator = 0.0
    denominator = 0.0

    for feature, weight in SEMANTIC_SCORING_WEIGHTS.items():
        evidence = features.get(feature)

        if evidence is None or not evidence.get("applicable", False):
            continue

        value = float(evidence.get("value", 0.0))
        numerator += weight * value
        denominator += weight

    if denominator <= 0:
        return 0.0

    return round(max(0.0, min(1.0, numerator / denominator)), 4)


def evidence_coverage(features: dict) -> float:
    """Fraction of applicable evidence that actually carries signal (value > 0).

    Only applicable features count. "No description configured" does not
    reduce coverage; "description exists but is dissimilar" does.
    """
    applicable = [
        features[feature]
        for feature in SEMANTIC_APPLICABLE_FEATURES
        if feature in features and features[feature].get("applicable", False)
    ]

    if not applicable:
        return 0.0

    active = sum(
        1 for evidence in applicable if float(evidence.get("value", 0.0)) > 0
    )

    return active / len(applicable)


def collect_gates(features: dict) -> dict:
    """Extract the gate flags from an evidence bundle (defaults False)."""
    name_evidence_entry = features.get("name_similarity") or {}
    profile_entry = features.get("profile_compatibility") or {}

    name_payload = name_evidence_entry.get("payload") or {}
    profile_payload = profile_entry.get("payload") or {}

    return {
        "entity_conflict": bool(name_payload.get("entity_conflict", False)),
        "head_conflict": bool(name_payload.get("head_conflict", False)),
        "profile_veto": bool(profile_payload.get("profile_veto", False)),
    }


def open_discovery_decision(features: dict) -> bool:
    """Decide whether the best candidate is a suitable KB match or Open Discovery.

    Open Discovery is a FALLBACK, never the default: the KB gate opens only
    when the best candidate shows a genuinely weak embedding similarity AND
    weak lexical name evidence, OR when a structural gate (entity conflict /
    profile veto) fired. This keeps obvious matches (customer_id ->
    Customer ID) as KB matches even when the sentence encoder is
    conservative, while gibberish columns (xyz_qq) are not forced into the
    closest concept.
    """
    embedding_evidence = features.get("embedding_similarity") or {}
    name_evidence_entry = features.get("name_similarity") or {}

    embedding_value = float(embedding_evidence.get("value", 0.0))
    name_value = float(name_evidence_entry.get("value", 0.0))

    gates = collect_gates(features)

    weak_embedding = embedding_value < SEMANTIC_KB_MIN_EMBEDDING_SIMILARITY
    weak_name = name_value < SEMANTIC_KB_MIN_EMBEDDING_SIMILARITY

    if gates["entity_conflict"] or gates["profile_veto"]:
        return True

    return weak_embedding and weak_name


def feature_payload(features: dict) -> dict:
    """Reduce an evidence bundle to plain values for persistence/UI.

    Gate flags and per-feature payloads (entity/head conflicts, profile
    veto, observed roles, checks) travel with the payload; error markers
    from optional features are preserved as-is.
    """
    payload: dict = {}

    for feature, evidence in features.items():
        if feature.endswith("_error"):
            payload[feature] = {"value": evidence.get("value"), "applicable": True}
            continue

        entry = {
            "value": round(float(evidence.get("value", 0.0)), 4),
            "applicable": bool(evidence.get("applicable", False)),
        }

        nested = evidence.get("payload")
        if isinstance(nested, dict):
            entry["payload"] = nested

        payload[feature] = entry

    return payload

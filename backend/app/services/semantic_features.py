import json
import re
from difflib import SequenceMatcher

from app.models import SemanticConcept


def normalize_text(value: str | None) -> str:
    """
    Normalize text for similarity comparison.
    """

    if not value:
        return ""

    value = value.lower().strip()
    value = re.sub(r"[_\-]+", " ", value)
    value = re.sub(r"\s+", " ", value)

    return value


def tokenize(value: str | None) -> set[str]:
    """
    Convert text into normalized tokens.
    """

    normalized = normalize_text(value)

    if not normalized:
        return set()

    return set(normalized.split())


def text_similarity(
    first: str | None,
    second: str | None,
) -> float:
    """
    Calculate a simple text similarity score.

    Combines token overlap and sequence similarity.
    """

    first_normalized = normalize_text(first)
    second_normalized = normalize_text(second)

    if not first_normalized or not second_normalized:
        return 0.0

    first_tokens = tokenize(first_normalized)
    second_tokens = tokenize(second_normalized)

    if first_tokens and second_tokens:
        intersection = len(
            first_tokens.intersection(second_tokens)
        )
        union = len(
            first_tokens.union(second_tokens)
        )

        token_similarity = (
            intersection / union
            if union
            else 0.0
        )
    else:
        token_similarity = 0.0

    sequence_similarity = SequenceMatcher(
        None,
        first_normalized,
        second_normalized,
    ).ratio()

    return (
        0.6 * token_similarity
        + 0.4 * sequence_similarity
    )


def alias_similarity(
    column_name: str,
    aliases: list[str],
) -> float:
    """
    Compare a column name against all concept aliases
    and return the strongest similarity.
    """

    if not aliases:
        return 0.0

    return max(
        text_similarity(column_name, alias)
        for alias in aliases
    )


def name_similarity(
    column_name: str,
    concept: SemanticConcept,
) -> float:
    """
    Calculate name similarity using the concept name
    and its approved aliases.
    """

    concept_score = text_similarity(
        column_name,
        concept.concept_name,
    )

    aliases = []

    if concept.aliases:
        try:
            aliases = json.loads(concept.aliases)
        except json.JSONDecodeError:
            aliases = []

    alias_score = alias_similarity(
        column_name,
        aliases,
    )

    return max(
        concept_score,
        alias_score,
    )


def description_similarity(
    column_description: str | None,
    concept: SemanticConcept,
) -> float:
    """
    Compare the incoming column description with
    the Semantic KB concept description.
    """

    if not column_description:
        return 0.0

    return text_similarity(
        column_description,
        concept.description,
    )


def normalize_data_type(data_type: str | None) -> str:
    """
    Normalize common dataframe data type names.
    """

    if not data_type:
        return ""

    value = data_type.lower()

    if any(
        token in value
        for token in [
            "int",
            "integer",
            "long",
        ]
    ):
        return "int"

    if any(
        token in value
        for token in [
            "float",
            "double",
            "decimal",
            "numeric",
        ]
    ):
        return "float"

    if any(
        token in value
        for token in [
            "datetime",
            "timestamp",
        ]
    ):
        return "datetime"

    if value == "date":
        return "date"

    if any(
        token in value
        for token in [
            "bool",
            "boolean",
        ]
    ):
        return "bool"

    if any(
        token in value
        for token in [
            "object",
            "string",
            "str",
            "text",
        ]
    ):
        return "string"

    return value


def data_type_compatibility(
    data_type: str | None,
    concept: SemanticConcept,
) -> float:
    """
    Compare incoming column data type with the concept's
    expected data types.
    """

    if not data_type or not concept.expected_data_types:
        return 0.0

    try:
        expected_types = json.loads(
            concept.expected_data_types
        )
    except json.JSONDecodeError:
        return 0.0

    normalized_actual = normalize_data_type(
        data_type
    )

    normalized_expected = {
        normalize_data_type(value)
        for value in expected_types
    }

    return (
        1.0
        if normalized_actual in normalized_expected
        else 0.0
    )


def profile_compatibility(
    profile: dict | None,
    concept: SemanticConcept,
    column_name: str | None = None,
    column_description: str | None = None,

) -> float:
    if not profile or not concept.profile_expectations:
        return 0.0

    try:
        expectations = json.loads(
            concept.profile_expectations
        )
    except json.JSONDecodeError:
        return 0.0

    if not expectations:
        return 0.0

    matches = 0
    checks = 0

    data_type = str(
        profile.get("data_type", "")
    ).lower()

    null_percentage = profile.get(
        "null_percentage"
    )

    distinct_percentage = profile.get(
        "distinct_percentage"
    )

    identifier_name_signal = bool(
        profile.get(
            "identifier_name_signal",
            False,
        )
    )

    identifier_signal = bool(
        profile.get(
            "identifier_signal",
            False,
        )
    )

    text_profile = profile.get(
        "text",
        {},
    ) or {}

    patterns = text_profile.get(
        "patterns",
        {},
    ) or {}

    numeric_profile = profile.get(
        "numeric",
        {},
    ) or {}

    # ---------------------------------------------------------
    # Basic type detection
    # ---------------------------------------------------------

    is_numeric = (
        data_type in {
            "int",
            "int64",
            "int32",
            "float",
            "float64",
            "float32",
            "decimal",
            "numeric",
        }
        or bool(numeric_profile)
    )

    is_text = (
        data_type in {
            "str",
            "string",
            "object",
            "text",
        }
        or bool(text_profile)
    )

    is_temporal = (
        data_type in {
            "date",
            "datetime",
            "datetime64",
            "datetime64[ns]",
            "timestamp",
        }
    )

    # ---------------------------------------------------------
    # identifier_like
    # ---------------------------------------------------------

       # ---------------------------------------------------------
    # identifier_like
    #
    # Identifier evidence can come from:
    # 1. Stage 03 identifier signals, or
    # 2. Strong semantic name/description evidence combined
    #    with a complete and unique profile.
    #
    # The second path prevents naturally unique measures such
    # as revenue from automatically becoming identifiers.
    # ---------------------------------------------------------

    if "identifier_like" in expectations:
        checks += 1

        actual_identifier_like = (
            identifier_name_signal
            or identifier_signal
        )

        if not actual_identifier_like:
            unique_profile = (
                distinct_percentage is not None
                and float(distinct_percentage) >= 99.0
            )

            complete_profile = (
                null_percentage is not None
                and float(null_percentage) == 0.0
            )

            semantic_name_evidence = name_similarity(
                column_name,
                concept,
            )

            semantic_description_evidence = (
                description_similarity(
                    column_description,
                    concept,
                )
            )

            semantic_identifier_evidence = (
                max(
                    semantic_name_evidence,
                    semantic_description_evidence,
                )
                >= 0.50
            )

            actual_identifier_like = (
                unique_profile
                and complete_profile
                and semantic_identifier_evidence
            )

        if actual_identifier_like == bool(
            expectations["identifier_like"]
        ):
            matches += 1

    # ---------------------------------------------------------
    # unique
    # ---------------------------------------------------------

    if "unique" in expectations:
        checks += 1

        actual_unique = False

        if distinct_percentage is not None:
            actual_unique = (
                float(distinct_percentage) >= 99.0
            )

        if actual_unique == bool(
            expectations["unique"]
        ):
            matches += 1

    # ---------------------------------------------------------
    # non_null
    # ---------------------------------------------------------

    if "non_null" in expectations:
        checks += 1

        actual_non_null = False

        if null_percentage is not None:
            actual_non_null = (
                float(null_percentage) == 0.0
            )

        if actual_non_null == bool(
            expectations["non_null"]
        ):
            matches += 1

    # ---------------------------------------------------------
    # numeric
    # ---------------------------------------------------------

    if "numeric" in expectations:
        checks += 1

        if is_numeric == bool(
            expectations["numeric"]
        ):
            matches += 1

    # ---------------------------------------------------------
    # text
    # ---------------------------------------------------------

    if "text" in expectations:
        checks += 1

        if is_text == bool(
            expectations["text"]
        ):
            matches += 1

    # ---------------------------------------------------------
    # temporal
    # ---------------------------------------------------------

    if "temporal" in expectations:
        checks += 1

        if is_temporal == bool(
            expectations["temporal"]
        ):
            matches += 1

    # ---------------------------------------------------------
    # categorical
    #
    # Textual columns with relatively low cardinality
    # are treated as categorical.
    # ---------------------------------------------------------

    if "categorical" in expectations:
        checks += 1

        actual_categorical = False

        if (
            is_text
            and distinct_percentage is not None
        ):
            actual_categorical = (
                float(distinct_percentage) < 50.0
            )

        if actual_categorical == bool(
            expectations["categorical"]
        ):
            matches += 1

    # ---------------------------------------------------------
    # non_negative
    # ---------------------------------------------------------

    if "non_negative" in expectations:
        checks += 1

        actual_non_negative = False

        if numeric_profile:
            minimum = numeric_profile.get("min")

            if minimum is not None:
                actual_non_negative = (
                    float(minimum) >= 0
                )

        if actual_non_negative == bool(
            expectations["non_negative"]
        ):
            matches += 1

    # ---------------------------------------------------------
    # continuous
    #
    # A numeric column with many distinct values is treated
    # as continuous. This is evidence, not a hard type.
    # ---------------------------------------------------------

    if "continuous" in expectations:
        checks += 1

        actual_continuous = False

        if (
            is_numeric
            and distinct_percentage is not None
        ):
            actual_continuous = (
                float(distinct_percentage) >= 50.0
            )

        if actual_continuous == bool(
            expectations["continuous"]
        ):
            matches += 1

    # ---------------------------------------------------------
    # bounded
    #
    # Basic bounded-value evidence using min/max.
    # ---------------------------------------------------------

    if "bounded" in expectations:
        checks += 1

        actual_bounded = False

        if numeric_profile:
            minimum = numeric_profile.get("min")
            maximum = numeric_profile.get("max")

            if (
                minimum is not None
                and maximum is not None
            ):
                actual_bounded = (
                    float(minimum) >= 0
                    and float(maximum) <= 150
                )

        if actual_bounded == bool(
            expectations["bounded"]
        ):
            matches += 1

    # ---------------------------------------------------------
    # Pattern-based expectations
    # ---------------------------------------------------------

    pattern_expectations = {
        "email_pattern": "email_like",
        "phone_pattern": "phone_like",
        "postal_pattern": "postal_like",
        "currency_code": "currency_like",
    }

    for expectation_key, profile_key in (
        pattern_expectations.items()
    ):
        if expectation_key not in expectations:
            continue

        checks += 1

        pattern_count = patterns.get(
            profile_key,
            0,
        )

        actual_pattern = (
            pattern_count > 0
        )

        if actual_pattern == bool(
            expectations[expectation_key]
        ):
            matches += 1

    # ---------------------------------------------------------
    # year
    #
    # Year-established columns should contain integer-like
    # values in a plausible calendar-year range.
    # ---------------------------------------------------------

    if "year" in expectations:
        checks += 1

        actual_year = False

        if numeric_profile:
            minimum = numeric_profile.get("min")
            maximum = numeric_profile.get("max")

            if (
                minimum is not None
                and maximum is not None
            ):
                actual_year = (
                    1900 <= float(minimum) <= 2100
                    and
                    1900 <= float(maximum) <= 2100
                )

        if actual_year == bool(
            expectations["year"]
        ):
            matches += 1

    # ---------------------------------------------------------
    # long_text
    # ---------------------------------------------------------

    if "long_text" in expectations:
        checks += 1

        actual_long_text = False

        max_length = text_profile.get(
            "max_length"
        )

        mean_length = text_profile.get(
            "mean_length"
        )

        if max_length is not None:
            actual_long_text = (
                float(max_length) >= 100
            )

        elif mean_length is not None:
            actual_long_text = (
                float(mean_length) >= 50
            )

        if actual_long_text == bool(
            expectations["long_text"]
        ):
            matches += 1

    if checks == 0:
        return 0.0

    return round(
        matches / checks,
        4,
    )

def context_similarity(
    dataset_domain: str | None,
    table_context: str | None,
    concept: SemanticConcept,
) -> float:
    """
    Estimate contextual compatibility using the concept's
    category and description.
    """

    context_parts = [
        dataset_domain or "",
        table_context or "",
    ]

    context_text = " ".join(
        part for part in context_parts if part
    )

    if not context_text:
        return 0.0

    return text_similarity(
        context_text,
        concept.category,
    )


def generate_features(
    column_name: str,
    concept: SemanticConcept,
    embedding_similarity: float,
    column_description: str | None = None,
    data_type: str | None = None,
    profile: dict | None = None,
    dataset_domain: str | None = None,
    table_context: str | None = None,
) -> dict:
    """
    Generate the complete deterministic feature vector
    for one column/concept candidate.
    """

    features = {
        "embedding_similarity": float(
            embedding_similarity
        ),
        "name_similarity": float(
            name_similarity(
                column_name,
                concept,
            )
        ),
        "description_similarity": float(
            description_similarity(
                column_description,
                concept,
            )
        ),
        "datatype_compatibility": float(
            data_type_compatibility(
                data_type,
                concept,
            )
        ),
                "profile_compatibility": float(
            profile_compatibility(
                profile,
                concept,
                column_name=column_name,
                column_description=column_description,
            )
        ),
        "context_similarity": float(
            context_similarity(
                dataset_domain,
                table_context,
                concept,
            )
        ),
    }

    return features


def calculate_deterministic_score(
    features: dict,
) -> float:
    """
    Calculate the initial deterministic semantic score.

    This weighting will later be replaced or enhanced
    by the XGBoost reranker once labeled feedback exists.
    """

    score = (
        0.40 * features["embedding_similarity"]
        + 0.20 * features["name_similarity"]
        + 0.15 * features["description_similarity"]
        + 0.10 * features["datatype_compatibility"]
        + 0.10 * features["profile_compatibility"]
        + 0.05 * features["context_similarity"]
    )

    return round(
        max(0.0, min(1.0, score)),
        4,
    )
"""Structured column-name normalization for Stage 04.

`normalize_name` turns a raw column name into a JSON-serialisable
`NameParts` record that later evidence stages consume:

- ``tokens``: ordered lowercase tokens (duplicates KEPT, camelCase /
  PascalCase / acronym / letter-digit boundaries split, unicode preserved).
- ``expanded_tokens``: tokens after abbreviation expansion (order kept).
- ``alternatives``: uncertain expansions, kept OUT of the primary reading
  but exposed as evidence.
- ``entity_tokens``: non-role tokens describing WHAT the column is about
  (e.g. ``warehouse`` in ``warehouse_id``).
- ``head``: the role word (``identifier``, ``date``, ``amount`` ... ) or None.
- ``stripped_prefix``: table-name tokens removed from the front (recorded,
  never silently dropped).
- ``glued_split`` / ``opaque``: conservative structural flags.

The legacy public functions (``normalize_text``, ``tokenize``,
``expanded_text``, ``expand_tokens``, ``text_similarity``, ``alias_similarity``,
``name_similarity``) remain importable with their original signatures and are
re-implemented on top of ``normalize_name`` so every consumer shares one
normalization path (ordered, camelCase-aware, never sorted).
"""

import re
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache

from app.services.abbreviation_lexicon import resolve_token

# ---------------------------------------------------------------------------
# Role vocabulary
# ---------------------------------------------------------------------------

#: Tokens that describe the ROLE of a column rather than its meaning.
HEAD_TOKENS: frozenset[str] = frozenset(
    {
        "identifier",
        "id",
        "code",
        "number",
        "name",
        "date",
        "time",
        "timestamp",
        "amount",
        "price",
        "cost",
        "fee",
        "quantity",
        "count",
        "total",
        "rate",
        "ratio",
        "percent",
        "percentage",
        "score",
        "rating",
        "flag",
        "status",
        "type",
        "category",
        "class",
        "description",
        "comment",
        "text",
        "address",
        "email",
        "phone",
        "url",
        "year",
        "month",
        "day",
        "hour",
        "age",
        "weight",
        "length",
        "width",
        "height",
        "area",
        "size",
        "level",
    }
)

#: Head synonyms: any two tokens in one group are the same role word.
_HEAD_SYNONYM_GROUPS: tuple[tuple[str, ...], ...] = (
    ("id", "identifier"),
    ("no", "number"),
    ("cd", "code"),
    ("percentage", "percent"),
    ("time", "timestamp"),
    ("rating", "score"),
)

#: Prefixes that only mark boolean-ness ("is_fraud") and never carry meaning.
_BOOLEAN_PREFIX_TOKENS: frozenset[str] = frozenset({"is", "has", "was"})

#: Structural noise: tokens that name the THING (a column) rather than its
#: meaning. Removed before opacity is judged and excluded from entity tokens.
_NOISE_TOKENS: frozenset[str] = frozenset(
    {
        "col",
        "column",
        "field",
        "var",
        "variable",
        "attr",
        "attribute",
        "feature",
        "unnamed",
        "unknown",
    }
)

#: Generic type prefixes/suffixes stripped only when a real token remains.
_GENERIC_TYPE_PREFIXES: tuple[str, ...] = ("col", "fld", "tbl")
_GENERIC_TYPE_SUFFIXES: tuple[str, ...] = ("txt", "str", "num")

_OPAQUE_PATTERN = re.compile(r"^(x|y|z|c|v|f)?[\s_:.\-]*\d*$")

_KEY_RE = re.compile(r"[^a-z0-9]+")

# ---------------------------------------------------------------------------
# Result record
# ---------------------------------------------------------------------------


@dataclass
class NameParts:
    """Normalized structure of one column name (JSON-serialisable)."""

    raw: str
    table_name: str | None
    tokens: list[str]
    expanded_tokens: list[str]
    alternatives: list[str]
    entity_tokens: list[str]
    head: str | None
    opaque: bool
    stripped_prefix: str | None
    glued_split: bool

    @property
    def key(self) -> str:
        """Ordered expanded-token key used by the exact/alias stage."""
        return " ".join(self.expanded_tokens)

    def to_dict(self) -> dict:
        return {
            "raw": self.raw,
            "table_name": self.table_name,
            "tokens": list(self.tokens),
            "expanded_tokens": list(self.expanded_tokens),
            "alternatives": list(self.alternatives),
            "entity_tokens": list(self.entity_tokens),
            "head": self.head,
            "opaque": self.opaque,
            "stripped_prefix": self.stripped_prefix,
            "glued_split": self.glued_split,
        }


# ---------------------------------------------------------------------------
# Raw splitting (camelCase / PascalCase / acronyms / digits / separators)
# ---------------------------------------------------------------------------


def _split_raw(name: str) -> list[str]:
    """Split a raw identifier into ordered lowercase tokens.

    Boundaries: separators (``_ - . /`` and whitespace), lower->upper and
    acronym-end transitions (``IDNumber`` -> ``ID Number``), letter<->digit
    transitions (``addr1`` -> ``addr 1``). Already-lowercase words are never
    split. Non-ASCII letters are preserved.
    """
    text = unicodedata.normalize("NFC", str(name))
    tokens: list[str] = []
    current: list[str] = []

    def is_cased(char: str) -> bool:
        return char.isupper() or char.islower()

    def flush() -> None:
        if current:
            tokens.append("".join(current).lower())
            current.clear()

    previous: str | None = None
    for position, char in enumerate(text):
        if not char.isalnum() or char == "_":
            # Separators: whitespace, _ - . / and any other punctuation.
            flush()
            previous = None
            continue

        if previous is not None:
            boundary = False
            if is_cased(char) and is_cased(previous):
                if previous.islower() and char.isupper():
                    # lower->Upper: camelCase / PascalCase boundary.
                    boundary = True
                elif previous.isupper() and char.isupper():
                    # Acronym end: upper-run breaks only before its LAST
                    # element ("IDNumber" -> "ID Number", "ABCd" -> "ab cd").
                    lookahead = text[position + 1] if position + 1 < len(text) else ""
                    if lookahead and lookahead.islower():
                        boundary = True
            elif char.isdigit() != previous.isdigit():
                # letter<->digit boundary ("addr1" -> "addr 1").
                boundary = True
            elif is_cased(char) and not is_cased(previous):
                # Uncased letter (e.g. CJK) next to a cased letter.
                boundary = char.isupper()

            if boundary:
                flush()

        current.append(char)
        previous = char

    flush()
    return tokens


# ---------------------------------------------------------------------------
# Glued-token splitting (conservative)
# ---------------------------------------------------------------------------


# Generic business-entity nouns a data steward recognises. Used ONLY by the
# conservative glued-token splitter (Part A.4) so "warehouseid" can split
# into known words; they are not abbreviations and never replace tokens.
_GENERIC_ENTITY_WORDS: frozenset[str] = frozenset(
    {
        "customer",
        "client",
        "account",
        "product",
        "item",
        "order",
        "store",
        "warehouse",
        "supplier",
        "seller",
        "vendor",
        "employee",
        "staff",
        "user",
        "session",
        "page",
        "website",
        "payment",
        "transaction",
        "invoice",
        "shipment",
        "review",
        "rating",
        "brand",
        "category",
        "company",
        "organization",
        "flight",
        "passenger",
        "host",
        "listing",
        "property",
        "house",
        "ticket",
        "patient",
        "student",
        "merchant",
        "cardholder",
        "device",
        "batch",
        "operator",
        "loyalty",
    }
)


@lru_cache(maxsize=1)
def _known_vocabulary() -> frozenset[str]:
    """Known words for glue splitting: lexicon keys/values + role tokens +
    generic entity nouns.

    KB vocabulary is added at runtime via `register_known_words` (the KB
    module imports this one, not the other way around).
    """
    from app.services.abbreviation_lexicon import _load_lexicon

    high, uncertain = _load_lexicon()
    words: set[str] = set(HEAD_TOKENS) | set(_GENERIC_ENTITY_WORDS)
    words.update(high.keys())
    for expansion in high.values():
        words.update(expansion.split())
    for alternatives in uncertain.values():
        for expansion in alternatives:
            words.update(expansion.split())
    return frozenset(words)


_EXTRA_KNOWN_WORDS: set[str] = set()


def register_known_words(words) -> None:
    """Register extra known words (KB vocabulary) for glue splitting."""
    for word in words:
        lowered = str(word).strip().lower()
        if len(lowered) >= 2:
            _EXTRA_KNOWN_WORDS.add(lowered)


def _is_known_word(token: str) -> bool:
    return token in _known_vocabulary() or token in _EXTRA_KNOWN_WORDS


def _split_glued(token: str) -> list[str] | None:
    """Split a glued token into known words, from the right, uniquely.

    Accepted only when EVERY part is a known word of length >= 2 and the
    split is unique. Never splits real dictionary words.
    """
    if len(token) < 4 or " " in token:
        return None
    if _is_known_word(token):
        return None

    n = len(token)
    splits: list[list[str]] = []

    def search(start: int, parts: list[str]) -> None:
        if len(splits) > 1:
            return  # not unique - give up early
        if start == n:
            if all(len(part) >= 2 and _is_known_word(part) for part in parts):
                splits.append(list(parts))
            return
        for end in range(start + 2, n + 1):
            part = token[start:end]
            if _is_known_word(part):
                parts.append(part)
                search(end, parts)
                parts.pop()

    search(0, [])
    return splits[0] if len(splits) == 1 else None


# ---------------------------------------------------------------------------
# Core normalization
# ---------------------------------------------------------------------------


def _expand_one(token: str) -> tuple[list[str], list[str], bool]:
    """Expand one token.

    Returns (replacement_tokens, alternatives, glue_split) where glue_split
    is True only when the token itself was split into known words (Part A.4).
    """
    resolution = resolve_token(token)

    if resolution.confidence == "high" and resolution.expansion:
        return resolution.expansion.split(), [], False

    if resolution.confidence == "uncertain":
        alternatives: list[str] = []
        for alternative in resolution.alternatives:
            alternatives.extend(alternative.split())
        # The FIRST alternative is the primary reading (matches the legacy
        # lexicon contract); the rest stay as evidence only.
        primary = alternatives[0] if alternatives else token
        return [primary], alternatives[1:], False

    glued = _split_glued(token)
    if glued:
        expanded: list[str] = []
        alternatives = []
        for part in glued:
            part_expanded, part_alternatives, _part_glued = _expand_one(part)
            expanded.extend(part_expanded)
            alternatives.extend(part_alternatives)
        return expanded, alternatives, True

    return [token], [], False


def _strip_table_prefix(
    tokens: list[str], table_name: str | None
) -> tuple[list[str], str | None]:
    """Strip leading tokens equal to the table-name tokens (>= 1 remains)."""
    if not table_name:
        return tokens, None

    table_tokens = [
        token for token in _split_raw(table_name) if token not in _NOISE_TOKENS
    ]
    if not table_tokens or len(tokens) <= len(table_tokens):
        return tokens, None

    if tokens[: len(table_tokens)] == table_tokens:
        remaining = tokens[len(table_tokens):]
        return remaining, " ".join(table_tokens)

    return tokens, None


def _strip_generic_types(
    tokens: list[str],
) -> tuple[list[str], str | None]:
    """Strip generic type prefixes/suffixes only when a real token remains."""
    stripped: str | None = None

    while len(tokens) > 1 and tokens[0] in _GENERIC_TYPE_PREFIXES:
        stripped = tokens[0]
        tokens = tokens[1:]

    while len(tokens) > 1 and tokens[-1] in _GENERIC_TYPE_SUFFIXES:
        stripped = tokens[-1]
        tokens = tokens[:-1]

    return tokens, stripped


def _effective_tokens(tokens: list[str]) -> list[str]:
    """Tokens that can carry meaning: no noise, no digits, no single letters."""
    return [
        token
        for token in tokens
        if token not in _NOISE_TOKENS and not token.isdigit() and len(token) > 1
    ]


def _head_and_entities(expanded_tokens: list[str]) -> tuple[str | None, list[str]]:
    """Head = last effective token when it is a role word; entities = rest."""
    effective = _effective_tokens(expanded_tokens)

    head: str | None = None
    entity_tokens: list[str] = []
    for token in reversed(effective):
        if head is None and token in HEAD_TOKENS:
            head = token
            continue
        if token in _BOOLEAN_PREFIX_TOKENS:
            continue
        entity_tokens.append(token)

    entity_tokens.reverse()
    return head, entity_tokens


@lru_cache(maxsize=4096)
def _normalize_name_cached(raw: str, table_name: str | None) -> NameParts:
    tokens = _split_raw(raw)

    tokens, table_prefix = _strip_table_prefix(tokens, table_name)
    tokens, type_marker = _strip_generic_types(tokens)
    stripped_prefix = table_prefix if table_prefix else type_marker

    expanded_tokens: list[str] = []
    alternatives: list[str] = []
    glued_split = False
    for token in tokens:
        replacement, token_alternatives, token_glued = _expand_one(token)
        glued_split = glued_split or token_glued
        expanded_tokens.extend(replacement)
        alternatives.extend(token_alternatives)

    head, entity_tokens = _head_and_entities(expanded_tokens)

    effective = _effective_tokens(tokens)
    opaque = len(effective) == 0 or bool(_OPAQUE_PATTERN.fullmatch(raw.strip().lower()))

    return NameParts(
        raw=raw,
        table_name=table_name,
        tokens=tokens,
        expanded_tokens=expanded_tokens,
        alternatives=alternatives,
        entity_tokens=entity_tokens,
        head=head,
        opaque=opaque,
        stripped_prefix=stripped_prefix,
        glued_split=glued_split,
    )


def normalize_name(raw: str | None, table_name: str | None = None) -> NameParts:
    """Normalize one column name into structured, ordered parts."""
    if raw is None:
        raw = ""
    return _normalize_name_cached(str(raw), str(table_name) if table_name else None)


# ---------------------------------------------------------------------------
# Legacy API (same signatures as the pre-existing semantic_features helpers)
# ---------------------------------------------------------------------------


def normalize_text(value: str | None) -> str:
    """Normalized, lowercase, separator-collapsed text (legacy contract)."""
    if not value:
        return ""
    return " ".join(_split_raw(value))


def tokenize(value: str | None) -> set[str]:
    """Ordered tokens as a set (legacy signature)."""
    return set(_split_raw(value)) if value else set()


def expand_tokens(tokens: set[str]) -> set[str]:
    """Expand known abbreviations into their full forms (legacy signature)."""
    expanded: set[str] = set()
    for token in tokens:
        replacement, _alternatives, _glued = _expand_one(token)
        expanded.update(replacement)
    return expanded


def expanded_text(value: str | None) -> str:
    """Ordered, abbreviation-expanded natural text ("warehouse identifier")."""
    if not value:
        return ""
    return normalize_name(value).key


def _token_f1(first: set[str], second: set[str]) -> float:
    if not first or not second:
        return 0.0
    intersection = len(first & second)
    if not intersection:
        return 0.0
    precision = intersection / len(first)
    recall = intersection / len(second)
    return 2 * precision * recall / (precision + recall)


def text_similarity(first: str | None, second: str | None) -> float:
    """Combine abbreviation-expanded token F1 with sequence similarity."""
    first_normalized = normalize_text(first)
    second_normalized = normalize_text(second)

    if not first_normalized or not second_normalized:
        return 0.0

    first_tokens = expand_tokens(set(first_normalized.split()))
    second_tokens = expand_tokens(set(second_normalized.split()))

    token_similarity = _token_f1(first_tokens, second_tokens)

    from difflib import SequenceMatcher

    sequence_similarity = SequenceMatcher(
        None,
        first_normalized,
        second_normalized,
    ).ratio()

    return 0.6 * token_similarity + 0.4 * sequence_similarity


def alias_similarity(column_name: str, aliases: list[str]) -> float:
    """Strongest similarity between a column name and concept aliases."""
    if not aliases:
        return 0.0
    return max(text_similarity(column_name, alias) for alias in aliases)


def name_similarity(column_name: str, concept) -> float:
    """Legacy lexical name similarity against a concept name and aliases.

    Kept for backward compatibility; the evidence-grade name similarity
    (entity/head aware) lives in semantic_features.name_similarity.
    """
    concept_score = text_similarity(column_name, concept.concept_name)

    import json

    aliases: list[str] = []
    if concept.aliases:
        try:
            loaded = json.loads(concept.aliases)
            if isinstance(loaded, list):
                aliases = [str(alias) for alias in loaded]
        except json.JSONDecodeError:
            aliases = []

    column_tokens = expand_tokens(tokenize(normalize_text(column_name)))

    alias_scores = []
    for alias in aliases:
        score = text_similarity(column_name, alias)
        alias_tokens = expand_tokens(tokenize(normalize_text(alias)))
        if column_tokens and alias_tokens and column_tokens < alias_tokens:
            # Strict containment: the alias carries an extra qualifier.
            score *= 0.5
        alias_scores.append(score)

    alias_score = max(alias_scores) if alias_scores else 0.0
    score = max(concept_score, alias_score)

    normalized_aliases = {normalize_text(alias) for alias in aliases}
    normalized_name = normalize_text(column_name)
    if normalized_name and (
        normalized_name == normalize_text(concept.concept_name)
        or normalized_name in normalized_aliases
    ):
        score = min(1.0, score + 0.1)

    return score

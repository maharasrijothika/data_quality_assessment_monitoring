"""Abbreviation lexicon for Stage 04 semantic normalization.

Loads backend/app/data/abbreviation_dictionary.json and classifies each
expansion as HIGH-CONFIDENCE, UNCERTAIN, or NO-EXPANSION:

- High-confidence expansions replace the token in the expanded
  representation.
- Uncertain expansions keep the original token AND add all alternatives as
  evidence; nothing is forced.
- Unknown tokens are carried through unchanged.

The lexicon never modifies source data and never mutates itself from user
decisions: feedback may propose candidate mappings, but admission into the
dictionary is a separate controlled process (see _meta.admission_policy).
"""

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

_DICTIONARY_PATH = Path(__file__).resolve().parent.parent / "data" / "abbreviation_dictionary.json"


@dataclass
class TokenResolution:
    """Normalization evidence for one token of a column name."""

    original: str
    expansion: str | None = None
    alternatives: list[str] = field(default_factory=list)
    confidence: str = "none"  # high | uncertain | none

    @property
    def expanded(self) -> bool:
        return self.expansion is not None


@dataclass
class NameResolution:
    """Full normalization evidence for one column name."""

    original_name: str
    normalized_name: str
    expanded_name: str
    tokens: list[TokenResolution] = field(default_factory=list)

    @property
    def abbreviation_evidence(self) -> list[dict]:
        """JSON-safe evidence records for persisted analysis payloads."""
        return [
            {
                "token": token.original,
                "expansion": token.expansion,
                "alternatives": list(token.alternatives),
                "confidence": token.confidence,
            }
            for token in self.tokens
            if token.expanded or token.alternatives
        ]


@lru_cache(maxsize=1)
def _load_lexicon() -> tuple[dict[str, str], dict[str, list[str]]]:
    if not _DICTIONARY_PATH.exists():
        return {}, {}

    payload = json.loads(_DICTIONARY_PATH.read_text(encoding="utf-8"))

    high_confidence = {
        str(token).lower(): str(expansion).lower()
        for token, expansion in (payload.get("high_confidence") or {}).items()
    }
    uncertain = {
        str(token).lower(): [str(alt).lower() for alt in alternatives]
        for token, alternatives in (payload.get("uncertain") or {}).items()
    }

    return high_confidence, uncertain


def clear_lexicon_cache() -> None:
    """Reset the cached lexicon (used after controlled dictionary edits)."""
    _load_lexicon.cache_clear()


def resolve_token(token: str) -> TokenResolution:
    """Classify one normalized token: high-confidence / uncertain / none."""
    lowered = token.lower()

    high_confidence, uncertain = _load_lexicon()

    if lowered in high_confidence:
        return TokenResolution(
            original=token,
            expansion=high_confidence[lowered],
            confidence="high",
        )

    if lowered in uncertain:
        alternatives = uncertain[lowered]
        return TokenResolution(
            original=token,
            expansion=alternatives[0] if alternatives else None,
            alternatives=alternatives,
            confidence="uncertain",
        )

    return TokenResolution(original=token, confidence="none")


def resolve_name(normalized_name: str) -> NameResolution:
    """Resolve every token of an already-normalized column name.

    Accepts raw snake_case / kebab-case / camelCase names too: tokens are
    re-split on separators and camelCase transitions so callers do not have
    to pre-normalize.
    """
    import re

    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", str(normalized_name))
    tokens = [token for token in re.split(r"[^A-Za-z0-9]+", text) if token]
    resolutions = [resolve_token(token) for token in tokens]

    expanded_tokens: list[str] = []
    for resolution in resolutions:
        if resolution.confidence == "high" and resolution.expansion:
            expanded_tokens.extend(resolution.expansion.split())
        else:
            # Uncertain and unknown tokens are preserved verbatim; uncertain
            # alternatives are kept as evidence, never forced into the name.
            expanded_tokens.append(resolution.original)

    return NameResolution(
        original_name=normalized_name,
        normalized_name=normalized_name,
        expanded_name=" ".join(expanded_tokens),
        tokens=resolutions,
    )

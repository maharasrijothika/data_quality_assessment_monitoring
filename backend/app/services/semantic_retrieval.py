import json

import numpy as np
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.config import (
    SEMANTIC_EMBEDDING_REPRESENTATION_VERSION,
    SEMANTIC_MODEL_NAME,
    SEMANTIC_TOP_K,
)
from app.models import SemanticConcept, SemanticConceptEmbedding
from app.services.semantic_features import concept_info, expanded_text


MODEL_NAME = SEMANTIC_MODEL_NAME

# KB embeddings are persisted as "<model>/<representation-version>" so the
# sentence encoder AND the concept text template are both versioned. Changing
# the representation triggers a one-time backfill of stale embeddings.
EMBEDDING_KEY = f"{SEMANTIC_MODEL_NAME}/{SEMANTIC_EMBEDDING_REPRESENTATION_VERSION}"

# Module-level model cache. The sentence transformer is loaded ONCE per
# process and reused for every request; KB-side embeddings are persisted in
# the database and are never regenerated per request.
_MODEL_CACHE: dict[str, object] = {}


def get_embedding_model():
    """Return the (cached) sentence transformer model."""
    if MODEL_NAME not in _MODEL_CACHE:
        from sentence_transformers import SentenceTransformer

        _MODEL_CACHE[MODEL_NAME] = SentenceTransformer(MODEL_NAME)

    return _MODEL_CACHE[MODEL_NAME]


def ensure_kb_embeddings(db: Session) -> bool:
    """Ensure every KB concept has a stored embedding for the current model
    and representation version.

    Embeddings are generated once and persisted (versioned by
    "<model>/<repr-version>"); they are never regenerated during normal
    retrieval. This backfill only runs for concepts missing a current-
    version embedding (e.g. freshly seeded KB or bumped representation).

    Returns True when at least one embedding was (re)generated - callers use
    this to invalidate the in-memory concept matrix.
    """
    concepts_without_embedding = (
        db.query(SemanticConcept)
        .outerjoin(
            SemanticConceptEmbedding,
            (
                SemanticConceptEmbedding.concept_id
                == SemanticConcept.concept_id
            )
            & (SemanticConceptEmbedding.model_name == EMBEDDING_KEY),
        )
        .filter(SemanticConceptEmbedding.embedding_id.is_(None))
        .all()
    )

    if not concepts_without_embedding:
        return False

    # Imported here to avoid a module-level circular import.
    from app.services.semantic_embedding import SemanticEmbeddingService

    embedding_service = SemanticEmbeddingService()

    for concept in concepts_without_embedding:
        embedding_service.create_or_update_embedding(db, concept)

    db.flush()
    invalidate_matrix_cache()
    return True


# ---------------------------------------------------------------------------
# In-memory concept matrix cache (Part C.1)
# ---------------------------------------------------------------------------

# {(db_key, EMBEDDING_KEY, kb_fingerprint): cache-entry}
# Entry: {"ids": [...], "matrix": np.float32 (n, dim), "info": {id: info}}
# The fingerprint changes when any concept or embedding row changes (row
# counts + max ids + max created_at), so a stale cache can never serve
# outdated vectors: a different fingerprint is a different cache key.
_MATRIX_CACHE: dict[tuple, dict] = {}


def invalidate_matrix_cache(db: Session | None = None) -> None:
    """Drop cached concept matrices.

    Called after seeding, backfills and any concept/embedding mutation.
    With ``db`` only the entries for that database are dropped; without it
    the whole cache is cleared.
    """
    if db is None:
        _MATRIX_CACHE.clear()
        return

    try:
        db_key = id(db.get_bind())
    except Exception:  # pragma: no cover - defensive
        _MATRIX_CACHE.clear()
        return

    for key in [key for key in _MATRIX_CACHE if key[0] == db_key]:
        _MATRIX_CACHE.pop(key, None)


def _kb_fingerprint(db: Session) -> str:
    """Cheap fingerprint of concept + embedding state (changes on any write)."""
    concept_stats = db.query(
        func.count(SemanticConcept.concept_id),
        func.max(SemanticConcept.concept_id),
    ).one()
    embedding_stats = db.query(
        func.count(SemanticConceptEmbedding.embedding_id),
        func.max(SemanticConceptEmbedding.embedding_id),
        func.max(SemanticConceptEmbedding.created_at),
    ).one()

    raw = "|".join(
        str(value)
        for value in (
            concept_stats[0],
            concept_stats[1],
            embedding_stats[0],
            embedding_stats[1],
            embedding_stats[2],
        )
    )
    return str(hash(raw))


def kb_fingerprint(db: Session) -> str:
    """Public fingerprint accessor (persisted with Stage 04 predictions so
    an analysis run can be traced to the exact KB state it used)."""
    return _kb_fingerprint(db)


def load_concept_matrix(db: Session) -> dict:
    """Load (or fetch from cache) the concept embedding matrix.

    One DB read per fingerprint per process; no per-column json.loads. The
    cache key includes the KB fingerprint so any concept or embedding change
    invalidates the entry implicitly.
    """
    db_key = id(db.get_bind())
    fingerprint = _kb_fingerprint(db)
    cache_key = (db_key, EMBEDDING_KEY, fingerprint)

    cached = _MATRIX_CACHE.get(cache_key)
    if cached is not None:
        return cached

    records = (
        db.query(
            SemanticConcept,
            SemanticConceptEmbedding,
        )
        .join(
            SemanticConceptEmbedding,
            SemanticConcept.concept_id
            == SemanticConceptEmbedding.concept_id,
        )
        .filter(
            SemanticConceptEmbedding.model_name == EMBEDDING_KEY
        )
        .all()
    )

    ids: list[int] = []
    vectors: list[list[float]] = []
    infos: dict[int, dict] = {}

    for concept, embedding_record in records:
        try:
            stored_vector = json.loads(embedding_record.embedding)
        except (json.JSONDecodeError, TypeError):
            continue
        ids.append(concept.concept_id)
        vectors.append(stored_vector)
        infos[concept.concept_id] = concept_info(concept)

    matrix = (
        np.asarray(vectors, dtype=np.float32)
        if vectors
        else np.zeros((0, 1), dtype=np.float32)
    )

    entry = {"ids": ids, "matrix": matrix, "info": infos}
    _MATRIX_CACHE[cache_key] = entry

    # Keep the cache small: drop other fingerprints of the same database.
    for key in [
        key
        for key in _MATRIX_CACHE
        if key[0] == db_key and key != cache_key
    ]:
        _MATRIX_CACHE.pop(key, None)

    return entry


def encode_queries(model, texts: list[str]) -> np.ndarray:
    """Encode a batch of query texts (one model call, normalized vectors)."""
    if not texts:
        return np.zeros((0, 1), dtype=np.float32)

    embeddings = model.encode(
        texts,
        batch_size=64,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.asarray(embeddings, dtype=np.float32)


class SemanticRetrievalService:
    """
    Retrieves the most semantically similar concepts from
    the Semantic Knowledge Base.

    v2: embeddings come from a cached in-memory matrix (one DB read per
    fingerprint), queries are encoded in batches by the dataset-level
    service, and the candidate set is a hybrid union of embedding top-K and
    lexical top-K so a lexically exact concept can never be missed by the
    encoder.
    """

    def __init__(self):
        self.model = get_embedding_model()

    @staticmethod
    def build_column_text(
        column_name: str,
        column_description: str | None = None,
        dataset_domain: str | None = None,
        table_context: str | None = None,
        sibling_names: list[str] | None = None,
    ) -> str:
        """
        Build the semantic representation of an incoming column.

        v2 template: the abbreviation-expanded ORDERED name is ALWAYS the
        core (fixes embedding the raw name when a description exists);
        description and table context (table name + up to 12 sibling column
        names, expanded) are appended when present. The dataset DOMAIN is
        deliberately NOT embedded: it pulls every column of a dataset
        toward the same concepts and is used as context evidence instead.
        """
        expanded_name = expanded_text(column_name) or str(column_name)

        parts = [expanded_name]

        if column_description:
            parts.append(str(column_description))

        context_bits: list[str] = []
        if table_context:
            context_bits.append(f"table: {table_context}")
        if sibling_names:
            siblings = [
                expanded_text(str(name))
                for name in sibling_names[:12]
                if str(name) != str(column_name)
            ]
            siblings = [sibling for sibling in siblings if sibling]
            if siblings:
                context_bits.append("columns: " + ", ".join(siblings))

        if context_bits:
            parts.append("; ".join(context_bits))

        return ". ".join(
            part.strip(". ") for part in parts if part.strip(". ")
        )

    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate a normalized embedding for an incoming column.
        """
        embedding = self.model.encode(
            text,
            normalize_embeddings=True,
        )

        return np.asarray(embedding, dtype=np.float32)

    @staticmethod
    def cosine_similarity(
        query_embedding: np.ndarray,
        stored_embedding: list[float],
    ) -> float:
        """
        Calculate cosine similarity between two embeddings.

        Because both embeddings are normalized, this is equivalent
        to their dot product.
        """
        candidate = np.asarray(
            stored_embedding,
            dtype=np.float32,
        )

        return float(
            np.dot(
                query_embedding,
                candidate,
            )
        )

    @staticmethod
    def lexical_scores(
        column_name: str,
        entries: dict,
    ) -> dict[int, float]:
        """Lexical candidate scores: token overlap with concept names/aliases.

        Used to build the hybrid candidate union: a concept that shares any
        expanded token with the column name (or matches the concept exactly)
        is a lexical candidate even when the encoder ranks it low.
        """
        from app.services.semantic_name import normalize_name

        parts = normalize_name(column_name)
        column_tokens = set(parts.expanded_tokens)
        column_key = parts.key

        scores: dict[int, float] = {}
        for concept_id, info in entries.items():
            score = 0.0
            if column_key and column_key in info["keys"]:
                score = 1.0
            elif column_tokens:
                overlap = len(column_tokens & info["token_pool"])
                if overlap:
                    score = overlap / len(column_tokens)
            if score > 0:
                scores[concept_id] = score

        return scores

    def retrieve_candidates(
        self,
        db: Session,
        column_name: str,
        column_description: str | None = None,
        dataset_domain: str | None = None,
        table_context: str | None = None,
        top_k: int | None = None,
        sibling_names: list[str] | None = None,
        query_embedding: np.ndarray | None = None,
    ) -> list[dict]:
        """
        Retrieve the top-K semantic concepts for an incoming column.

        v2: candidate ranking uses the cached matrix (no per-column DB
        reads or json.loads). ``query_embedding`` lets the dataset-level
        service pass a pre-computed batched embedding; when omitted the
        text is encoded on the fly (single-column path). Embedding and
        lexical rankings are unioned: a concept appears once with its
        embedding similarity plus a ``lexical`` score.
        """
        top_k = top_k or SEMANTIC_TOP_K

        # Guarantee the shipped KB baseline exists for THIS database before
        # retrieval. Seeding is idempotent; the matrix cache is fingerprint-
        # keyed so any KB change invalidates itself.
        from app.services.semantic_kb import seed_semantic_concepts

        if db.query(SemanticConcept).count() == 0:
            seed_semantic_concepts(db)

        ensure_kb_embeddings(db)

        entry = load_concept_matrix(db)
        ids: list[int] = entry["ids"]
        matrix: np.ndarray = entry["matrix"]

        if not ids:
            return []

        if query_embedding is None:
            text = self.build_column_text(
                column_name=column_name,
                column_description=column_description,
                dataset_domain=dataset_domain,
                table_context=table_context,
                sibling_names=sibling_names,
            )
            query_embedding = self.generate_embedding(text)

        similarities = matrix @ query_embedding

        order = np.argsort(-similarities)[:top_k]
        embedding_scores = {
            ids[index]: float(similarities[index]) for index in order
        }

        # Hybrid: lexical candidates join even when the encoder misses them.
        lexical_scores = self.lexical_scores(column_name, entry["info"])
        for concept_id in lexical_scores:
            if concept_id not in embedding_scores:
                embedding_scores[concept_id] = float(
                    similarities[ids.index(concept_id)]
                )

        candidates = []
        for concept_id, similarity in embedding_scores.items():
            info = entry["info"].get(concept_id)
            candidates.append(
                {
                    "concept_id": concept_id,
                    "concept": info["concept_name"] if info else str(concept_id),
                    "category": None,
                    "similarity": similarity,
                    "lexical": lexical_scores.get(concept_id, 0.0),
                }
            )

        candidates.sort(
            key=lambda item: (item["similarity"], item["lexical"]),
            reverse=True,
        )

        return candidates[:top_k]

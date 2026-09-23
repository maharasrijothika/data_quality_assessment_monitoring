import json

import numpy as np
from sentence_transformers import SentenceTransformer
from sqlalchemy.orm import Session

from app.models import SemanticConcept, SemanticConceptEmbedding


MODEL_NAME = "all-MiniLM-L6-v2"


class SemanticRetrievalService:
    """
    Retrieves the most semantically similar concepts from
    the Semantic Knowledge Base.
    """

    def __init__(self):
        self.model = SentenceTransformer(MODEL_NAME)

    @staticmethod
    def build_column_text(
        column_name: str,
        column_description: str | None = None,
        dataset_domain: str | None = None,
        table_context: str | None = None,
    ) -> str:
        """
        Build the semantic representation of an incoming column.
        """

        parts = [
            f"Column: {column_name}",
        ]

        if column_description:
            parts.append(
                f"Description: {column_description}"
            )

        if dataset_domain:
            parts.append(
                f"Domain: {dataset_domain}"
            )

        if table_context:
            parts.append(
                f"Table context: {table_context}"
            )

        return "\n".join(parts)

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

    def retrieve_candidates(
        self,
        db: Session,
        column_name: str,
        column_description: str | None = None,
        dataset_domain: str | None = None,
        table_context: str | None = None,
        top_k: int = 5,
    ) -> list[dict]:
        """
        Retrieve the top-K semantic concepts for an incoming column.
        """

        text = self.build_column_text(
            column_name=column_name,
            column_description=column_description,
            dataset_domain=dataset_domain,
            table_context=table_context,
        )

        query_embedding = self.generate_embedding(text)

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
                SemanticConceptEmbedding.model_name
                == MODEL_NAME
            )
            .all()
        )

        candidates = []

        for concept, embedding_record in records:
            stored_vector = json.loads(
                embedding_record.embedding
            )

            similarity = self.cosine_similarity(
                query_embedding,
                stored_vector,
            )

            candidates.append(
                {
                    "concept_id": concept.concept_id,
                    "concept": concept.concept_name,
                    "category": concept.category,
                    "similarity": similarity,
                }
            )

        candidates.sort(
            key=lambda item: item["similarity"],
            reverse=True,
        )

        return candidates[:top_k]
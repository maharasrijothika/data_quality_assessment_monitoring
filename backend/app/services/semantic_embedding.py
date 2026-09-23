import json

from sentence_transformers import SentenceTransformer
from sqlalchemy.orm import Session

from app.models import SemanticConcept, SemanticConceptEmbedding


MODEL_NAME = "all-MiniLM-L6-v2"


class SemanticEmbeddingService:
    """
    Generates and stores embeddings for Semantic Knowledge Base concepts.
    """

    def __init__(self):
        self.model = SentenceTransformer(MODEL_NAME)

    @staticmethod
    def build_concept_text(concept: SemanticConcept) -> str:
        """
        Build a semantic representation of a concept using
        its name, description, aliases, data types, and
        profile expectations.
        """

        aliases = ""
        expected_data_types = ""
        profile_expectations = ""

        if concept.aliases:
            aliases = f"Aliases: {concept.aliases}"

        if concept.expected_data_types:
            expected_data_types = (
                f"Expected data types: "
                f"{concept.expected_data_types}"
            )

        if concept.profile_expectations:
            profile_expectations = (
                f"Profile expectations: "
                f"{concept.profile_expectations}"
            )

        return "\n".join(
            part
            for part in [
                f"Concept: {concept.concept_name}",
                f"Category: {concept.category}",
                f"Description: {concept.description}",
                aliases,
                expected_data_types,
                profile_expectations,
            ]
            if part
        )

    def generate_embedding(self, text: str) -> list[float]:
        """
        Generate a normalized sentence embedding.
        """

        embedding = self.model.encode(
            text,
            normalize_embeddings=True,
        )

        return embedding.tolist()

    def create_or_update_embedding(
        self,
        db: Session,
        concept: SemanticConcept,
    ) -> SemanticConceptEmbedding:
        """
        Generate an embedding for a concept and create or
        update its stored embedding.
        """

        text = self.build_concept_text(concept)
        vector = self.generate_embedding(text)

        existing = (
            db.query(SemanticConceptEmbedding)
            .filter(
                SemanticConceptEmbedding.concept_id
                == concept.concept_id
            )
            .first()
        )

        if existing:
            existing.model_name = MODEL_NAME
            existing.embedding = json.dumps(vector)
            embedding_record = existing
        else:
            embedding_record = SemanticConceptEmbedding(
                concept_id=concept.concept_id,
                model_name=MODEL_NAME,
                embedding=json.dumps(vector),
            )

            db.add(embedding_record)

        db.commit()
        db.refresh(embedding_record)

        return embedding_record
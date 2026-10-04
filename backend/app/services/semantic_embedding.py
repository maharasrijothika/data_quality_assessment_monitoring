import json

from sqlalchemy.orm import Session

from app.config import SEMANTIC_MODEL_NAME
from app.models import SemanticConcept, SemanticConceptEmbedding
from app.services.semantic_retrieval import EMBEDDING_KEY, get_embedding_model


MODEL_NAME = EMBEDDING_KEY


class SemanticEmbeddingService:
    """
    Generates and stores embeddings for Semantic Knowledge Base concepts.
    """

    def __init__(self):
        self.model = get_embedding_model()

    @staticmethod
    def build_concept_text(concept: SemanticConcept) -> str:
        """
        Build the semantic representation of a concept.

        Only natural-language fields (name, category, description, aliases)
        are embedded. Raw JSON config (expected data types, profile
        expectations) is machine-readable evidence used elsewhere in scoring;
        embedding its serialized text pollutes the sentence-embedding space
        and depresses similarities for every concept.
        """

        aliases = ""

        if concept.aliases:
            try:
                alias_values = json.loads(concept.aliases)
                if isinstance(alias_values, list):
                    aliases = ", ".join(str(value) for value in alias_values)
            except json.JSONDecodeError:
                aliases = concept.aliases

        parts = [
            f"Concept: {concept.concept_name}",
            f"Category: {concept.category}",
            f"Description: {concept.description}",
        ]

        if aliases:
            parts.append(f"Also known as: {aliases}")

        return "\n".join(parts)

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
            if existing.model_name == MODEL_NAME and existing.embedding == json.dumps(vector):
                return existing

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
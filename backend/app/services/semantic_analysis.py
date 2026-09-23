from sqlalchemy.orm import Session

from app.models import SemanticConcept
from app.services.semantic_features import (
    calculate_deterministic_score,
    generate_features,
)
from app.services.semantic_retrieval import (
    SemanticRetrievalService,
)


class SemanticAnalysisService:
    """
    Performs Stage 04 semantic understanding by combining
    embedding retrieval with deterministic evidence scoring.
    """

    def __init__(self):
        self.retrieval_service = SemanticRetrievalService()

    @staticmethod
    def confidence_level(score: float) -> str:
        """
        Convert semantic score into an interpretable
        confidence category.
        """

        if score > 0.85:
            return "Strong"

        if score >= 0.60:
            return "Probable"

        if score >= 0.40:
            return "Ambiguous"

        return "Unknown"

    def analyze_column(
        self,
        db: Session,
        column_name: str,
        column_description: str | None = None,
        data_type: str | None = None,
        profile: dict | None = None,
        dataset_domain: str | None = None,
        table_context: str | None = None,
        top_k: int = 5,
    ) -> dict:
        """
        Analyze the semantic meaning of one incoming column.
        """

        candidates = (
            self.retrieval_service.retrieve_candidates(
                db=db,
                column_name=column_name,
                column_description=column_description,
                dataset_domain=dataset_domain,
                table_context=table_context,
                top_k=top_k,
            )
        )

        ranked_candidates = []

        for candidate in candidates:
            concept = (
                db.query(SemanticConcept)
                .filter(
                    SemanticConcept.concept_id
                    == candidate["concept_id"]
                )
                .first()
            )

            if concept is None:
                continue

            features = generate_features(
                column_name=column_name,
                concept=concept,
                embedding_similarity=candidate[
                    "similarity"
                ],
                column_description=column_description,
                data_type=data_type,
                profile=profile,
                dataset_domain=dataset_domain,
                table_context=table_context,
            )

            score = calculate_deterministic_score(
                features
            )

            ranked_candidates.append(
                {
                    "concept_id": concept.concept_id,
                    "concept": concept.concept_name,
                    "category": concept.category,
                    "confidence_score": score,
                    "confidence_level": (
                        self.confidence_level(score)
                    ),
                    "evidence": features,
                }
            )

        ranked_candidates.sort(
            key=lambda item: item[
                "confidence_score"
            ],
            reverse=True,
        )

        if not ranked_candidates:
            return {
                "column": column_name,
                "semantic_concept": None,
                "concept_id": None,
                "confidence_score": 0.0,
                "confidence_level": "Unknown",
                "alternatives": [],
                "evidence": {},
            }

        best = ranked_candidates[0]

        alternatives = [
            {
                "concept": item["concept"],
                "score": item["confidence_score"],
            }
            for item in ranked_candidates[1:]
        ]

        return {
            "column": column_name,
            "semantic_concept": best["concept"],
            "concept_id": best["concept_id"],
            "confidence_score": best[
                "confidence_score"
            ],
            "confidence_level": best[
                "confidence_level"
            ],
            "alternatives": alternatives,
            "evidence": best["evidence"],
        }
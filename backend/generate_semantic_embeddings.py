from app.database import SessionLocal
from app.models import SemanticConcept
from app.services.semantic_embedding import SemanticEmbeddingService


def main():
    db = SessionLocal()

    try:
        concepts = (
            db.query(SemanticConcept)
            .order_by(SemanticConcept.concept_id)
            .all()
        )

        print(f"Found {len(concepts)} semantic concepts.")

        if not concepts:
            print("No semantic concepts found.")
            return

        service = SemanticEmbeddingService()

        generated = 0

        for concept in concepts:
            print(
                f"Generating embedding: "
                f"{concept.concept_id} - "
                f"{concept.concept_name}"
            )

            service.create_or_update_embedding(
                db,
                concept,
            )

            generated += 1

        print(
            f"\nSuccessfully generated embeddings: "
            f"{generated}"
        )

    finally:
        db.close()


if __name__ == "__main__":
    main()
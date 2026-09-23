from app.database import SessionLocal
from app.models import SemanticConcept

db = SessionLocal()

try:
    concept = (
        db.query(SemanticConcept)
        .filter(
            SemanticConcept.concept_name == "Parent Company"
        )
        .first()
    )

    if concept is None:
        raise ValueError("Parent Company concept not found.")

    concept.aliases = (
        '["parent company", "parent organization", '
        '"holding company", "subsidiary parent", "owned by"]'
    )

    db.commit()

    print("Updated:", concept.concept_name)
    print("Aliases:", concept.aliases)

finally:
    db.close()

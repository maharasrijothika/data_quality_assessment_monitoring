import sys
from app.database import SessionLocal
from app.models import Dataset
from app.routers.profiling import get_dataset_profiling
from app.services.semantic_dataset_analysis import (
    SemanticDatasetAnalysisService,
)


def main():
    if len(sys.argv) != 2:
        raise ValueError(
            "Usage: python test_semantic_dataset_analysis.py <dataset_id>"
        )

    dataset_id = int(sys.argv[1])

    db = SessionLocal()

    try:
        dataset = (
            db.query(Dataset)
            .filter(Dataset.dataset_id == dataset_id)
            .first()
        )

        if dataset is None:
            raise ValueError(
                f"Dataset {dataset_id} was not found."
            )

        # Get the existing Stage 03 profiling result.
        profiling_result = get_dataset_profiling(
            dataset_id=dataset_id,
            db=db,
        )

        service = SemanticDatasetAnalysisService()

        result = service.analyze_dataset(
            db=db,
            dataset=dataset,
            profiling_result=profiling_result,
        )

        print("\nSTAGE 04 DATASET SEMANTIC ANALYSIS")
        print("=" * 60)

        print("Dataset:", result["dataset_name"])
        print("Version:", result["version_number"])
        print("Domain:", result["domain"])
        print("Source:", result["source_system"])
        print("Columns analyzed:", result["column_count"])

        print("\nRESULTS")
        print("-" * 60)

        for column in result["columns"]:
            analysis = column["semantic_analysis"]

            print(
                f"\nColumn: {column['column_name']}"
            )

            print(
                f"Description: "
                f"{column['description']}"
            )

            print(
                f"Data type: "
                f"{column['data_type']}"
            )

            print(
                f"Concept: "
                f"{analysis['semantic_concept']}"
            )

            print(
                f"Confidence: "
                f"{analysis['confidence_score']:.4f}"
            )

            print(
                f"Level: "
                f"{analysis['confidence_level']}"
            )

            print("Alternatives:")

            for alternative in analysis[
                "alternatives"
            ]:
                print(
                    f"  - {alternative['concept']}: "
                    f"{alternative['score']:.4f}"
                )

    finally:
        db.close()


if __name__ == "__main__":
    main()
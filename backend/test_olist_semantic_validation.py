import sys
from pathlib import Path

import pandas as pd

from app.database import SessionLocal
from app.models import (
    ColumnMetadata,
    Dataset,
    DatasetVersion,
    TableMetadata,
)
from app.services.profiling import profile_dataframe
from app.services.semantic_dataset_analysis import (
    SemanticDatasetAnalysisService,
)


BASE_DIR = Path(__file__).resolve().parents[1]

FILES = [
    BASE_DIR
    / "data"
    / "raw"
    / "datasets"
    / "7"
    / "v1"
    / "olist_customers_dataset.csv",

    BASE_DIR
    / "data"
    / "raw"
    / "datasets"
    / "8"
    / "v1"
    / "olist_orders_dataset.csv",

    BASE_DIR
    / "data"
    / "raw"
    / "datasets"
    / "15"
    / "v1"
    / "olist_order_items_dataset.csv",

    BASE_DIR
    / "data"
    / "raw"
    / "datasets"
    / "6"
    / "v1"
    / "olist_products_dataset.csv",
]


def main():
    db = SessionLocal()

    try:
        print("=" * 60)
        print("STAGE 04 UNSEEN DATASET VALIDATION")
        print("=" * 60)

        print("\nFiles:")
        for path in FILES:
            print(f"  - {path.name}")
            if not path.exists():
                raise FileNotFoundError(
                    f"File not found: {path}"
                )

        # ---------------------------------------------------------
        # 1. Create temporary dataset metadata
        # ---------------------------------------------------------

        dataset = Dataset(
            dataset_name="Olist Validation Temporary",
            source_system="Public Dataset",
            domain="E-commerce",
            description=(
                "Related customer, order, order-item, and "
                "product data used for unseen semantic validation."
            ),
            update_cadence="Static",
        )

        db.add(dataset)
        db.flush()

        version = DatasetVersion(
            dataset_id=dataset.dataset_id,
            parent_version_id=None,
            version_number=1,
            schema_fingerprint="temporary-validation",
            content_fingerprint="temporary-validation",
        )

        db.add(version)
        db.flush()

        # ---------------------------------------------------------
        # 2. Profile each file using existing Stage 03 profiler
        # ---------------------------------------------------------

        profiling_tables = []

        for file_path in FILES:
            print(f"\nProfiling: {file_path.name}")

            dataframe = pd.read_csv(file_path)

            table_name = file_path.stem

            profile = profile_dataframe(
                dataframe,
                table_name=table_name,
            )

            table = TableMetadata(
                version_id=version.version_id,
                table_name=table_name,
                source_file=file_path.name,
                sheet_name=None,
                row_count=len(dataframe),
                description=None,
            )

            db.add(table)
            db.flush()

            # -----------------------------------------------------
            # 3. Create temporary column metadata
            # -----------------------------------------------------

            for column_name in dataframe.columns:
                column_profile = next(
                    (
                        item
                        for item in profile["columns"]
                        if item["column_name"] == column_name
                    ),
                    None,
                )

                data_type = (
                    column_profile.get("data_type", "unknown")
                    if column_profile
                    else str(dataframe[column_name].dtype)
                )

                column_metadata = ColumnMetadata(
                    table_id=table.table_id,
                    column_name=column_name,
                    data_type=data_type,
                    description=None,
                )

                db.add(column_metadata)

            profiling_tables.append(profile)

            print(
                f"  Rows: {len(dataframe):,} | "
                f"Columns: {len(dataframe.columns)}"
            )

        db.flush()

        # ---------------------------------------------------------
        # 4. Build the same structure expected by Stage 04
        # ---------------------------------------------------------

        profiling_result = {
            "dataset_id": dataset.dataset_id,
            "dataset_name": dataset.dataset_name,
            "version_id": version.version_id,
            "version_number": version.version_number,
            "parent_version_id": None,
            "table_count": len(profiling_tables),
            "total_rows": sum(
                table["row_count"]
                for table in profiling_tables
            ),
            "total_columns": sum(
                table["column_count"]
                for table in profiling_tables
            ),
            "tables": profiling_tables,
        }

        # ---------------------------------------------------------
        # 5. Run existing Stage 04 semantic analysis
        # ---------------------------------------------------------

        service = SemanticDatasetAnalysisService()

        result = service.analyze_dataset(
            db=db,
            dataset=dataset,
            profiling_result=profiling_result,
        )
       

        # ---------------------------------------------------------
        # 6. Print results
        # ---------------------------------------------------------

        print("\n")
        print("=" * 60)
        print("SEMANTIC RESULTS")
        print("=" * 60)

        current_table = None

        for column in result["columns"]:
            table_name = column["table_name"]

            if table_name != current_table:
                current_table = table_name

                print(f"\nTABLE: {table_name}")
                print("-" * 60)

            semantic = column["semantic_analysis"]

            print(
                f"\nColumn: {column['column_name']}"
            )

            print(
                f"Concept: "
                f"{semantic['semantic_concept']}"
            )

            print(
                f"Confidence: "
                f"{semantic['confidence_score']:.4f}"
            )

            print(
                f"Level: "
                f"{semantic['confidence_level']}"
            )

            print("Alternatives:")

            for alternative in semantic.get(
                "alternatives",
                [],
            ):
                print(
                    f"  - "
                    f"{alternative['concept']}: "
                    f"{alternative['score']:.4f}"
                )

        # ---------------------------------------------------------
        # 7. Never persist the temporary validation records
        # ---------------------------------------------------------

        print("\n")
        print("=" * 60)
        print("VALIDATION COMPLETE")
        print("=" * 60)
        print(
            "Temporary validation records will be rolled back."
        )

        db.rollback()

    except Exception:
        db.rollback()
        raise

    finally:
        db.close()


if __name__ == "__main__":
    main()
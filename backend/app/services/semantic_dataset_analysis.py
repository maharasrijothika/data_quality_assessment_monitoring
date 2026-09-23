from sqlalchemy.orm import Session

from app.models import (
    ColumnMetadata,
    Dataset,
    DatasetVersion,
    TableMetadata,
)
from app.services.semantic_analysis import SemanticAnalysisService


class SemanticDatasetAnalysisService:
    """
    Orchestrates Stage 04 semantic analysis for all columns
    in a specific dataset version.

    Stage 03 profiling remains the source of truth for
    profiling information.
    """

    def __init__(self):
        self.semantic_service = SemanticAnalysisService()

    def analyze_dataset(
        self,
        db: Session,
        dataset: Dataset,
        profiling_result: dict,
    ) -> dict:
        """
        Run semantic analysis for every column returned
        by the Stage 03 profiling result.
        """

        version_id = profiling_result.get("version_id")

        if version_id is None:
            raise ValueError(
                "Stage 03 profiling result does not contain version_id."
            )

        # ---------------------------------------------------------
        # 1. Get the exact dataset version
        # ---------------------------------------------------------
        version = (
            db.query(DatasetVersion)
            .filter(
                DatasetVersion.version_id == version_id,
                DatasetVersion.dataset_id == dataset.dataset_id,
            )
            .first()
        )

        if version is None:
            raise ValueError(
                f"Dataset version {version_id} does not belong "
                f"to dataset {dataset.dataset_id}."
            )

        # ---------------------------------------------------------
        # 2. Load table metadata for this exact version
        # ---------------------------------------------------------
        tables = (
            db.query(TableMetadata)
            .filter(
                TableMetadata.version_id == version.version_id
            )
            .all()
        )

        table_lookup = {
            table.table_name.strip().lower(): table
            for table in tables
        }

        # ---------------------------------------------------------
        # 3. Load column metadata for those tables
        # ---------------------------------------------------------
        table_ids = [table.table_id for table in tables]

        column_lookup = {}

        if table_ids:
            columns = (
                db.query(ColumnMetadata)
                .filter(
                    ColumnMetadata.table_id.in_(table_ids)
                )
                .all()
            )

            column_lookup = {
                (
                    column.table_id,
                    column.column_name.strip().lower(),
                ): column
                for column in columns
            }

        # ---------------------------------------------------------
        # 4. Analyze every profiled table
        # ---------------------------------------------------------
        results = []

        profiling_tables = profiling_result.get(
            "tables",
            [],
        )

        for table_profile in profiling_tables:
            table_name = table_profile.get("table_name")

            if not table_name:
                continue

            table_record = table_lookup.get(
                table_name.strip().lower()
            )

            table_context = None

            if table_record is not None:
                table_context = table_record.description

            table_id = (
                table_record.table_id
                if table_record is not None
                else None
            )

            for profile in table_profile.get(
                "columns",
                [],
            ):
                column_name = profile.get("column_name")

                if not column_name:
                    continue

                # -------------------------------------------------
                # Match Stage 03 profile with ColumnMetadata
                # -------------------------------------------------
                metadata = None

                if table_id is not None:
                    metadata = column_lookup.get(
                        (
                            table_id,
                            column_name.strip().lower(),
                        )
                    )

                column_description = (
                    metadata.description
                    if metadata is not None
                    else None
                )

                data_type = profile.get(
                    "data_type"
                )

                # -------------------------------------------------
                # Run existing column-level semantic engine
                # -------------------------------------------------
                analysis = (
                    self.semantic_service.analyze_column(
                        db=db,
                        column_name=column_name,
                        column_description=column_description,
                        data_type=data_type,
                        profile=profile,
                        dataset_domain=dataset.domain,
                        table_context=table_context,
                        top_k=5,
                    )
                )

                results.append(
                    {
                        "table_name": table_name,
                        "column_name": column_name,
                        "description": column_description,
                        "data_type": data_type,
                        "semantic_analysis": analysis,
                    }
                )

        # ---------------------------------------------------------
        # 5. Return dataset-level Stage 04 result
        # ---------------------------------------------------------
        return {
            "dataset_id": dataset.dataset_id,
            "dataset_name": dataset.dataset_name,
            "version_id": version.version_id,
            "version_number": version.version_number,
            "domain": dataset.domain,
            "source_system": dataset.source_system,
            "update_cadence": dataset.update_cadence,
            "description": dataset.description,
            "column_count": len(results),
            "columns": results,
        }
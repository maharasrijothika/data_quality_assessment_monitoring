import re

from sqlalchemy.orm import Session

from app.models import (
    ColumnMetadata,
    Dataset,
    DatasetVersion,
    TableMetadata,
)
from app.config import (
    SEMANTIC_EMBEDDING_REPRESENTATION_VERSION,
    SEMANTIC_MODEL_NAME,
)
from app.services.abbreviation_lexicon import resolve_name
from app.services.semantic_analysis import SemanticAnalysisService
from app.services.semantic_features import normalize_text
from app.services.semantic_retrieval import (
    SemanticRetrievalService,
    encode_queries,
    get_embedding_model,
)


class SemanticDatasetAnalysisService:
    """
    Orchestrates Stage 04 semantic analysis for all columns
    in a specific dataset version.

    Stage 03 profiling remains the source of truth for
    profiling information.

    v2: sibling names are collected per table, every column query text is
    encoded in ONE batched model call per dataset run, and pre-computed
    query embeddings are handed to the column-level engine.
    """

    def __init__(self):
        self.semantic_service = SemanticAnalysisService()
        self.retrieval = SemanticRetrievalService()

    # ------------------------------------------------------------------
    # Relationship evidence (optional, read-only)
    # ------------------------------------------------------------------
    def _collect_relationship_hints(
        self,
        db: Session,
        version_id: int,
    ) -> dict[tuple[str, str], list[str]]:
        """Read-only relationship hints keyed by (table, column).

        Tolerant of absence: any failure returns an empty mapping so
        semantic analysis never depends on Stage 05 having run. Only
        explicit human-approved relationships become evidence hints.
        """
        hints: dict[tuple[str, str], list[str]] = {}
        try:
            from app.models import RelationshipCandidate

            rows = (
                db.query(RelationshipCandidate)
                .filter(
                    RelationshipCandidate.version_id == version_id,
                    RelationshipCandidate.status == "approved",
                )
                .all()
            )
        except Exception:
            return hints

        for row in rows:
            parent_hints = hints.setdefault(
                (row.parent_table, row.parent_column), []
            )
            parent_hints.append(f"pk of {row.parent_table}")
            child_hints = hints.setdefault(
                (row.child_table, row.child_column), []
            )
            child_hints.append(
                f"references {row.parent_table}.{row.parent_column}"
            )
        return hints

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

        def _resolve_table(name: str | None):
            """Match a profile table name against TableMetadata.

            Re-uploaded versions get a dedup suffix in the profile table name
            (file stem `dq_core_1`) while TableMetadata keeps the original
            (`dq_core`). Without this fallback the join fails silently and
            column descriptions + table context never reach the engine.
            """
            if not name:
                return None

            record = table_lookup.get(name.strip().lower())
            if record is not None:
                return record

            stripped = re.sub(r"(_\d+)+$", "", name.strip().lower()).strip()
            if stripped and stripped != name.strip().lower():
                return table_lookup.get(stripped)

            return None

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
        # 4. Optional relationship hints (Stage 05, read-only)
        # ---------------------------------------------------------
        relationship_hints = self._collect_relationship_hints(
            db, version.version_id
        )

        # ---------------------------------------------------------
        # 5. Flatten profiled columns; collect sibling names per table
        # ---------------------------------------------------------
        pending: list[dict] = []

        profiling_tables = profiling_result.get(
            "tables",
            [],
        )

        for table_profile in profiling_tables:
            table_name = table_profile.get("table_name")

            if not table_name:
                continue

            table_record = _resolve_table(table_name)

            table_context = None

            if table_record is not None:
                table_context = table_record.description

            table_id = (
                table_record.table_id
                if table_record is not None
                else None
            )

            profile_columns = list(
                table_profile.get("columns", [])
            )

            sibling_names = [
                column.get("column_name")
                for column in profile_columns
                if column.get("column_name")
            ]

            for profile in profile_columns:
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

                pending.append(
                    {
                        "table_name": table_name,
                        "table_context": table_context,
                        "column_name": column_name,
                        "column_description": column_description,
                        "data_type": data_type,
                        "profile": profile,
                        "sibling_names": sibling_names,
                    }
                )

        # ---------------------------------------------------------
        # 6. ONE batched encoding call for the whole dataset
        # ---------------------------------------------------------
        query_texts = [
            self.retrieval.build_column_text(
                column_name=entry["column_name"],
                column_description=entry["column_description"],
                table_context=entry["table_context"],
                sibling_names=entry["sibling_names"],
            )
            for entry in pending
        ]

        # Audit snapshot: the EXACT text every column contributed to the
        # embedding call, plus the model + representation version. Persisted
        # with the run so the embedding input is always reproducible.
        for index, entry in enumerate(pending):
            entry["embedding_text"] = query_texts[index]

        query_embeddings = None
        if query_texts:
            model = get_embedding_model()
            query_embeddings = encode_queries(model, query_texts)

        # ---------------------------------------------------------
        # 7. Analyze every profiled column (batched embeddings)
        # ---------------------------------------------------------
        results = []

        for index, entry in enumerate(pending):
            column_name = entry["column_name"]
            profile = entry["profile"]

            # -------------------------------------------------
            # Name normalization + abbreviation resolution evidence.
            # The original name is always preserved; normalized and
            # abbreviation-expanded forms are derived views only.
            # -------------------------------------------------
            name_resolution = resolve_name(column_name)

            # -------------------------------------------------
            # Run existing column-level semantic engine
            # -------------------------------------------------
            analysis = (
                self.semantic_service.analyze_column(
                    db=db,
                    column_name=column_name,
                    column_description=entry["column_description"],
                    data_type=entry["data_type"],
                    profile=profile,
                    dataset_domain=dataset.domain,
                    table_context=entry["table_context"],
                    top_k=5,
                    sibling_names=entry["sibling_names"],
                    relationship_hints=relationship_hints.get(
                        (entry["table_name"], column_name)
                    ),
                    query_embedding=(
                        query_embeddings[index]
                        if query_embeddings is not None
                        else None
                    ),
                )
            )

            # Normalization/abbreviation evidence travels with the
            # analysis payload so Stage 04 can display WHY a name was
            # matched a particular way.
            analysis["normalization"] = {
                "original_name": column_name,
                "normalized_name": normalize_text(column_name),
                "expanded_name": name_resolution.expanded_name,
                "abbreviation_evidence": name_resolution.abbreviation_evidence,
            }

            # Audit snapshot of the semantic input for THIS column.
            analysis["semantic_input"] = {
                "table_name": entry["table_name"],
                "column_name": column_name,
                "data_type": entry["data_type"],
                "column_description": entry["column_description"],
                "dataset_domain": dataset.domain,
                "source_system": dataset.source_system,
                "update_cadence": dataset.update_cadence,
                "dataset_description": dataset.description,
                "table_context": entry["table_context"],
                "sibling_names": entry["sibling_names"],
                "embedding_text": entry["embedding_text"],
                "embedding_model": SEMANTIC_MODEL_NAME,
                "embedding_representation_version": (
                    SEMANTIC_EMBEDDING_REPRESENTATION_VERSION
                ),
            }

            results.append(
                {
                    "table_name": entry["table_name"],
                    "column_name": column_name,
                    "description": entry["column_description"],
                    "data_type": entry["data_type"],
                    "semantic_analysis": analysis,
                }
            )

        # ---------------------------------------------------------
        # 8. Return dataset-level Stage 04 result
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

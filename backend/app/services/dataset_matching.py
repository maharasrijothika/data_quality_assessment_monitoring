from pathlib import Path

from sqlalchemy.orm import Session

from app.models import (
    ColumnMetadata,
    DatasetVersion,
    TableMetadata,
)


def calculate_column_similarity(
    incoming_columns: set[str],
    existing_columns: set[str],
) -> float:
    """
    Calculate Jaccard similarity between two column-name sets.
    """

    if not incoming_columns or not existing_columns:
        return 0.0

    intersection = incoming_columns & existing_columns
    union = incoming_columns | existing_columns

    return len(intersection) / len(union)


def find_possible_dataset_matches(
    db: Session,
    filename: str,
    tables: list[dict],
) -> list[dict]:
    """
    Find existing dataset versions that may correspond
    to an incoming file.

    This function only identifies possible matches.
    It does NOT automatically create a new dataset version.

    Matching evidence:
        - filename
        - table name
        - column similarity

    A match is only a candidate. Final version creation
    requires an explicit application decision.
    """

    incoming_stem = Path(filename).stem.lower()

    incoming_table_names = {
        str(table["table_name"]).lower()
        for table in tables
    }

    incoming_columns = {
        str(column["column_name"]).lower()
        for table in tables
        for column in table["columns"]
    }

    versions = (
        db.query(DatasetVersion)
        .order_by(
            DatasetVersion.dataset_id,
            DatasetVersion.version_number.desc(),
        )
        .all()
    )

    candidates = []

    for version in versions:

        version_tables = (
            db.query(TableMetadata)
            .filter(
                TableMetadata.version_id
                == version.version_id
            )
            .all()
        )

        if not version_tables:
            continue

        existing_table_names = {
            table.table_name.lower()
            for table in version_tables
        }

        existing_columns = set()

        for table in version_tables:

            columns = (
                db.query(ColumnMetadata)
                .filter(
                    ColumnMetadata.table_id
                    == table.table_id
                )
                .all()
            )

            existing_columns.update(
                column.column_name.lower()
                for column in columns
            )

        table_name_overlap = bool(
            incoming_table_names
            & existing_table_names
        )

        column_similarity = calculate_column_similarity(
            incoming_columns,
            existing_columns,
        )

        filename_match = any(
            incoming_stem
            == Path(table.source_file).stem.lower()
            for table in version_tables
        )

        evidence = []

        if filename_match:
            evidence.append("filename_match")

        if table_name_overlap:
            evidence.append("table_name_match")

        if column_similarity >= 0.80:
            evidence.append(
                "strong_column_similarity"
            )
        elif column_similarity >= 0.50:
            evidence.append(
                "moderate_column_similarity"
            )

        if not evidence:
            continue

        candidates.append(
            {
                "dataset_id": version.dataset_id,
                "version_id": version.version_id,
                "version_number": version.version_number,
                "filename_match": filename_match,
                "table_name_overlap": table_name_overlap,
                "column_similarity": round(
                    column_similarity,
                    4,
                ),
                "evidence": evidence,
            }
        )

    return candidates
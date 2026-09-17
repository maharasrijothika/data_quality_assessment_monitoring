from sqlalchemy.orm import Session

from app.models import DatasetVersion
from app.services.dataset_matching import (
    find_possible_dataset_matches,
)


def propose_dataset_version(
    db: Session,
    filename: str,
    tables: list[dict],
) -> list[dict]:
    """
    Identify existing dataset versions that may correspond
    to an incoming changed dataset.

    This function does not create or modify any database records.

    The returned candidates must be reviewed before a new
    dataset version is created.
    """

    candidates = find_possible_dataset_matches(
        db=db,
        filename=filename,
        tables=tables,
    )

    proposals = []

    for candidate in candidates:

        version = (
            db.query(DatasetVersion)
            .filter(
                DatasetVersion.version_id
                == candidate["version_id"]
            )
            .first()
        )

        if version is None:
            continue

        proposals.append(
            {
                "dataset_id": candidate["dataset_id"],
                "version_id": candidate["version_id"],
                "version_number": candidate[
                    "version_number"
                ],
                "proposed_parent_version_id": (
                    candidate["version_id"]
                ),
                "filename_match": candidate[
                    "filename_match"
                ],
                "table_name_overlap": candidate[
                    "table_name_overlap"
                ],
                "column_similarity": candidate[
                    "column_similarity"
                ],
                "evidence": candidate["evidence"],
            }
        )

    return proposals
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.models import (
    ColumnMetadata,
    Dataset,
    DatasetVersion,
    TableMetadata,
)
from app.services.dataset_matching import (
    calculate_column_similarity,
    find_possible_dataset_matches,
)


def create_test_db():
    """
    Create an isolated in-memory SQLite database
    for dataset matching tests.
    """
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={
            "check_same_thread": False,
        },
    )

    Base.metadata.create_all(bind=engine)

    TestSessionLocal = sessionmaker(
        autocommit=False,
        autoflush=False,
        bind=engine,
    )

    return TestSessionLocal()


def test_column_similarity():
    incoming = {
        "customer_id",
        "name",
        "email",
    }

    existing = {
        "customer_id",
        "name",
        "email",
    }

    similarity = calculate_column_similarity(
        incoming,
        existing,
    )

    assert similarity == 1.0


def test_column_similarity_partial_match():
    incoming = {
        "customer_id",
        "name",
        "email",
    }

    existing = {
        "customer_id",
        "name",
        "phone",
    }

    similarity = calculate_column_similarity(
        incoming,
        existing,
    )

    assert 0 < similarity < 1


def test_find_possible_dataset_matches():
    db = create_test_db()

    try:
        dataset = Dataset(
            dataset_name="Matching Test Dataset",
        )

        db.add(dataset)
        db.flush()

        version = DatasetVersion(
            dataset_id=dataset.dataset_id,
            version_number=1,
        )

        db.add(version)
        db.flush()

        table = TableMetadata(
            version_id=version.version_id,
            table_name="customers",
            source_file="customers.csv",
            sheet_name=None,
            row_count=3,
        )

        db.add(table)
        db.flush()

        columns = [
            ColumnMetadata(
                table_id=table.table_id,
                column_name="customer_id",
                data_type="int64",
            ),
            ColumnMetadata(
                table_id=table.table_id,
                column_name="name",
                data_type="object",
            ),
            ColumnMetadata(
                table_id=table.table_id,
                column_name="email",
                data_type="object",
            ),
        ]

        db.add_all(columns)
        db.commit()

        incoming_tables = [
            {
                "table_name": "customers",
                "source_file": "customers.csv",
                "sheet_name": None,
                "row_count": 3,
                "columns": [
                    {
                        "column_name": "customer_id",
                        "data_type": "int64",
                    },
                    {
                        "column_name": "name",
                        "data_type": "object",
                    },
                    {
                        "column_name": "email",
                        "data_type": "object",
                    },
                ],
            }
        ]

        matches = find_possible_dataset_matches(
            db,
            "customers.csv",
            incoming_tables,
        )

        assert len(matches) == 1

        match = matches[0]

        assert match["dataset_id"] == dataset.dataset_id
        assert match["version_id"] == version.version_id
        assert match["filename_match"] is True
        assert match["table_name_overlap"] is True
        assert match["column_similarity"] == 1.0

    finally:
        db.close()
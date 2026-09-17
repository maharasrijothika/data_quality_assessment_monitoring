from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.models import (
    ColumnMetadata,
    Dataset,
    DatasetVersion,
    TableMetadata,
)
from app.services.version_proposal import (
    propose_dataset_version,
)


def create_test_db():
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


def test_propose_dataset_version():
    db = create_test_db()

    try:
        dataset = Dataset(
            dataset_name="Customers",
        )

        db.add(dataset)
        db.flush()

        version = DatasetVersion(
            dataset_id=dataset.dataset_id,
            version_number=1,
            schema_fingerprint="schema_v1",
            content_fingerprint="content_v1",
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

        db.add_all(
            [
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
        )

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

        proposals = propose_dataset_version(
            db=db,
            filename="customers.csv",
            tables=incoming_tables,
        )

        assert len(proposals) == 1

        proposal = proposals[0]

        assert proposal["dataset_id"] == dataset.dataset_id
        assert proposal["version_id"] == version.version_id
        assert (
            proposal["proposed_parent_version_id"]
            == version.version_id
        )
        assert proposal["version_number"] == 1
        assert proposal["filename_match"] is True
        assert proposal["table_name_overlap"] is True
        assert proposal["column_similarity"] == 1.0

    finally:
        db.close()


def test_proposal_does_not_create_new_version():
    db = create_test_db()

    try:
        dataset = Dataset(
            dataset_name="Customers",
        )

        db.add(dataset)
        db.commit()

        version = DatasetVersion(
            dataset_id=dataset.dataset_id,
            version_number=1,
        )

        db.add(version)
        db.commit()

        incoming_tables = [
            {
                "table_name": "customers",
                "source_file": "customers.csv",
                "sheet_name": None,
                "row_count": 1,
                "columns": [
                    {
                        "column_name": "customer_id",
                        "data_type": "int64",
                    }
                ],
            }
        ]

        propose_dataset_version(
            db=db,
            filename="customers.csv",
            tables=incoming_tables,
        )

        versions = (
            db.query(DatasetVersion)
            .filter(
                DatasetVersion.dataset_id
                == dataset.dataset_id
            )
            .all()
        )

        assert len(versions) == 1

    finally:
        db.close()
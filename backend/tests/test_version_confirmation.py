from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.models import (
    ColumnMetadata,
    Dataset,
    DatasetVersion,
    FileMetadata,
    TableMetadata,
)
from app.services.version_confirmation import (
    confirm_dataset_version,
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


def create_parent_version(db):
    dataset = Dataset(
        dataset_name="Customers",
    )

    db.add(dataset)
    db.commit()

    version = DatasetVersion(
        dataset_id=dataset.dataset_id,
        version_number=1,
        schema_fingerprint="schema_v1",
        content_fingerprint="content_v1",
    )

    db.add(version)
    db.commit()

    return dataset, version


def test_confirm_creates_complete_version_snapshot(
    tmp_path,
    monkeypatch,
):
    import app.services.storage as storage

    monkeypatch.setattr(
        storage,
        "RAW_DATA_DIR",
        tmp_path / "datasets",
    )

    db = create_test_db()

    try:
        dataset, version_1 = create_parent_version(db)

        source_file = tmp_path / "customers.csv"

        source_file.write_text(
            "customer_id,email\n"
            "1,alice@example.com\n"
            "2,bob@example.com\n"
            "3,charlie@example.com\n"
            "4,david@example.com\n",
            encoding="utf-8",
        )

        files = [
            {
                "filename": "customers.csv",
                "source_path": source_file,
                "fingerprint": "content_v2",
                "tables": [
                    {
                        "table_name": "customers",
                        "source_file": "customers.csv",
                        "sheet_name": None,
                        "row_count": 4,
                        "columns": [
                            {
                                "column_name": "customer_id",
                                "data_type": "int64",
                            },
                            {
                                "column_name": "email",
                                "data_type": "object",
                                "description": "Customer email",
                            },
                        ],
                    }
                ],
            }
        ]

        version_2 = confirm_dataset_version(
            db=db,
            dataset_id=dataset.dataset_id,
            parent_version_id=version_1.version_id,
            schema_fingerprint="schema_v2",
            content_fingerprint="content_v2",
            files=files,
        )

        db.commit()

        assert version_2.version_number == 2

        assert (
            version_2.parent_version_id
            == version_1.version_id
        )

        stored_file_path = (
            tmp_path
            / "datasets"
            / str(dataset.dataset_id)
            / "v2"
            / "customers.csv"
        )

        assert stored_file_path.exists()

        stored_file = (
            db.query(FileMetadata)
            .filter(
                FileMetadata.version_id
                == version_2.version_id
            )
            .first()
        )

        assert stored_file is not None

        assert (
            stored_file.original_filename
            == "customers.csv"
        )

        assert (
            stored_file.stored_filename
            == "customers.csv"
        )

        assert (
            stored_file.content_fingerprint
            == "content_v2"
        )

        table = (
            db.query(TableMetadata)
            .filter(
                TableMetadata.version_id
                == version_2.version_id
            )
            .first()
        )

        assert table is not None
        assert table.row_count == 4

        columns = (
            db.query(ColumnMetadata)
            .filter(
                ColumnMetadata.table_id
                == table.table_id
            )
            .all()
        )

        assert len(columns) == 2

    finally:
        db.close()


def test_parent_version_remains_unchanged(
    tmp_path,
    monkeypatch,
):
    import app.services.storage as storage

    monkeypatch.setattr(
        storage,
        "RAW_DATA_DIR",
        tmp_path / "datasets",
    )

    db = create_test_db()

    try:
        dataset, version_1 = create_parent_version(db)

        original_schema = version_1.schema_fingerprint
        original_content = version_1.content_fingerprint

        source_file = tmp_path / "customers.csv"

        source_file.write_text(
            "customer_id\n"
            "1\n"
            "2\n"
            "3\n"
            "4\n",
            encoding="utf-8",
        )

        version_2 = confirm_dataset_version(
            db=db,
            dataset_id=dataset.dataset_id,
            parent_version_id=version_1.version_id,
            schema_fingerprint="schema_v2",
            content_fingerprint="content_v2",
            files=[
                {
                    "filename": "customers.csv",
                    "source_path": source_file,
                    "fingerprint": "content_v2",
                    "tables": [
                        {
                            "table_name": "customers",
                            "source_file": "customers.csv",
                            "sheet_name": None,
                            "row_count": 4,
                            "columns": [
                                {
                                    "column_name": "customer_id",
                                    "data_type": "int64",
                                }
                            ],
                        }
                    ],
                }
            ],
        )

        db.commit()

        db.refresh(version_1)

        assert (
            version_1.schema_fingerprint
            == original_schema
        )

        assert (
            version_1.content_fingerprint
            == original_content
        )

        assert version_2.version_number == 2

        stored_file_path = (
            tmp_path
            / "datasets"
            / str(dataset.dataset_id)
            / "v2"
            / "customers.csv"
        )

        assert stored_file_path.exists()

    finally:
        db.close()


def test_confirmation_is_rollback_safe(
    tmp_path,
    monkeypatch,
):
    import app.services.storage as storage

    monkeypatch.setattr(
        storage,
        "RAW_DATA_DIR",
        tmp_path / "datasets",
    )

    db = create_test_db()

    try:
        dataset, version_1 = create_parent_version(db)

        source_file = tmp_path / "customers.csv"

        source_file.write_text(
            "customer_id\n"
            "1\n"
            "2\n",
            encoding="utf-8",
        )

        try:
            confirm_dataset_version(
                db=db,
                dataset_id=dataset.dataset_id,
                parent_version_id=version_1.version_id,
                schema_fingerprint="schema_v2",
                content_fingerprint="content_v2",
                files=[
                    {
                        "filename": "customers.csv",
                        "source_path": source_file,
                        "fingerprint": "content_v2",
                        "tables": [
                            {
                                "table_name": "customers",
                                "source_file": "customers.csv",
                                "sheet_name": None,
                                "row_count": 2,
                                "columns": [],
                            }
                        ],
                    }
                ],
            )

            raise ValueError(
                "Simulated snapshot failure"
            )

        except ValueError:
            db.rollback()

        versions = (
            db.query(DatasetVersion)
            .filter(
                DatasetVersion.dataset_id
                == dataset.dataset_id
            )
            .all()
        )

        assert len(versions) == 1

        assert (
            versions[0].version_number
            == 1
        )

        files = (
            db.query(FileMetadata)
            .all()
        )

        assert len(files) == 0

    finally:
        db.close()
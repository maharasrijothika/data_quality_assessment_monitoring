import pytest
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
from app.services.version_snapshot import (
    register_version_snapshot,
)


@pytest.fixture
def db():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
    )

    Base.metadata.create_all(engine)

    SessionLocal = sessionmaker(
        autocommit=False,
        autoflush=False,
        bind=engine,
    )

    session = SessionLocal()

    try:
        yield session
    finally:
        session.close()
        engine.dispose()


def create_dataset_and_version(db):
    dataset = Dataset(
        dataset_name="Test Dataset",
    )

    db.add(dataset)
    db.flush()

    version = DatasetVersion(
        dataset_id=dataset.dataset_id,
        parent_version_id=None,
        version_number=1,
        schema_fingerprint="schema123",
        content_fingerprint="content123",
    )

    db.add(version)
    db.flush()

    return dataset, version


def test_register_version_snapshot(
    db,
    tmp_path,
    monkeypatch,
):
    """
    A version snapshot should register the file,
    table and column metadata and store the raw file
    under the correct version directory.
    """

    import app.services.storage as storage

    monkeypatch.setattr(
        storage,
        "RAW_DATA_DIR",
        tmp_path / "datasets",
    )

    dataset, version = create_dataset_and_version(db)

    source_file = tmp_path / "customers.csv"

    source_file.write_text(
        "customer_id,name\n1,Alice\n",
        encoding="utf-8",
    )

    files = [
        {
            "filename": "customers.csv",
            "source_path": source_file,
            "fingerprint": "abc123",
            "tables": [
                {
                    "table_name": "customers",
                    "source_file": "customers.csv",
                    "sheet_name": None,
                    "row_count": 1,
                    "columns": [
                        {
                            "column_name": "customer_id",
                            "data_type": "int64",
                        },
                        {
                            "column_name": "name",
                            "data_type": "object",
                        },
                    ],
                }
            ],
        }
    ]

    result = register_version_snapshot(
        db=db,
        dataset_id=dataset.dataset_id,
        version_id=version.version_id,
        version_number=version.version_number,
        files=files,
    )

    db.commit()

    assert len(result) == 1

    stored_file = (
        tmp_path
        / "datasets"
        / str(dataset.dataset_id)
        / "v1"
        / "customers.csv"
    )

    assert stored_file.exists()

    assert (
        stored_file.read_text(encoding="utf-8")
        == "customer_id,name\n1,Alice\n"
    )

    file_metadata = (
        db.query(FileMetadata)
        .filter(
            FileMetadata.version_id
            == version.version_id
        )
        .one()
    )

    assert file_metadata.original_filename == "customers.csv"
    assert file_metadata.stored_filename == "customers.csv"
    assert file_metadata.content_fingerprint == "abc123"

    table_metadata = (
        db.query(TableMetadata)
        .filter(
            TableMetadata.version_id
            == version.version_id
        )
        .one()
    )

    assert table_metadata.table_name == "customers"
    assert table_metadata.row_count == 1

    columns = (
        db.query(ColumnMetadata)
        .filter(
            ColumnMetadata.table_id
            == table_metadata.table_id
        )
        .all()
    )

    assert len(columns) == 2


def test_version_snapshot_is_version_isolated(
    db,
    tmp_path,
    monkeypatch,
):
    """
    V1 and V2 must have independent physical files
    and metadata snapshots.
    """

    import app.services.storage as storage

    monkeypatch.setattr(
        storage,
        "RAW_DATA_DIR",
        tmp_path / "datasets",
    )

    dataset, version1 = create_dataset_and_version(db)

    source_v1 = tmp_path / "customers_v1.csv"

    source_v1.write_text(
        "customer_id,name\n1,Alice\n",
        encoding="utf-8",
    )

    register_version_snapshot(
        db=db,
        dataset_id=dataset.dataset_id,
        version_id=version1.version_id,
        version_number=version1.version_number,
        files=[
            {
                "filename": "customers.csv",
                "source_path": source_v1,
                "fingerprint": "fingerprint-v1",
                "tables": [
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
                ],
            }
        ],
    )

    version2 = DatasetVersion(
        dataset_id=dataset.dataset_id,
        parent_version_id=version1.version_id,
        version_number=2,
        schema_fingerprint="schema456",
        content_fingerprint="content456",
    )

    db.add(version2)
    db.flush()

    source_v2 = tmp_path / "customers_v2.csv"

    source_v2.write_text(
        "customer_id,name\n1,Alice\n2,Bob\n",
        encoding="utf-8",
    )

    register_version_snapshot(
        db=db,
        dataset_id=dataset.dataset_id,
        version_id=version2.version_id,
        version_number=version2.version_number,
        files=[
            {
                "filename": "customers.csv",
                "source_path": source_v2,
                "fingerprint": "fingerprint-v2",
                "tables": [
                    {
                        "table_name": "customers",
                        "source_file": "customers.csv",
                        "sheet_name": None,
                        "row_count": 2,
                        "columns": [
                            {
                                "column_name": "customer_id",
                                "data_type": "int64",
                            },
                            {
                                "column_name": "name",
                                "data_type": "object",
                            },
                        ],
                    }
                ],
            }
        ],
    )

    db.commit()

    v1_file = (
        tmp_path
        / "datasets"
        / str(dataset.dataset_id)
        / "v1"
        / "customers.csv"
    )

    v2_file = (
        tmp_path
        / "datasets"
        / str(dataset.dataset_id)
        / "v2"
        / "customers.csv"
    )

    assert v1_file.exists()
    assert v2_file.exists()

    assert (
        v1_file.read_text(encoding="utf-8")
        == "customer_id,name\n1,Alice\n"
    )

    assert (
        v2_file.read_text(encoding="utf-8")
        == "customer_id,name\n1,Alice\n2,Bob\n"
    )

    v1_metadata = (
        db.query(FileMetadata)
        .filter(
            FileMetadata.version_id
            == version1.version_id
        )
        .one()
    )

    v2_metadata = (
        db.query(FileMetadata)
        .filter(
            FileMetadata.version_id
            == version2.version_id
        )
        .one()
    )

    assert (
        v1_metadata.content_fingerprint
        == "fingerprint-v1"
    )

    assert (
        v2_metadata.content_fingerprint
        == "fingerprint-v2"
    )


def test_snapshot_fails_without_source_path(db):
    """
    A snapshot cannot register a physical raw file
    without knowing its source path.
    """

    dataset, version = create_dataset_and_version(db)

    files = [
        {
            "filename": "customers.csv",
            "fingerprint": "abc123",
            "tables": [],
        }
    ]

    with pytest.raises(
        ValueError,
        match="source_path is required",
    ):
        register_version_snapshot(
            db=db,
            dataset_id=dataset.dataset_id,
            version_id=version.version_id,
            version_number=version.version_number,
            files=files,
        )
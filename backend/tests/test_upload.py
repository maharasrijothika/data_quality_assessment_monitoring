
import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool
from sqlalchemy.orm import sessionmaker

from app.database import Base, get_db
from app.main import app
from app.models import (
    ColumnMetadata,
    Dataset,
    DatasetVersion,
    FileMetadata,
    TableMetadata,
)


@pytest.fixture
def db():
    engine = create_engine(
        "sqlite://",
        connect_args={
            "check_same_thread": False,
        },
        poolclass=StaticPool,
    )

    Base.metadata.create_all(bind=engine)

    TestSessionLocal = sessionmaker(
        autocommit=False,
        autoflush=False,
        bind=engine,
    )

    session = TestSessionLocal()

    try:
        yield session
    finally:
        session.close()
        engine.dispose()


@pytest.fixture
def client(db):
    def override_get_db():
        try:
            yield db
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db

    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.clear()


@pytest.fixture
def isolated_storage(tmp_path, monkeypatch):
    """
    Redirect raw dataset storage to a temporary directory
    so tests never modify the real data/raw directory.
    """

    import app.services.storage as storage

    raw_data_dir = tmp_path / "datasets"

    monkeypatch.setattr(
        storage,
        "RAW_DATA_DIR",
        raw_data_dir,
    )

    return raw_data_dir


def upload_file(
    client,
    filename="customers.csv",
    content=(
        "customer_id,name,email\n"
        "1,Alice,alice@example.com\n"
        "2,Bob,bob@example.com\n"
    ),
    dataset_name="Customers",
):
    """
    Helper for uploading a CSV file through the real API.
    """

    return client.post(
        "/datasets/upload",
        data={
            "dataset_name": dataset_name,
            "source_system": "CRM",
            "description": "Customer data",
        },
        files={
            "files": (
                filename,
                io.BytesIO(content.encode("utf-8")),
                "text/csv",
            )
        },
    )


def test_upload_creates_version_1_under_v1_storage(
    client,
    db,
    isolated_storage,
):
    """
    First upload should create dataset version 1 and
    physically store the immutable raw file under:

        data/raw/datasets/{dataset_id}/v1/
    """

    response = upload_file(client)

    assert response.status_code == 200

    data = response.json()

    assert data["dataset_id"] == 1
    assert data["version_id"] == 1
    assert data["version_number"] == 1

    stored_file = (
        isolated_storage
        / "1"
        / "v1"
        / "customers.csv"
    )

    assert stored_file.exists()

    assert (
        stored_file.read_text(encoding="utf-8")
        == (
            "customer_id,name,email\n"
            "1,Alice,alice@example.com\n"
            "2,Bob,bob@example.com\n"
        )
    )

    dataset = (
        db.query(Dataset)
        .filter(
            Dataset.dataset_id == 1
        )
        .one()
    )

    assert dataset.dataset_name == "Customers"
    assert dataset.source_system == "CRM"
    assert dataset.description == "Customer data"


def test_upload_creates_correct_database_snapshot(
    client,
    db,
    isolated_storage,
):
    """
    Verify that upload creates:

        Dataset
        DatasetVersion
        FileMetadata
        TableMetadata
        ColumnMetadata
    """

    response = upload_file(client)

    assert response.status_code == 200

    version = (
        db.query(DatasetVersion)
        .filter(
            DatasetVersion.version_id == 1
        )
        .one()
    )

    assert version.dataset_id == 1
    assert version.version_number == 1
    assert version.parent_version_id is None

    assert version.schema_fingerprint is not None
    assert version.content_fingerprint is not None

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
    assert file_metadata.file_extension == ".csv"
    assert len(file_metadata.content_fingerprint) == 64

    table = (
        db.query(TableMetadata)
        .filter(
            TableMetadata.version_id
            == version.version_id
        )
        .one()
    )

    assert table.table_name == "customers"
    assert table.source_file == "customers.csv"
    assert table.sheet_name is None
    assert table.row_count == 2

    columns = (
        db.query(ColumnMetadata)
        .filter(
            ColumnMetadata.table_id
            == table.table_id
        )
        .order_by(
            ColumnMetadata.column_id
        )
        .all()
    )

    assert len(columns) == 3

    assert columns[0].column_name == "customer_id"
    assert columns[1].column_name == "name"
    assert columns[2].column_name == "email"


def test_duplicate_file_is_rejected(
    client,
    db,
    isolated_storage,
):
    """
    Uploading the exact same file content again should be
    rejected using the content fingerprint.

    No second dataset/version should be created.
    """

    first_response = upload_file(
        client,
        filename="customers.csv",
    )

    assert first_response.status_code == 200

    second_response = upload_file(
        client,
        filename="customers_copy.csv",
    )

    assert second_response.status_code == 409

    detail = second_response.json()["detail"]

    assert (
        "already exist"
        in detail["message"]
    )

    assert len(detail["duplicate_files"]) == 1

    duplicate = detail["duplicate_files"][0]

    assert duplicate["filename"] == "customers_copy.csv"
    assert (
        duplicate["content_fingerprint"]
        == first_response.json()["files"][0][
            "content_fingerprint"
        ]
    )

    versions = (
        db.query(DatasetVersion)
        .all()
    )

    assert len(versions) == 1

    datasets = (
        db.query(Dataset)
        .all()
    )

    assert len(datasets) == 1

    second_stored_file = (
        isolated_storage
        / "2"
        / "v1"
        / "customers_copy.csv"
    )

    assert not second_stored_file.exists()


def test_same_filename_with_different_content_is_not_duplicate(
    client,
    db,
    isolated_storage,
):
    """
    A same filename does not mean duplicate content.

    Different content must produce a different fingerprint.
    """

    first_content = (
        "customer_id,name\n"
        "1,Alice\n"
    )

    second_content = (
        "customer_id,name\n"
        "1,Alice\n"
        "2,Bob\n"
    )

    first_response = upload_file(
        client,
        filename="customers.csv",
        content=first_content,
    )

    assert first_response.status_code == 200

    second_response = upload_file(
        client,
        filename="customers.csv",
        content=second_content,
        dataset_name="Customers Updated",
    )

    assert second_response.status_code == 200

    first_fingerprint = (
        first_response.json()["files"][0][
            "content_fingerprint"
        ]
    )

    second_fingerprint = (
        second_response.json()["files"][0][
            "content_fingerprint"
        ]
    )

    assert first_fingerprint != second_fingerprint

    assert (
        db.query(Dataset)
        .count()
        == 2
    )

    assert (
        db.query(DatasetVersion)
        .count()
        == 2
    )

    v1_file = (
        isolated_storage
        / "1"
        / "v1"
        / "customers.csv"
    )

    v2_dataset_file = (
        isolated_storage
        / "2"
        / "v1"
        / "customers.csv"
    )

    assert v1_file.exists()
    assert v2_dataset_file.exists()

    assert (
        v1_file.read_text(
            encoding="utf-8"
        )
        == first_content
    )

    assert (
        v2_dataset_file.read_text(
            encoding="utf-8"
        )
        == second_content
    )


def test_failed_upload_does_not_leave_database_records(
    client,
    db,
    isolated_storage,
):
    """
    An invalid upload should not leave partial database
    records behind.
    """

    response = client.post(
        "/datasets/upload",
        data={
            "dataset_name": "Invalid Dataset",
            "source_system": "CRM",
            "description": "Invalid file",
        },
        files={
            "files": (
                "invalid.txt",
                io.BytesIO(
                    b"this is not a supported file"
                ),
                "text/plain",
            )
        },
    )

    assert response.status_code == 400

    assert (
        db.query(Dataset)
        .count()
        == 0
    )

    assert (
        db.query(DatasetVersion)
        .count()
        == 0
    )

    assert (
        db.query(FileMetadata)
        .count()
        == 0
    )

    assert (
        db.query(TableMetadata)
        .count()
        == 0
    )

    assert (
        db.query(ColumnMetadata)
        .count()
        == 0
    )

    if isolated_storage.exists():
        assert not any(
            isolated_storage.rglob("*")
        )


def test_empty_file_is_rejected(
    client,
    db,
    isolated_storage,
):
    """
    Empty files must be rejected before dataset creation.
    """

    response = client.post(
        "/datasets/upload",
        data={
            "dataset_name": "Empty Dataset",
        },
        files={
            "files": (
                "empty.csv",
                io.BytesIO(b""),
                "text/csv",
            )
        },
    )

    assert response.status_code == 400

    assert (
        db.query(Dataset)
        .count()
        == 0
    )

    assert (
        db.query(DatasetVersion)
        .count()
        == 0
    )

    if isolated_storage.exists():
        assert not any(
            isolated_storage.rglob("*")
        )


def test_unsupported_extension_is_rejected(
    client,
    db,
    isolated_storage,
):
    """
    Unsupported file extensions must be rejected.
    """

    response = client.post(
        "/datasets/upload",
        data={
            "dataset_name": "Unsupported Dataset",
        },
        files={
            "files": (
                "data.json",
                io.BytesIO(
                    b'{"id": 1}'
                ),
                "application/json",
            )
        },
    )

    assert response.status_code == 400

    assert (
        db.query(Dataset)
        .count()
        == 0
    )

    assert (
        db.query(DatasetVersion)
        .count()
        == 0
    )

    if isolated_storage.exists():
        assert not any(
            isolated_storage.rglob("*")
        )


def test_multiple_files_create_single_dataset_version(
    client,
    db,
    isolated_storage,
):
    """
    Multiple related files uploaded together should
    belong to one dataset and one version.
    """

    customers = (
        "customer_id,name\n"
        "1,Alice\n"
        "2,Bob\n"
    )

    orders = (
        "order_id,customer_id,total\n"
        "100,1,500\n"
        "101,2,700\n"
    )

    response = client.post(
        "/datasets/upload",
        data={
            "dataset_name": "Ecommerce",
            "source_system": "ERP",
            "description": "Ecommerce related tables",
        },
        files=[
            (
                "files",
                (
                    "customers.csv",
                    io.BytesIO(
                        customers.encode("utf-8")
                    ),
                    "text/csv",
                ),
            ),
            (
                "files",
                (
                    "orders.csv",
                    io.BytesIO(
                        orders.encode("utf-8")
                    ),
                    "text/csv",
                ),
            ),
        ],
    )

    assert response.status_code == 200

    data = response.json()

    assert data["dataset_id"] == 1
    assert data["version_id"] == 1
    assert data["version_number"] == 1

    assert len(data["files"]) == 2

    assert (
        db.query(Dataset)
        .count()
        == 1
    )

    assert (
        db.query(DatasetVersion)
        .count()
        == 1
    )

    assert (
        db.query(FileMetadata)
        .count()
        == 2
    )

    assert (
        db.query(TableMetadata)
        .count()
        == 2
    )

    customers_file = (
        isolated_storage
        / "1"
        / "v1"
        / "customers.csv"
    )

    orders_file = (
        isolated_storage
        / "1"
        / "v1"
        / "orders.csv"
    )

    assert customers_file.exists()
    assert orders_file.exists()

    assert (
        customers_file.read_text(
            encoding="utf-8"
        )
        == customers
    )

    assert (
        orders_file.read_text(
            encoding="utf-8"
        )
        == orders
    )


def test_duplicate_upload_does_not_create_physical_raw_file(
    client,
    db,
    isolated_storage,
):
    """
    When duplicate detection rejects an upload,
    no physical raw file should be stored for
    the rejected upload.
    """

    content = (
        "customer_id,name\n"
        "1,Alice\n"
    )

    first_response = upload_file(
        client,
        filename="customers.csv",
        content=content,
    )

    assert first_response.status_code == 200

    second_response = upload_file(
        client,
        filename="customers_duplicate.csv",
        content=content,
    )

    assert second_response.status_code == 409

    dataset_directories = [
        path
        for path in isolated_storage.iterdir()
        if path.is_dir()
    ]

    assert len(dataset_directories) == 1

    dataset_dir = dataset_directories[0]

    version_directories = [
        path
        for path in dataset_dir.iterdir()
        if path.is_dir()
    ]

    assert len(version_directories) == 1

    files = [
        path
        for path in version_directories[0].iterdir()
        if path.is_file()
    ]

    assert len(files) == 1
    assert files[0].name == "customers.csv"


def test_upload_rejects_missing_dataset_name(
    client,
    db,
    isolated_storage,
):
    """
    dataset_name is required for a dataset registration.
    """

    response = client.post(
        "/datasets/upload",
        files={
            "files": (
                "customers.csv",
                io.BytesIO(
                    b"customer_id,name\n1,Alice\n"
                ),
                "text/csv",
            )
        },
    )

    assert response.status_code == 422

    assert (
        db.query(Dataset)
        .count()
        == 0
    )

    assert (
        db.query(DatasetVersion)
        .count()
        == 0
    )

    if isolated_storage.exists():
        assert not any(
            isolated_storage.rglob("*")
        )


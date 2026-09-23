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
def dataset_with_version(db):
    dataset = Dataset(
        dataset_name="Test Customer Dataset",
        source_system="Test CRM",
        description="Original dataset description",
    )

    db.add(dataset)
    db.flush()

    version = DatasetVersion(
        dataset_id=dataset.dataset_id,
        parent_version_id=None,
        version_number=1,
        schema_fingerprint="schema-v1",
        content_fingerprint="content-v1",
    )

    db.add(version)
    db.flush()

    table = TableMetadata(
        version_id=version.version_id,
        table_name="customers",
        source_file="customers.csv",
        sheet_name=None,
        row_count=3,
        description=None,
    )

    db.add(table)
    db.flush()

    columns = [
        ColumnMetadata(
            table_id=table.table_id,
            column_name="customer_id",
            data_type="int64",
            description=None,
        ),
        ColumnMetadata(
            table_id=table.table_id,
            column_name="name",
            data_type="object",
            description=None,
        ),
        ColumnMetadata(
            table_id=table.table_id,
            column_name="email",
            data_type="object",
            description=None,
        ),
    ]

    db.add_all(columns)
    db.commit()

    return {
        "dataset": dataset,
        "version": version,
        "table": table,
        "columns": columns,
    }


def test_get_dataset_context(
    client,
    dataset_with_version,
):
    dataset = dataset_with_version["dataset"]

    response = client.get(
        f"/datasets/{dataset.dataset_id}/context"
    )

    assert response.status_code == 200

    data = response.json()

    assert data["dataset_id"] == dataset.dataset_id
    assert data["dataset_name"] == "Test Customer Dataset"
    assert data["source_system"] == "Test CRM"
    assert data["description"] == "Original dataset description"

    assert data["domain"] is None
    assert data["update_cadence"] is None

    assert data["current_version"]["version_number"] == 1

    assert len(data["tables"]) == 1
    assert data["tables"][0]["table_name"] == "customers"

    assert len(data["tables"][0]["columns"]) == 3


def test_update_dataset_level_context(
    client,
    dataset_with_version,
):
    dataset = dataset_with_version["dataset"]

    payload = {
        "description": "Customer master dataset.",
        "domain": "Customer Management",
        "source_system": "CRM",
        "update_cadence": "Daily",
        "tables": [],
    }

    response = client.put(
        f"/datasets/{dataset.dataset_id}/context",
        json=payload,
    )

    assert response.status_code == 200

    data = response.json()

    assert data["message"] == "Dataset context updated successfully."
    assert data["dataset_id"] == dataset.dataset_id
    assert data["current_version"]["version_number"] == 1

    updated_dataset = (
        client.get(
            f"/datasets/{dataset.dataset_id}/context"
        )
        .json()
    )

    assert updated_dataset["description"] == "Customer master dataset."
    assert updated_dataset["domain"] == "Customer Management"
    assert updated_dataset["source_system"] == "CRM"
    assert updated_dataset["update_cadence"] == "Daily"


def test_update_table_and_column_context(
    client,
    dataset_with_version,
):
    dataset = dataset_with_version["dataset"]
    table = dataset_with_version["table"]
    columns = dataset_with_version["columns"]

    payload = {
        "description": "Customer master dataset.",
        "domain": "Customer Management",
        "source_system": "CRM",
        "update_cadence": "Daily",
        "tables": [
            {
                "table_id": table.table_id,
                "description": (
                    "Customer table containing "
                    "customer identity and contact information."
                ),
                "columns": [
                    {
                        "column_id": columns[0].column_id,
                        "description": (
                            "Unique identifier assigned "
                            "to each customer."
                        ),
                    },
                    {
                        "column_id": columns[1].column_id,
                        "description": (
                            "Customer's full name."
                        ),
                    },
                    {
                        "column_id": columns[2].column_id,
                        "description": (
                            "Customer email address."
                        ),
                    },
                ],
            }
        ],
    }

    response = client.put(
        f"/datasets/{dataset.dataset_id}/context",
        json=payload,
    )

    assert response.status_code == 200

    context = client.get(
        f"/datasets/{dataset.dataset_id}/context"
    ).json()

    table_data = context["tables"][0]

    assert (
        table_data["description"]
        == "Customer table containing "
        "customer identity and contact information."
    )

    assert (
        table_data["columns"][0]["description"]
        == "Unique identifier assigned "
        "to each customer."
    )

    assert (
        table_data["columns"][1]["description"]
        == "Customer's full name."
    )

    assert (
        table_data["columns"][2]["description"]
        == "Customer email address."
    )


def test_invalid_table_id_is_rejected(
    client,
    dataset_with_version,
):
    dataset = dataset_with_version["dataset"]

    payload = {
        "description": "Test dataset.",
        "domain": "Testing",
        "source_system": "Test CRM",
        "update_cadence": "Daily",
        "tables": [
            {
                "table_id": 999999,
                "description": "Invalid table.",
                "columns": [],
            }
        ],
    }

    response = client.put(
        f"/datasets/{dataset.dataset_id}/context",
        json=payload,
    )

    assert response.status_code == 400


def test_invalid_column_id_is_rejected(
    client,
    dataset_with_version,
):
    dataset = dataset_with_version["dataset"]
    table = dataset_with_version["table"]

    payload = {
        "description": "Test dataset.",
        "domain": "Testing",
        "source_system": "Test CRM",
        "update_cadence": "Daily",
        "tables": [
            {
                "table_id": table.table_id,
                "description": "Customer table.",
                "columns": [
                    {
                        "column_id": 999999,
                        "description": "Invalid column.",
                    }
                ],
            }
        ],
    }

    response = client.put(
        f"/datasets/{dataset.dataset_id}/context",
        json=payload,
    )

    assert response.status_code == 400


def test_context_only_updates_current_version(
    client,
    db,
    dataset_with_version,
):
    dataset = dataset_with_version["dataset"]
    first_version = dataset_with_version["version"]

    second_version = DatasetVersion(
        dataset_id=dataset.dataset_id,
        parent_version_id=first_version.version_id,
        version_number=2,
        schema_fingerprint="schema-v2",
        content_fingerprint="content-v2",
    )

    db.add(second_version)
    db.flush()

    second_table = TableMetadata(
        version_id=second_version.version_id,
        table_name="customers_v2",
        source_file="customers_v2.csv",
        sheet_name=None,
        row_count=5,
        description=None,
    )

    db.add(second_table)
    db.flush()

    second_column = ColumnMetadata(
        table_id=second_table.table_id,
        column_name="customer_id",
        data_type="int64",
        description=None,
    )

    db.add(second_column)
    db.commit()

    payload = {
        "description": "Updated dataset.",
        "domain": "Customer Management",
        "source_system": "CRM",
        "update_cadence": "Daily",
        "tables": [
            {
                "table_id": second_table.table_id,
                "description": "Version 2 customer table.",
                "columns": [
                    {
                        "column_id": second_column.column_id,
                        "description": "Version 2 customer ID.",
                    }
                ],
            }
        ],
    }

    response = client.put(
        f"/datasets/{dataset.dataset_id}/context",
        json=payload,
    )

    assert response.status_code == 200

    context = client.get(
        f"/datasets/{dataset.dataset_id}/context"
    ).json()

    assert context["current_version"]["version_number"] == 2
    assert len(context["tables"]) == 1

    assert (
        context["tables"][0]["table_name"]
        == "customers_v2"
    )

    assert (
        context["tables"][0]["description"]
        == "Version 2 customer table."
    )

    assert (
        context["tables"][0]["columns"][0]["description"]
        == "Version 2 customer ID."
    )

    original_table = db.get(
        TableMetadata,
        dataset_with_version["table"].table_id,
    )

    original_column = db.get(
        ColumnMetadata,
        dataset_with_version["columns"][0].column_id,
    )

    assert original_table.description is None
    assert original_column.description is None


def test_dataset_not_found(
    client,
):
    response = client.get(
        "/datasets/999999/context"
    )

    assert response.status_code == 404
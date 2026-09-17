from app.services.storage import (
    create_version_storage,
    get_unique_filename,
    store_raw_file,
)


def test_create_version_storage(tmp_path, monkeypatch):
    """
    Version storage should create:

        dataset_id/
            v1/
    """

    import app.services.storage as storage

    monkeypatch.setattr(
        storage,
        "RAW_DATA_DIR",
        tmp_path / "datasets",
    )

    version_dir = create_version_storage(
        dataset_id=1,
        version_number=1,
    )

    assert version_dir.exists()
    assert version_dir.is_dir()

    assert version_dir == (
        tmp_path / "datasets" / "1" / "v1"
    )


def test_create_multiple_version_storage(
    tmp_path,
    monkeypatch,
):
    """
    Different dataset versions must have
    separate physical directories.
    """

    import app.services.storage as storage

    monkeypatch.setattr(
        storage,
        "RAW_DATA_DIR",
        tmp_path / "datasets",
    )

    v1 = create_version_storage(
        dataset_id=1,
        version_number=1,
    )

    v2 = create_version_storage(
        dataset_id=1,
        version_number=2,
    )

    assert v1 != v2

    assert v1.exists()
    assert v2.exists()

    assert v1.name == "v1"
    assert v2.name == "v2"


def test_same_filename_allowed_across_versions(
    tmp_path,
    monkeypatch,
):
    """
    The same filename should be allowed in
    different dataset versions.
    """

    import app.services.storage as storage

    monkeypatch.setattr(
        storage,
        "RAW_DATA_DIR",
        tmp_path / "datasets",
    )

    source = tmp_path / "customers.csv"

    source.write_text(
        "customer_id,name\n1,Alice\n",
        encoding="utf-8",
    )

    v1_file = store_raw_file(
        source_path=source,
        dataset_id=1,
        version_number=1,
        filename="customers.csv",
    )

    v2_file = store_raw_file(
        source_path=source,
        dataset_id=1,
        version_number=2,
        filename="customers.csv",
    )

    assert v1_file.name == "customers.csv"
    assert v2_file.name == "customers.csv"

    assert v1_file != v2_file

    assert v1_file.exists()
    assert v2_file.exists()


def test_no_overwrite_within_same_version(
    tmp_path,
    monkeypatch,
):
    """
    Uploading the same filename twice into the
    same version must not overwrite the first file.
    """

    import app.services.storage as storage

    monkeypatch.setattr(
        storage,
        "RAW_DATA_DIR",
        tmp_path / "datasets",
    )

    source = tmp_path / "customers.csv"

    source.write_text(
        "customer_id,name\n1,Alice\n",
        encoding="utf-8",
    )

    first = store_raw_file(
        source_path=source,
        dataset_id=1,
        version_number=1,
        filename="customers.csv",
    )

    second = store_raw_file(
        source_path=source,
        dataset_id=1,
        version_number=1,
        filename="customers.csv",
    )

    assert first.name == "customers.csv"
    assert second.name == "customers_1.csv"

    assert first.exists()
    assert second.exists()


def test_v1_is_not_overwritten_by_v2(
    tmp_path,
    monkeypatch,
):
    """
    Creating V2 must never modify V1.
    """

    import app.services.storage as storage

    monkeypatch.setattr(
        storage,
        "RAW_DATA_DIR",
        tmp_path / "datasets",
    )

    source_v1 = tmp_path / "customers_v1.csv"

    source_v1.write_text(
        "customer_id,name\n1,Alice\n",
        encoding="utf-8",
    )

    v1_file = store_raw_file(
        source_path=source_v1,
        dataset_id=1,
        version_number=1,
        filename="customers.csv",
    )

    original_content = v1_file.read_text(
        encoding="utf-8"
    )

    source_v2 = tmp_path / "customers_v2.csv"

    source_v2.write_text(
        "customer_id,name\n1,Alice\n2,Bob\n",
        encoding="utf-8",
    )

    v2_file = store_raw_file(
        source_path=source_v2,
        dataset_id=1,
        version_number=2,
        filename="customers.csv",
    )

    assert v1_file.exists()
    assert v2_file.exists()

    assert (
        v1_file.read_text(encoding="utf-8")
        == original_content
    )

    assert v1_file != v2_file
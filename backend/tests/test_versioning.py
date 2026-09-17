from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.models import Dataset, DatasetVersion
from app.services.versioning import (
    create_dataset_version,
    get_next_version_number,
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


def test_first_version_number():
    db = create_test_db()

    try:
        dataset = Dataset(
            dataset_name="Version Test",
        )

        db.add(dataset)
        db.commit()

        assert (
            get_next_version_number(
                db,
                dataset.dataset_id,
            )
            == 1
        )

    finally:
        db.close()


def test_version_numbers_increment():
    db = create_test_db()

    try:
        dataset = Dataset(
            dataset_name="Version Test",
        )

        db.add(dataset)
        db.commit()

        version_1 = create_dataset_version(
            db=db,
            dataset_id=dataset.dataset_id,
            parent_version_id=None,
            schema_fingerprint="schema_v1",
            content_fingerprint="content_v1",
        )

        db.commit()

        assert version_1.version_number == 1
        assert version_1.parent_version_id is None

        version_2 = create_dataset_version(
            db=db,
            dataset_id=dataset.dataset_id,
            parent_version_id=version_1.version_id,
            schema_fingerprint="schema_v2",
            content_fingerprint="content_v2",
        )

        db.commit()

        assert version_2.version_number == 2
        assert (
            version_2.parent_version_id
            == version_1.version_id
        )

    finally:
        db.close()


def test_existing_version_is_not_modified():
    db = create_test_db()

    try:
        dataset = Dataset(
            dataset_name="Immutable Version Test",
        )

        db.add(dataset)
        db.commit()

        version_1 = create_dataset_version(
            db=db,
            dataset_id=dataset.dataset_id,
            parent_version_id=None,
            schema_fingerprint="schema_v1",
            content_fingerprint="content_v1",
        )

        db.commit()

        original_content = (
            version_1.content_fingerprint
        )

        version_2 = create_dataset_version(
            db=db,
            dataset_id=dataset.dataset_id,
            parent_version_id=version_1.version_id,
            schema_fingerprint="schema_v2",
            content_fingerprint="content_v2",
        )

        db.commit()

        db.refresh(version_1)

        assert (
            version_1.content_fingerprint
            == original_content
        )

        assert version_1.version_number == 1
        assert version_2.version_number == 2

    finally:
        db.close()

def test_invalid_parent_version_is_rejected():
    db = create_test_db()

    try:
        dataset = Dataset(
            dataset_name="Invalid Parent Test",
        )

        db.add(dataset)
        db.commit()

        try:
            create_dataset_version(
                db=db,
                dataset_id=dataset.dataset_id,
                parent_version_id=999,
                schema_fingerprint="schema_v2",
                content_fingerprint="content_v2",
            )

            assert False, (
                "Expected invalid parent version "
                "to raise ValueError"
            )

        except ValueError as exc:
            assert "does not exist" in str(exc)

    finally:
        db.close()


def test_parent_version_must_belong_to_same_dataset():
    db = create_test_db()

    try:
        dataset_1 = Dataset(
            dataset_name="Dataset One",
        )

        dataset_2 = Dataset(
            dataset_name="Dataset Two",
        )

        db.add_all(
            [
                dataset_1,
                dataset_2,
            ]
        )

        db.commit()

        version_1 = create_dataset_version(
            db=db,
            dataset_id=dataset_1.dataset_id,
            parent_version_id=None,
            schema_fingerprint="schema_v1",
            content_fingerprint="content_v1",
        )

        db.commit()

        try:
            create_dataset_version(
                db=db,
                dataset_id=dataset_2.dataset_id,
                parent_version_id=version_1.version_id,
                schema_fingerprint="schema_v2",
                content_fingerprint="content_v2",
            )

            assert False, (
                "Expected cross-dataset parent "
                "version to raise ValueError"
            )

        except ValueError as exc:
            assert (
                "different dataset"
                in str(exc)
            )

    finally:
        db.close()
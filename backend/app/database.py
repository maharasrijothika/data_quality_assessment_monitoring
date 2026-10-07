from pathlib import Path

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import declarative_base, sessionmaker


BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

DATABASE_URL = f"sqlite:///{DATA_DIR / 'dq_assessment.db'}"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)

Base = declarative_base()

# Idempotent light-weight column guard for the existing SQLite database.
# create_all() creates NEW tables with every column, but never adds columns
# to tables that already exist; these nullable additions must be present on
# upgrades too. Plain SQLite ALTER TABLE, checked via PRAGMA first.
_REQUIRED_COLUMNS: dict[str, list[tuple[str, str]]] = {
    "datasets": [
        ("deleted_at", "DATETIME"),
    ],
    "semantic_predictions": [
        ("user_defined_type", "TEXT"),
        ("decision_source", "TEXT"),
        ("decided_at", "DATETIME"),
    ],
    "relationship_candidates": [
        ("candidate_kind", "TEXT"),
    ],
    "rules": [
        ("rule_code", "TEXT"),
        ("version_number", "INTEGER"),
        ("updated_at", "DATETIME"),
    ],
}


def ensure_sqlite_columns() -> None:
    """Add any missing _REQUIRED_COLUMNS entries to existing SQLite tables."""
    inspector = inspect(engine)

    with engine.connect() as connection:
        for table_name, columns in _REQUIRED_COLUMNS.items():
            if table_name not in inspector.get_table_names():
                continue

            existing = {
                column["name"] for column in inspector.get_columns(table_name)
            }


            for column_name, column_type in columns:
                if column_name not in existing:
                    connection.execute(
                        text(
                            f"ALTER TABLE {table_name} "
                            f"ADD COLUMN {column_name} {column_type}"
                        )
                    )

        connection.commit()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
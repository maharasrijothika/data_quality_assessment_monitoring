from pathlib import Path
import shutil


BASE_DIR = Path(__file__).resolve().parents[3]

RAW_DATA_DIR = BASE_DIR / "data" / "raw" / "datasets"


def create_dataset_storage(dataset_id: int) -> Path:
    """
    Create the root storage directory for a dataset.
    """

    dataset_dir = RAW_DATA_DIR / str(dataset_id)

    dataset_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    return dataset_dir


def create_version_storage(
    dataset_id: int,
    version_number: int,
) -> Path:
    """
    Create an immutable storage directory for a
    specific dataset version.

    Example:

        data/raw/datasets/1/v1/
        data/raw/datasets/1/v2/
    """

    if version_number < 1:
        raise ValueError("Version number must be >= 1.")

    dataset_dir = create_dataset_storage(dataset_id)

    version_dir = dataset_dir / f"v{version_number}"

    version_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    return version_dir


def get_unique_filename(
    storage_dir: Path,
    filename: str,
) -> str:
    """
    Prevent overwriting a file within the same
    dataset version.
    """

    safe_filename = Path(filename).name

    original_path = storage_dir / safe_filename

    if not original_path.exists():
        return safe_filename

    stem = original_path.stem
    suffix = original_path.suffix

    counter = 1

    while True:
        new_filename = f"{stem}_{counter}{suffix}"
        new_path = storage_dir / new_filename

        if not new_path.exists():
            return new_filename

        counter += 1


def store_raw_file(
    source_path: str | Path,
    dataset_id: int,
    version_number: int,
    filename: str,
) -> Path:
    """
    Store an immutable raw file inside its dataset version.

    Example:

        dataset_id=1
        version_number=2
        filename="customers.csv"

    becomes:

        data/raw/datasets/1/v2/customers.csv
    """

    source_path = Path(source_path)

    if not source_path.exists():
        raise FileNotFoundError(
            f"Source file not found: {source_path}"
        )

    version_dir = create_version_storage(
        dataset_id=dataset_id,
        version_number=version_number,
    )

    safe_filename = get_unique_filename(
        version_dir,
        filename,
    )

    destination = version_dir / safe_filename

    shutil.copy2(
        source_path,
        destination,
    )

    return destination
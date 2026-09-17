from pathlib import Path
import hashlib


HASH_ALGORITHM = "sha256"
CHUNK_SIZE = 1024 * 1024  # 1 MB


def calculate_file_fingerprint(
    file_path: str | Path,
) -> str:
    """
    Calculate a SHA-256 fingerprint for a file.

    The file is read in chunks so that large files do not
    need to be loaded entirely into memory.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {path}"
        )

    hasher = hashlib.sha256()

    with path.open("rb") as file:
        while chunk := file.read(CHUNK_SIZE):
            hasher.update(chunk)

    return hasher.hexdigest()
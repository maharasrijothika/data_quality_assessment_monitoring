from pathlib import Path, PurePosixPath


SUPPORTED_EXTENSIONS = {
    ".csv",
    ".xls",
    ".xlsx",
    ".parquet",
}

MAX_FILE_SIZE_MB = 100
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024


def validate_filename(filename: str) -> None:
    """
    Validate that an uploaded filename or relative folder path is safe
    and has a supported extension.

    Folder uploads may provide relative paths such as:
        products/flipkart_laptops.csv

    Path traversal and absolute paths are rejected.
    """
    if not filename:
        raise ValueError("Filename is required.")

    # Normalize browser-provided Windows separators to POSIX separators.
    normalized = filename.replace("\\", "/")

    path = PurePosixPath(normalized)

    # Reject absolute paths.
    if path.is_absolute():
        raise ValueError("Invalid filename.")

    # Reject path traversal.
    if ".." in path.parts:
        raise ValueError("Invalid filename.")

    # Reject empty/current-directory names.
    if normalized in {".", "..", ""}:
        raise ValueError("Invalid filename.")

    # The actual file component must exist.
    basename = path.name

    if not basename or basename in {".", ".."}:
        raise ValueError("Invalid filename.")

    extension = Path(basename).suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file format: {extension or 'no extension'}. "
            f"Supported formats: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )


def validate_file_size(file_size: int) -> None:
    """
    Validate that the uploaded file is within the allowed size limit.
    """
    if file_size <= 0:
        raise ValueError("Uploaded file is empty.")

    if file_size > MAX_FILE_SIZE_BYTES:
        raise ValueError(
            f"File exceeds the maximum allowed size of "
            f"{MAX_FILE_SIZE_MB} MB."
        )


def validate_uploaded_file(filename: str, file_size: int) -> None:
    """
    Run all basic validation checks for an uploaded file.
    """
    validate_filename(filename)
    validate_file_size(file_size)
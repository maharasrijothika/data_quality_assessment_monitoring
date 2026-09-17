from pathlib import Path


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
    Validate that the uploaded filename is safe and has a supported extension.
    """
    if not filename:
        raise ValueError("Filename is required.")

    path = Path(filename)

    if path.name != filename:
        raise ValueError("Invalid filename.")

    if filename in {".", ".."}:
        raise ValueError("Invalid filename.")

    extension = path.suffix.lower()

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
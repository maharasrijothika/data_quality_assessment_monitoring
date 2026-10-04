from pathlib import Path
import re

import pandas as pd


SUPPORTED_EXTENSIONS = {
    ".csv",
    ".xls",
    ".xlsx",
    ".parquet",
}

# Code-like column NAME tokens (same set as profiling Part D3). Only these
# columns are candidates for the leading-zero-preserving re-read.
_CODE_LIKE_NAME_TOKENS = {
    "zip",
    "zipcode",
    "postal",
    "postcode",
    "pin",
    "pincode",
    "phone",
    "mobile",
    "tel",
    "telephone",
    "fax",
    "id",
    "code",
    "sku",
    "serial",
    "account",
    "acct",
    "ssn",
    "ref",
    "reference",
}

_TOKEN_SPLIT_RE = re.compile(r"[^A-Za-z0-9]+")

# A leading-zero string: digits only, starts with 0, more than one digit.
_LEADING_ZERO_RE = re.compile(r"^0\d+$")


def _column_name_has_code_token(name: str) -> bool:
    """True when any name token matches a code-like token (word boundaries)."""
    text = str(name)
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", text)
    tokens = [token.lower() for token in _TOKEN_SPLIT_RE.split(text) if token]
    return any(token in _CODE_LIKE_NAME_TOKENS for token in tokens)


def _restore_leading_zero_columns(
    dataframe: pd.DataFrame,
    path: Path,
    sheet_name: str | int | None,
    encoding: str | None,
) -> pd.DataFrame:
    """Re-read code-like numeric columns as strings when zeros were lost.

    Conservative: a column is replaced by its raw string form ONLY when the
    raw text shows digits-only values starting with 0 (length > 1) — i.e.
    information was actually lost to numeric inference. Otherwise the
    numeric column is kept untouched. Empty strings become NaN.
    """
    candidates = [
        column
        for column in dataframe.columns
        if pd.api.types.is_numeric_dtype(dataframe[column])
        and not pd.api.types.is_bool_dtype(dataframe[column])
        and _column_name_has_code_token(str(column))
    ]

    if not candidates:
        return dataframe

    try:
        if path.suffix.lower() in {".xls", ".xlsx"}:
            raw = pd.read_excel(
                path,
                sheet_name=sheet_name,
                dtype=str,
                usecols=list(candidates),
            )
        else:
            raw = pd.read_csv(
                path,
                dtype=str,
                usecols=list(candidates),
                encoding=encoding or "utf-8",
            )
    except (ValueError, OSError, UnicodeDecodeError):
        # Column-list mismatch or unreadable raw text: keep the numeric read.
        return dataframe

    for column in candidates:
        if column not in raw.columns:
            continue

        raw_values = raw[column].dropna().astype(str).str.strip()
        raw_values = raw_values[raw_values != ""]

        if not raw_values.empty and bool(raw_values.str.fullmatch(_LEADING_ZERO_RE).any()):
            string_values = raw[column].astype("string").str.strip()
            string_values = string_values.replace("", pd.NA)
            dataframe[column] = string_values

    return dataframe


def read_table(
    file_path: str | Path,
    sheet_name: str | int | None = None,
) -> pd.DataFrame:
    """
    Read one logical table from a supported data file.

    Code-like numeric columns (zip/postal/code/...) are re-read from the
    raw text as strings when leading zeros were actually lost; everything
    else keeps pandas' normal inference.
    """
    path = Path(file_path)
    extension = path.suffix.lower()

    if extension == ".csv":
        encoding_used = "utf-8"
        try:
            dataframe = pd.read_csv(path, encoding=encoding_used)
        except UnicodeDecodeError:
            encoding_used = "cp1252"
            try:
                dataframe = pd.read_csv(path, encoding=encoding_used)
            except UnicodeDecodeError:
                encoding_used = "latin-1"
                dataframe = pd.read_csv(path, encoding=encoding_used)

        return _restore_leading_zero_columns(
            dataframe, path, None, encoding_used
        )

    if extension == ".parquet":
        return pd.read_parquet(path)

    if extension in {".xls", ".xlsx"}:
        dataframe = pd.read_excel(path, sheet_name=sheet_name)

        if isinstance(dataframe, pd.DataFrame):
            return _restore_leading_zero_columns(
                dataframe, path, sheet_name, None
            )

        return dataframe

    raise ValueError(
        f"Unsupported file format: {extension}"
    )


def discover_excel_sheets(
    file_path: str | Path,
) -> list[str]:
    """
    Return all sheet names from an Excel workbook.
    """
    path = Path(file_path)

    if path.suffix.lower() not in {".xls", ".xlsx"}:
        raise ValueError("File is not an Excel workbook.")

    workbook = pd.ExcelFile(path)

    return workbook.sheet_names


def discover_tables(
    file_path: str | Path,
) -> list[dict]:
    """
    Discover logical tables and their columns from a supported file.

    CSV/Parquet:
        One file = one logical table.

    Excel:
        One sheet = one logical table.
    """
    path = Path(file_path)
    extension = path.suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file format: {extension}"
        )

    tables = []

    if extension in {".csv", ".parquet"}:
        dataframe = read_table(path)

        tables.append(
            {
                "table_name": path.stem,
                "source_file": path.name,
                "sheet_name": None,
                "row_count": len(dataframe),
                "columns": [
                    {
                        "column_name": str(column),
                        "data_type": str(dataframe[column].dtype),
                    }
                    for column in dataframe.columns
                ],
            }
        )

        return tables

    sheet_names = discover_excel_sheets(path)

    for sheet_name in sheet_names:
        dataframe = read_table(
            path,
            sheet_name=sheet_name,
        )

        tables.append(
            {
                "table_name": str(sheet_name),
                "source_file": path.name,
                "sheet_name": str(sheet_name),
                "row_count": len(dataframe),
                "columns": [
                    {
                        "column_name": str(column),
                        "data_type": str(dataframe[column].dtype),
                    }
                    for column in dataframe.columns
                ],
            }
        )

    return tables


def discover_multiple_files(
    file_paths: list[str | Path],
) -> list[dict]:
    """
    Discover tables across multiple uploaded files.
    """
    discovered_tables = []

    for file_path in file_paths:
        discovered_tables.extend(
            discover_tables(file_path)
        )

    return discovered_tables
def discover_folder(
    folder_path: str | Path,
) -> list[dict]:
    """
    Discover tables from all supported files in a folder.

    Files directly inside the folder are processed.
    Unsupported files are ignored here and will be rejected
    later by the upload validation layer.
    """
    folder = Path(folder_path)

    if not folder.exists():
        raise ValueError("Folder does not exist.")

    if not folder.is_dir():
        raise ValueError("Provided path is not a folder.")

    file_paths = [
        path
        for path in folder.iterdir()
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    if not file_paths:
        raise ValueError(
            "No supported data files were found in the folder."
        )

    return discover_multiple_files(file_paths)
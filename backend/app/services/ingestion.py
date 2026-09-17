from pathlib import Path

import pandas as pd


SUPPORTED_EXTENSIONS = {
    ".csv",
    ".xls",
    ".xlsx",
    ".parquet",
}


def read_table(
    file_path: str | Path,
    sheet_name: str | int | None = None,
) -> pd.DataFrame:
    """
    Read one logical table from a supported data file.
    """
    path = Path(file_path)
    extension = path.suffix.lower()

    if extension == ".csv":
        return pd.read_csv(path)

    if extension == ".parquet":
        return pd.read_parquet(path)

    if extension in {".xls", ".xlsx"}:
        return pd.read_excel(path, sheet_name=sheet_name)

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
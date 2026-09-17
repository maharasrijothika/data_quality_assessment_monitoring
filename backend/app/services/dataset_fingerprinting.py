from hashlib import sha256


def calculate_schema_fingerprint(
    tables: list[dict],
) -> str:
    """
    Calculate a deterministic SHA-256 fingerprint
    representing the logical schema of a dataset.

    Table and column ordering is preserved because the
    logical table structure is part of the dataset schema.
    """

    schema_parts = []

    for table in tables:
        schema_parts.append(
            f"TABLE:{table['table_name']}"
        )

        for column in table["columns"]:
            schema_parts.append(
                f"COLUMN:{column['column_name']}|"
                f"TYPE:{column['data_type']}"
            )

    schema_text = "\n".join(schema_parts)

    return sha256(
        schema_text.encode("utf-8")
    ).hexdigest()


def calculate_dataset_content_fingerprint(
    file_fingerprints: list[tuple[str, str]],
) -> str:
    """
    Calculate a deterministic SHA-256 fingerprint for
    the complete uploaded dataset.

    Each tuple contains:
        (filename, file_content_fingerprint)

    Files are sorted by filename so that upload order does
    not change the resulting dataset fingerprint.
    """

    normalized = sorted(
        file_fingerprints,
        key=lambda item: item[0].lower(),
    )

    content_text = "\n".join(
        f"{filename}|{fingerprint}"
        for filename, fingerprint in normalized
    )

    return sha256(
        content_text.encode("utf-8")
    ).hexdigest()
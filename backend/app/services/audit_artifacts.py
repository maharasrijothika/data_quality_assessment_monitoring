"""Audit artifact exporters (reproducibility requirement).

The profiling, semantic and relationship stages already persist their
results in the database. These helpers EXPORT the exact stored payloads to
machine-readable JSON files under ``artifacts/`` so every run can be
inspected without touching the application:

    artifacts/profiling/<dataset_id>/<version_id>/profile.json
    artifacts/semantic/<dataset_id>/<version_id>/semantic_input.json
    artifacts/relationships/<dataset_id>/<version_id>/relationships.json
    artifacts/semantic_kb.json

Nothing is recalculated: every writer serializes data the engines already
produced.
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


def write_json(relative_path: str, payload) -> Path:
    """Serialize ``payload`` as JSON under artifacts/ (UTF-8, no truncation)."""
    path = ARTIFACTS_DIR / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return path


def export_profile_artifact(
    dataset_id: int,
    version_id: int,
    profile_result: dict,
) -> Path:
    """Export the EXACT stored profiling result (no recalculation)."""
    return write_json(
        f"profiling/{dataset_id}/{version_id}/profile.json",
        profile_result,
    )


def export_semantic_input_artifact(
    dataset_id: int,
    version_id: int,
    run_number: int,
    run_payload: dict,
) -> Path:
    """Export the semantic run snapshot (inputs, embedding texts, decisions)."""
    return write_json(
        f"semantic/{dataset_id}/{version_id}/semantic_input.json",
        run_payload,
    )


def export_relationships_artifact(
    dataset_id: int,
    version_id: int,
    result: dict,
) -> Path:
    """Export the relationship discovery result (all evidence signals)."""
    return write_json(
        f"relationships/{dataset_id}/{version_id}/relationships.json",
        result,
    )


def export_kb_inventory(concepts: list[dict]) -> Path:
    """Export the COMPLETE knowledge base (one JSON object per concept)."""
    return write_json("semantic_kb.json", {"concept_count": len(concepts), "concepts": concepts})

"""AUDIT RUNNER (read-only against the DB).

Runs the EXISTING semantic engine (SemanticDatasetAnalysisService.analyze_dataset)
for every runnable version (stored profile present) and stores the complete
predictions + evidence + candidates as JSON artifacts under
artifacts/semantic_audit_results/.

No code/model/KB changes: this imports the production engine as-is and never
commits the DB session. The engine's own idempotent embedding backfill
(ensure_kb_embeddings) may maintain semantic_concept_embeddings — that is
normal system behavior of any semantic run, not a code change.

Run from repo root:  .venv/Scripts/python.exe scripts/audit/run_semantic_audit.py
"""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.database import SessionLocal  # noqa: E402
from app.models import Dataset, DatasetVersion, StoredProfile  # noqa: E402
from app.services.semantic_dataset_analysis import (  # noqa: E402
    SemanticDatasetAnalysisService,
)
from app.services.semantic_stage import confidence_level  # noqa: E402

OUT_DIR = ROOT / "artifacts" / "semantic_audit_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    db = SessionLocal()
    svc = SemanticDatasetAnalysisService()

    profiles = db.query(StoredProfile).all()
    version_ids = sorted({p.version_id for p in profiles})
    print(f"Stored profiles for versions: {version_ids}")

    version_rows = db.query(DatasetVersion).all()
    ds_by_version = {v.version_id: v.dataset_id for v in version_rows}

    summary = []
    for vid in version_ids:
        stored = next(p for p in profiles if p.version_id == vid)
        dataset_id = ds_by_version.get(vid)
        dataset = (
            db.query(Dataset).filter(Dataset.dataset_id == dataset_id).first()
            if dataset_id is not None
            else None
        )
        if dataset is None:
            print(f"[{vid}] dataset {dataset_id} missing, skipped")
            continue

        profile_json = json.loads(json.dumps(stored.profile_json))  # deep copy
        profile_json.setdefault("version_id", vid)

        t0 = time.time()
        try:
            analysis = svc.analyze_dataset(
                db=db, dataset=dataset, profiling_result=profile_json
            )
        except Exception as exc:  # noqa: BLE001 - audit must not die mid-run
            print(f"[{vid}] FAILED: {type(exc).__name__}: {exc}")
            summary.append({"version_id": vid, "dataset_id": dataset.dataset_id, "status": "failed", "error": str(exc)})
            continue

        columns = []
        for col in analysis.get("columns", []):
            s = col.get("semantic_analysis", {}) or {}
            coverage = s.get("evidence_coverage", 0.0)
            score = s.get("confidence_score", 0.0)
            decision_kind = s.get("decision_kind", "UNKNOWN")
            margin = s.get("margin")
            columns.append(
                {
                    "table_name": col.get("table_name"),
                    "column_name": col.get("column_name"),
                    "data_type": col.get("data_type"),
                    "column_description": col.get("description"),
                    "predicted_concept": s.get("semantic_concept"),
                    "concept_id": s.get("concept_id"),
                    "confidence_score": score,
                    "evidence_coverage": coverage,
                    "confidence_level": confidence_level(score, coverage, margin=margin, decision_kind=decision_kind),
                    "decision_kind": decision_kind,
                    "decision_rule": s.get("decision_rule"),
                    "source": s.get("source"),
                    "match_status": s.get("match_status"),
                    "semantic_family": s.get("semantic_family"),
                    "margin": margin,
                    "exact_stage": s.get("exact_stage", False),
                    "exact_but_profile_conflict": s.get("exact_but_profile_conflict", False),
                    "proposal": s.get("proposal"),
                    "normalization": s.get("normalization"),
                    "evidence": s.get("evidence", {}),
                    "candidates": s.get("candidates", []),
                }
            )

        payload = {
            "dataset_id": dataset.dataset_id,
            "dataset_name": dataset.dataset_name,
            "version_id": vid,
            "column_count": len(columns),
            "columns": columns,
        }
        out = OUT_DIR / f"predictions_v{vid}.json"
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

        n_kb = sum(1 for c in columns if c["decision_kind"] == "KB_MATCH")
        n_fam = sum(1 for c in columns if c["decision_kind"] == "FAMILY_MATCH")
        n_unk = sum(1 for c in columns if c["decision_kind"] == "UNKNOWN")
        print(
            f"[{vid}] {dataset.dataset_name}: {len(columns)} cols "
            f"(KB={n_kb} FAMILY={n_fam} UNKNOWN={n_unk}) in {time.time() - t0:.1f}s -> {out.name}"
        )
        summary.append(
            {
                "version_id": vid,
                "dataset_id": dataset.dataset_id,
                "dataset_name": dataset.dataset_name,
                "status": "ok",
                "column_count": len(columns),
                "kb_matches": n_kb,
                "family_matches": n_fam,
                "unknown": n_unk,
            }
        )

    db.rollback()  # read-only guarantee: never persist anything
    db.close()

    (OUT_DIR / "run_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"\nDone. {len(summary)} versions -> {OUT_DIR}")


if __name__ == "__main__":
    main()

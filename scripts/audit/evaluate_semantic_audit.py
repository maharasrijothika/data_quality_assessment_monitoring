"""AUDIT EVALUATION (read-only; no DB writes, no code changes).

1. Parses docs/semantic_kb_inventory.md as the knowledge base (concept names
   + approved aliases).
2. Loads the audit prediction artifacts produced by run_semantic_audit.py.
3. Compares semantic predictions against backend/data/semantic_eval/
   semantic_gold_labels.csv (the gold labels confirmed by the user).
4. Classifies every labeled column:
      kb_name_path = unique_alias | alias_ambiguous | open_world
   where "open_world" means NO exact-name and NO approved-alias match exists
   in the KB — the only columns where the user wants XGBoost to be used.
5. Scores the deterministic engine per labeled column:
      deterministic_correct = (predicted concept == gold kb_acceptable),
      with multi-target gold ("Price|Amount") counted correct when the
      prediction is ANY of the listed targets.
6. Emits artifacts/semantic_audit_results/eval_columns.json (one record per
   labeled column, with evidence + candidates for the XGBoost stage),
   kb_match_summary.json, and prints a summary.

Run from repo root:  .venv/Scripts/python.exe scripts/audit/evaluate_semantic_audit.py
"""

import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AUDIT_DIR = ROOT / "artifacts" / "semantic_audit_results"
GOLD_CSV = ROOT / "backend" / "data" / "semantic_eval" / "semantic_gold_labels.csv"
KB_MD = ROOT / "docs" / "semantic_kb_inventory.md"

sys.path.insert(0, str(ROOT / "backend"))


# ---------------------------------------------------------------------------
# KB from docs/semantic_kb_inventory.md
# ---------------------------------------------------------------------------

def parse_kb_inventory() -> dict:
    """Extract (concept name -> id, category, aliases) from the MD inventory."""
    text = KB_MD.read_text(encoding="utf-8")
    kb: dict[str, dict] = {}
    header_re = re.compile(r"^##\s+(.+?)\s+\(id (\d+), category: (.+?)\)\s*$")
    alias_re = re.compile(r"^-\s+Aliases:\s*(.+)$")
    current: dict | None = None
    for line in text.splitlines():
        m = header_re.match(line.strip())
        if m:
            current = {
                "concept_id": int(m.group(2)),
                "category": m.group(3).strip(),
                "aliases": [],
            }
            kb[m.group(1).strip()] = current
            continue
        if current is not None:
            a = alias_re.match(line.strip())
            if a:
                current["aliases"] = [s.strip() for s in a.group(1).split(",") if s.strip()]
    return kb


def norm(s: str | None) -> str:
    """Lowercase, strip non-alphanumerics — same shape as the engine's key."""
    return re.sub(r"[^a-z0-9]+", "", (s or "").lower())


def build_alias_index(kb: dict) -> dict[str, list[str]]:
    """normalized surface form -> list of concept names claiming it."""
    index: dict[str, list[str]] = defaultdict(list)
    for concept_name, info in kb.items():
        for surface in [concept_name] + info["aliases"]:
            index[norm(surface)].append(concept_name)
    return dict(index)


# ---------------------------------------------------------------------------
# Gold labels
# ---------------------------------------------------------------------------

def load_gold() -> list[dict]:
    rows = []
    with open(GOLD_CSV, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            row["dataset_id"] = int(row["dataset"])
            rows.append(row)
    return rows


def version_key(version_str: str) -> int | None:
    """'v1' -> 1, '' -> 1 (version_number 1). Returns the version NUMBER."""
    s = (version_str or "").strip().lower()
    if not s:
        return 1
    m = re.match(r"^v(\d+)$", s)
    return int(m.group(1)) if m else None


def main() -> None:
    kb = parse_kb_inventory()
    alias_index = build_alias_index(kb)
    print(f"KB parsed: {len(kb)} concepts, {sum(len(v['aliases']) + 1 for v in kb.values())} surface forms")

    gold_rows = load_gold()
    print(f"Gold rows: {len(gold_rows)}")

    # dataset_id + version_number -> prediction lookup (table, column) -> pred
    predictions: dict[tuple[int, int], dict[tuple[str, str], dict]] = {}
    for f in sorted(AUDIT_DIR.glob("predictions_v*.json")):
        payload = json.loads(f.read_text(encoding="utf-8"))
        vid = payload["version_id"]
        # find version_number via dataset_versions mapping baked into run summary
        predictions[(payload["dataset_id"], vid)] = {
            (c["table_name"], c["column_name"]): c for c in payload["columns"]
        }

    # Build dataset_id -> {version_number: version_id} from the audit payloads
    # (payload files are keyed by version_id; we need dataset->versions.)
    import sqlite3

    con = sqlite3.connect(str(ROOT / "data" / "dq_assessment.db"))
    version_map: dict[tuple[int, int], int] = {}  # (dataset_id, version_number) -> version_id
    for ds_id, vid, vnum in con.execute(
        "SELECT dataset_id, version_id, version_number FROM dataset_versions"
    ):
        version_map[(ds_id, vnum)] = vid
    con.close()

    records = []
    matched_pred = 0
    for g in gold_rows:
        ds_id = g["dataset_id"]
        vnum = version_key(g["version"])
        vid = version_map.get((ds_id, vnum)) if vnum is not None else None
        key = (g["table"].strip(), g["column"].strip())
        preds = predictions.get((ds_id, vid), {}) if vid else {}

        pred = preds.get(key)
        if pred is None:
            # tolerant fallback: case-insensitive match within the version
            for (t, c), p in preds.items():
                if t.strip().lower() == key[0].lower() and c.strip().lower() == key[1].lower():
                    pred = p
                    break
        if pred is None:
            records.append(
                {
                    "dataset_id": ds_id,
                    "version_id": vid,
                    "table": key[0],
                    "column": key[1],
                    "gold_concept": g["gold_concept"],
                    "gold_family": g["gold_family"],
                    "gold_kb": g["kb_acceptable"],
                    "confidence": g["confidence"],
                    "split": g["split"],
                    "kb_name_path": "no_prediction",
                    "deterministic_correct": None,
                    "predicted_concept": None,
                    "note": "no semantic prediction found (version not run or column absent)",
                }
            )
            continue

        matched_pred += 1
        # ---- KB name path for the COLUMN NAME ----
        col_norm = norm(g["column"])
        kb_hits = alias_index.get(col_norm, [])
        if len(kb_hits) == 1:
            kb_name_path = "unique_alias"
        elif len(kb_hits) > 1:
            kb_name_path = "alias_ambiguous"
        else:
            kb_name_path = "open_world"

        # ---- deterministic correctness ----
        gold_targets = [t.strip() for t in g["kb_acceptable"].split("|") if t.strip()]
        predicted = pred.get("predicted_concept")
        if gold_targets:
            det_correct = predicted in gold_targets
        else:
            det_correct = None  # gold says NO KB concept is acceptable

        records.append(
            {
                "dataset_id": ds_id,
                "version_id": vid,
                "table": key[0],
                "column": key[1],
                "gold_concept": g["gold_concept"],
                "gold_family": g["gold_family"],
                "gold_kb": g["kb_acceptable"],
                "confidence": g["confidence"],
                "split": g["split"],
                "kb_name_path": kb_name_path,
                "deterministic_correct": det_correct,
                "predicted_concept": predicted,
                "confidence_score": pred.get("confidence_score"),
                "confidence_level": pred.get("confidence_level"),
                "decision_kind": pred.get("decision_kind"),
                "semantic_family": pred.get("semantic_family"),
                "evidence": pred.get("evidence", {}),
                "candidates": pred.get("candidates", []),
                "note": g.get("notes", ""),
            }
        )

    out_records = AUDIT_DIR / "eval_columns.json"
    out_records.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---- summary ----
    scored = [r for r in records if r["deterministic_correct"] is not None]
    correct = [r for r in scored if r["deterministic_correct"]]
    path_counter = Counter(r["kb_name_path"] for r in records)
    per_ds = defaultdict(lambda: [0, 0])
    for r in scored:
        per_ds[r["dataset_id"]][0] += 1
        per_ds[r["dataset_id"]][1] += 1 if r["deterministic_correct"] else 0

    summary = {
        "gold_rows": len(records),
        "with_prediction": matched_pred,
        "without_prediction": len(records) - matched_pred,
        "kb_name_path_counts": dict(path_counter),
        "deterministic_scored_columns": len(scored),
        "deterministic_correct": len(correct),
        "deterministic_accuracy": round(len(correct) / len(scored), 4) if scored else None,
        "per_dataset_accuracy": {
            str(ds): {"scored": n, "correct": c, "accuracy": round(c / n, 4) if n else None}
            for ds, (n, c) in sorted(per_ds.items())
        },
    }
    (AUDIT_DIR / "kb_match_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"\nWrote {out_records.name}, kb_match_summary.json")


if __name__ == "__main__":
    main()

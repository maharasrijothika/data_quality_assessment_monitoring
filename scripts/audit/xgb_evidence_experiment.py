"""XGBoost-ON-EVIDENCE EXPERIMENT (read-only; no DB writes, no code changes).

User instruction: consider XGBoost ONLY where there is no exact match and no
approved alias. Design:

  Train set : labeled columns from ALL versions except the held-out test
              version (10). Labels come from semantic_gold_labels.csv:
              positive = the specific candidate concept is one of the gold
              kb_acceptable targets for that column; negative = otherwise.
  Test set  : labeled columns of version 10 whose kb_name_path is
              "open_world" (no exact-name / approved-alias KB match).
  Features  : the engine's own evidence values for each candidate
              (embedding, name, description, datatype, profile, context,
              value) + applicability flags, margin, coverage, rank.

Two deterministic baselines on the same test columns:
  det_top1  : engine's argmax prediction == gold (only when gold allows a
              KB concept)
  det_abstain: engine's argmax, abstaining (wrong) when gold allows no KB
              concept.

Output: xgb_experiment_summary.json + console table.

Run:  .venv/Scripts/python.exe scripts/audit/xgb_evidence_experiment.py
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
AUDIT_DIR = ROOT / "artifacts" / "semantic_audit_results"

sys.path.insert(0, str(ROOT / "backend"))

TEST_VERSION_ID = 10  # dataset 10, the largest labeled version (user-confirmed)

EVIDENCE_KEYS = [
    "embedding_similarity",
    "name_similarity",
    "description_similarity",
    "datatype_compatibility",
    "profile_compatibility",
    "context_similarity",
    "value_compatibility",
]

FEATURE_NAMES = []
for k in EVIDENCE_KEYS:
    FEATURE_NAMES += [f"ev_{k}", f"app_{k}"]
FEATURE_NAMES += ["ev_margin", "ev_coverage", "cand_rank"]


def evidence_vector(evidence: dict) -> list[float]:
    vec = []
    for k in EVIDENCE_KEYS:
        entry = evidence.get(k, {})
        if isinstance(entry, dict):
            vec.append(float(entry.get("value") or 0.0))
            vec.append(1.0 if entry.get("applicable") else 0.0)
        else:
            vec.append(float(entry or 0.0))
            vec.append(0.0)
    return vec


def build_rows(records: list[dict], only_open_world: bool, test_vid: int | None):
    """(X rows, meta) per labeled column: one row per candidate concept."""
    rows = []
    for rec in records:
        if test_vid is not None and rec["version_id"] != test_vid and only_open_world:
            # open-world restriction is applied at TEST time only
            pass
        if only_open_world and test_vid is not None and rec["version_id"] == test_vid:
            if rec.get("kb_name_path") != "open_world":
                continue
        gold_targets = [t.strip() for t in (rec.get("gold_kb") or "").split("|") if t.strip()]
        if not gold_targets:
            continue  # gold allows no KB concept -> not a ranking task
        cands = rec.get("candidates") or []
        if not cands:
            continue
        margin = float(rec.get("margin") or 0.0)
        coverage = float(rec.get("evidence_coverage") or 0.0)
        for rank, cand in enumerate(cands):
            feat = evidence_vector(cand.get("evidence", {}))
            feat += [margin, coverage, float(rank)]
            label = 1 if cand.get("concept") in gold_targets else 0
            rows.append(
                {
                    "dataset_id": rec["dataset_id"],
                    "version_id": rec["version_id"],
                    "table": rec["table"],
                    "column": rec["column"],
                    "concept": cand.get("concept"),
                    "rank": rank,
                    "features": feat,
                    "label": label,
                }
            )
    return rows


def main() -> None:
    import xgboost as xgb
    from sklearn.metrics import accuracy_score, f1_score

    records = json.loads((AUDIT_DIR / "eval_columns.json").read_text(encoding="utf-8"))

    train_recs = [r for r in records if r["version_id"] != TEST_VERSION_ID and r.get("version_id")]
    test_recs = [r for r in records if r["version_id"] == TEST_VERSION_ID]

    # ---- feature sanity: check evidence keys actually present ----
    n_with_ev = sum(1 for r in records if r.get("candidates"))
    print(f"records total={len(records)} with_candidates={n_with_ev}")

    train_rows = build_rows(train_recs, only_open_world=False, test_vid=None)
    test_rows = build_rows(test_recs, only_open_world=True, test_vid=TEST_VERSION_ID)

    print(f"train rows (column,candidate): {len(train_rows)}  positives: {sum(r['label'] for r in train_rows)}")
    print(f"test rows (open-world only):   {len(test_rows)}  positives: {sum(r['label'] for r in test_rows)}")

    if not train_rows:
        print("No training rows — aborting.")
        return
    if not test_rows:
        print("No open-world test rows for version 10 — XGBoost cannot be evaluated on this split.")
        # still train to report feature importances
        X = np.array([r["features"] for r in train_rows])
        y = np.array([r["label"] for r in train_rows])
        model = xgb.XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.1, subsample=0.9,
            colsample_bytree=0.9, eval_metric="logloss", random_state=42,
        )
        model.fit(X, y)
        imp = sorted(zip(FEATURE_NAMES, model.feature_importances_), key=lambda t: -t[1])[:10]
        (AUDIT_DIR / "xgb_experiment_summary.json").write_text(
            json.dumps({"status": "no_open_world_test_rows", "feature_importance": dict(imp)}, indent=2),
            encoding="utf-8",
        )
        return

    X_train = np.array([r["features"] for r in train_rows])
    y_train = np.array([r["label"] for r in train_rows])
    X_test = np.array([r["features"] for r in test_rows])

    model = xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.1, subsample=0.9,
        colsample_bytree=0.9, eval_metric="logloss", random_state=42,
    )
    model.fit(X_train, y_train)

    # ---- score test columns ----
    probs = model.predict_proba(X_test)[:, 1]

    by_col: dict[tuple, list[dict]] = defaultdict(list)
    for row, p in zip(test_rows, probs):
        by_col[(row["table"], row["column"])].append({**row, "prob": float(p)})

    xgb_correct = 0
    det_correct = 0
    abstain_correct = 0
    per_col = []
    for (table, column), cands in sorted(by_col.items()):
        cands_sorted = sorted(cands, key=lambda r: (-r["prob"], r["rank"]))
        pick = cands_sorted[0]
        gold = next(
            r["gold_kb"] for r in test_recs if r["table"] == table and r["column"] == column
        )
        gold_targets = [t.strip() for t in gold.split("|") if t.strip()]
        xgb_hit = pick["concept"] in gold_targets
        xgb_correct += xgb_hit

        det = next(
            (r for r in test_recs if r["table"] == table and r["column"] == column), None
        )
        det_hit = bool(det and det.get("deterministic_correct"))
        det_correct += det_hit
        # det_abstain: if engine predicted nothing (concept None) on a
        # no-KB-gold column it would be correct; here gold always allows a KB
        # concept so abstain == det_top1.
        abstain_correct += det_hit

        per_col.append(
            {
                "table": table,
                "column": column,
                "gold_kb": gold,
                "engine_pick": det.get("predicted_concept") if det else None,
                "engine_level": det.get("confidence_level") if det else None,
                "xgb_pick": pick["concept"],
                "xgb_prob": round(pick["prob"], 4),
                "xgb_correct": xgb_hit,
                "deterministic_correct": det_hit,
            }
        )

    n = len(by_col)
    results = {
        "test_version_id": TEST_VERSION_ID,
        "train_versions": sorted({r["version_id"] for r in train_rows}),
        "train_rows": len(train_rows),
        "train_positives": int(y_train.sum()),
        "test_columns_open_world": n,
        "xgb_top1_accuracy": round(xgb_correct / n, 4) if n else None,
        "deterministic_top1_accuracy": round(det_correct / n, 4) if n else None,
        "per_column": per_col,
        "feature_importance": dict(
            sorted(zip(FEATURE_NAMES, [round(float(v), 4) for v in model.feature_importances_]), key=lambda t: -t[1])[:12]
        ),
    }
    (AUDIT_DIR / "xgb_experiment_summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"\nTest columns (open-world, v10): {n}")
    print(f"  XGBoost top-1 accuracy:   {results['xgb_top1_accuracy']}")
    print(f"  Deterministic top-1 acc:  {results['deterministic_top1_accuracy']}")
    print("\nPer-column detail:")
    for c in per_col:
        print(
            f"  {c['table']}.{c['column']:<22} gold={c['gold_kb']:<14} "
            f"engine={str(c['engine_pick']):<16} xgb={c['xgb_pick']:<16} "
            f"xgb_ok={c['xgb_correct']} det_ok={c['deterministic_correct']}"
        )


if __name__ == "__main__":
    main()

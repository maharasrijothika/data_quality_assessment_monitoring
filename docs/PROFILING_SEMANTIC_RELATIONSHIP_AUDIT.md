# PROFILING → SEMANTIC → RELATIONSHIP EVIDENCE AUDIT

Audited: 2026-10-04, commit `d23561d` (main). Method: direct code inspection plus live
database queries (`backend/data/dq_assessment.db`) and exported artifacts under `artifacts/`.
Every claim reflects what the code **currently does**, not what documentation says it should do.

---

## 1. Executive summary

The pipeline is a three-stage deterministic evidence chain:

1. **Profiling (Stage 03)** — `backend/app/services/profiling.py` computes the full
   per-column statistical profile. Result is persisted in `stored_profiles.profile_json`
   per dataset version and (new) exported to `artifacts/profiling/<dataset_id>/<version_id>/profile.json`.
2. **Semantic understanding (Stage 04)** — `semantic_dataset_analysis.py` orchestrates
   `semantic_analysis.py` (two-level decisions: exact stage → retrieval + evidence scoring →
   family fallback → Open Discovery). Scoring is a documented weighted mean of applicable
   evidence (`app.config.SEMANTIC_SCORING_WEIGHTS`). Embeddings use `all-MiniLM-L6-v2`
   (representation version 2). Results persisted in `semantic_predictions` and — new —
   append-only run records (`semantic_runs`) with the exact semantic input snapshot.
3. **Relationship discovery (Stage 05)** — `relationship_discovery.py` produces ranked
   PK/FK candidates with per-feature evidence (containment, orphans, uniqueness, semantic
   compatibility). Explicitly *not* RI validation.

**Biggest verified problems found** (all fixed this audit unless marked P1/P2):

- Description evidence: `description_similarity` **IS calculated** in
  `semantic_features.generate_features` and passed through the API. The UI's
  "(no evidence available)" impression comes from (a) decided (approved/edited) predictions
  whose stored evidence predates the column descriptions — decided rows keep their stored
  evidence forever by design — and (b) the raw wording. Root cause verified empirically:
  dataset 22 has 18 column descriptions in metadata; all 18 predictions are approved/edited
  from a run made **before** the descriptions existed, so the stored flag stays
  `applicable=False`. The engine never fabricated anything; the UI now says
  "(no column description was supplied to the engine)" — truthful for that stored run.
- Table-name join bug: re-uploaded versions get profile table names with a dedup suffix
  (`dq_core_1`) while `TableMetadata` keeps the original (`dq_core`). The join failed
  silently, so column descriptions and table context **never reached the engine** for
  re-uploads (verified: DS 25 v27/v28/v29, DS 26 v31). Fixed with a suffix-stripping fallback.
- Semantic UI displayed stage-foreign evidence rows: `relationship_evidence` and
  `value_evidence` were rendered with "(no evidence available)" although they are narrower /
  later-stage signals. Fixed: hidden unless actually applicable; relationship evidence is
  only shown when approved Stage-05 relationships genuinely fed hints into scoring
  (the architecture DOES consume approved-relationship hints at semantic time, weight 0.03).
- No semantic run versioning existed: re-runs deleted pending predictions in place.
  Fixed with append-only `semantic_runs` records + S(n-1) vs S(n) comparison API/UI.
- No machine-readable audit artifacts existed. Fixed: `artifacts/` exports for profiling,
  semantic input (incl. exact embedding text), relationships, and the complete KB.

---

## 2. Profiling implementation

- **Inputs**: raw files per dataset version (`storage.RAW_DATA_DIR/<dataset>/v<n>/<file>`),
  read via `ingestion.read_table` (shared dtypes across stages). Excel workbooks profiled per sheet.
- **Calculations** (per column, in `profile_dataframe`): dtype, null count/percentage,
  empty/whitespace counts, distinct count/percentage, cardinality, text statistics
  (length mean/max, shape patterns with distinct-weighted regex counting), numeric statistics
  (min/max/mean/std, quantiles, IQR, MAD, outlier evidence, integer-valued flag, code-like,
  constant/near-constant, integer-sequence counter-like), datetime evidence (parse rate,
  format-valid percentage, has-time, 500-distinct pre-check), identifier evidence
  (`identifier_like`, `identifier_signal`, `identifier_name_signal`, `identifier_repeats`),
  near-constant, top values, and cross-column: functional dependencies, composite uniqueness
  candidates (with minimality), row completeness, co-missing pairs.
- **Outputs**: dict with `dataset_id`, `version_id`, `version_number`, `parent_version_id`,
  `table_count`, `total_rows`, `total_columns`, `tables[]` each `{table_name, source_file,
  sheet_name, row_count, column_count, columns[] (full per-column profile),
  composite_uniqueness_candidates[], functional_dependency_candidates[] (reported pairs),
  row_completeness, key_findings}`.
- **Persistence**: `routers/profiling.py::get_dataset_profiling` → `persist_profile`
  (upsert in `stored_profiles`). `semantic_stage.get_or_create_profile` reuses it; a second
  call for the same version reuses the stored JSON (no recalculation).
- **Downstream consumption**: semantic (profile compatibility checks, role veto, datatype
  resolution, value evidence) and relationship discovery (PK evidence, containment inputs,
  float-measure/free-text/counter-like gates) read the **stored** profile.

## 3. Profiling output inventory → downstream

Key fields and their fate (A=passed directly, B=transformed, C=profiling/UI only, D=not consumed):

| Profiling field | Passed to semantic | Transformation | Used in scoring |
| --- | --- | --- | --- |
| `data_type` | YES (A) | `normalize_data_type` (+ datetime promotion via `format_valid_percentage>=95`) | YES (datatype weight 0.08; exact-stage contradiction check) |
| `null_count` / `null_percentage` | YES (A) | non-null completeness checks | YES (profile checks: non_null, identifier_like, bounded) |
| `empty_string_count` / `whitespace_only_count` | NO | — | NO — profiling/UI only (C) |
| `distinct_count` / `distinct_percentage` | YES (A) | threshold comparisons | YES (categorical, uniqueness hint, identifier fallback) |
| `duplicate info` / row-level duplicates | NO | — | C (UI) |
| text stats (`mean_length`, `max_length`, `shape`) | YES (A) | long_text / free-text checks | YES (long_text check; relationship free-text gate) |
| pattern evidence (`email_like`, `phone_like`, `postal_like`, `currency_like`) | YES (A) | coverage of NON-NULL rows vs `SEMANTIC_PATTERN_MIN_COVERAGE=0.6` | YES (pattern expectations) |
| numeric stats (`min`, `max`, quantiles, IQR, MAD, outliers) | PARTIAL (B) | only `min`/`max` read (non_negative, bounded≤150, year, domain_range, time_of_day); quantiles/IQR/MAD NOT consumed | min/max YES; quantiles/IQR/MAD **calculated but not currently consumed by semantic** |
| `integer_valued`, `code_like`, `counter_like` | YES (A) | observed role inference | YES (role veto; relationship gates) |
| datetime evidence | YES (A) | `has_time_component`, parse rate | YES (role; temporal checks; `_actual_data_type`) |
| identifier evidence (`identifier_like/signal/name_signal`) | YES (A) | boolean gates | YES (identifier_like check; observed role) |
| candidate PK evidence | YES (A) | `identifier_*` + uniqueness | YES (identifier checks; relationship `_parent_pk_evidence`) |
| composite uniqueness candidates | NO to semantic | — | C (relationship key candidates + UI) |
| functional dependency candidates | NO | — | C (profiling/UI; **not consumed downstream**) |
| top values (`categorical.top_values`) | YES (A) | value vocabulary matching | YES (value_evidence 0.07; boolean_like vocabulary) |
| row completeness / co-missing | NO | — | C (UI key findings) |

## 4. Semantic input contract

Per column, `semantic_dataset_analysis.analyze_dataset` supplies:
`column_name`, `column_description` (from `ColumnMetadata.description` — **after this audit's
join fix**), `data_type` (raw profile dtype), full `profile` dict, `dataset_domain`
(`Dataset.domain`), `table_context` (`TableMetadata.description`), `sibling_names`
(all profiled column names of the same table), and (only if approved relationships exist)
`relationship_hints` from Stage 05. The dataset description, `source_system`, `update_cadence`
are recorded in the run snapshot but are **not embedded and not scored** (D).

**Calculated but not currently consumed by semantic understanding**: text quantiles, IQR, MAD,
outlier evidence, empty/whitespace counts, FD candidates, row completeness, duplicate row info.

## 5. Incoming column embedding construction

`SemanticRetrievalService.build_column_text` (v2 template, representation version 2):

```
<abbreviation-expanded ordered column name>. [<column description>]. [table: <table description>; columns: <up to 12 expanded sibling names>]
```

- The **dataset domain is deliberately NOT embedded** (it pulls every column toward the same
  concepts); it is used as context evidence instead.
- Internal normalization (`normalize_name`, `expand_tokens`, abbreviation lexicon) is a
  technical detail — kept in the audit artifact and the collapsed "Name normalization details"
  UI block, never as the primary display.
- New: the exact text per column is persisted in `evidence_json.semantic_input.embedding_text`,
  shown in the UI under "Semantic input (audit: exact embedding text)", and exported in
  `artifacts/semantic/<dataset_id>/<version_id>/semantic_input.json`.
- Embedding model: `all-MiniLM-L6-v2`, normalized vectors, one batched `encode_queries` call
  per dataset run. Path: **column text → SentenceTransformer → query embedding**.

## 6. KB concept embedding construction

`SemanticEmbeddingService.build_concept_text`:

```
Concept: <concept_name>
Category: <category>
Description: <description>
Also known as: <alias1, alias2, ...>
```

JSON config (expected types, profile expectations) is deliberately NOT embedded.
Path: **concept text → SentenceTransformer → stored embedding** (`semantic_concept_embeddings`,
keyed by `<model>/<representation-version>`). Retrieval: normalized dot product (cosine) →
hybrid union with lexical token-overlap candidates → evidence scoring of ALL concepts
(no top-k truncation before scoring). KB fingerprint (row counts + max ids + max created_at)
is persisted with every prediction/run for exact-state traceability.

## 7. Candidate retrieval

`retrieve_candidates`: embedding top-K over the cached matrix (top_k = all concepts by
default), unioned with lexical candidates (expanded-token overlap / exact key), sorted by
(similarity, lexical). Exact/alias stage runs first: a full name/alias key match with a
non-contradicting profile and a clear margin (≥0.05) is decided WITHOUT retrieval.

## 8. Evidence calculation

`generate_features` produces `{value, applicable}` for: `name_similarity` (entity/head-aware,
with `entity_conflict`/`head_conflict` gates), `embedding_similarity`, `description_similarity`
(lexical `text_similarity` of the column description vs concept description/name/aliases —
**a real description-specific score, not the combined embedding**), `datatype_compatibility`
(logical impossibilities count as applicable negative evidence), `profile_compatibility`
(checks + observed-role veto), `value_evidence` (concept vocabulary/regex vs observed values),
`context_similarity` (sibling/domain keywords vs concept keywords), `relationship_evidence`
(approved parent concepts, only when approved relationships exist). The score is the weighted
mean over **applicable** features; structurally absent evidence (no description configured)
is excluded rather than counted as failure.

## 9. Ranking / reranking

Exact stage → retrieval stage; each candidate is capped by structural gates
(`_apply_caps`: entity_conflict → below ambiguous; profile_veto/head_conflict → below
probable). Two-pass decision: best non-family-only specific concept, else family fallback
(`FAMILY_MATCH`, capped at Ambiguous), else UNKNOWN with a name-derived proposal.

## 10. Confidence calculation

`semantic_stage.confidence_level`: Strong ≥0.85 / Probable ≥0.80 / Ambiguous ≥0.40 / Unknown,
each additionally requiring evidence coverage ≥0.50 and top-2 margin (≥0.12 Strong, ≥0.05
Probable). `FAMILY_MATCH` caps at Ambiguous. The score is explicitly documented as a weighted
evidence score, **not** a calibrated probability.

## 11. Open Discovery

Exists as a real fallback (not a rename): `decision_kind ∈ {KB_MATCH, FAMILY_MATCH, UNKNOWN}`.
`FAMILY_MATCH`/`UNKNOWN` produce a `proposal` `{name (from expanded tokens, no guessed role
word), family, source, ask_user}`. The proposal is NOT inserted into the KB; approving it
stores a **user-defined type** on the prediction (`user_defined_type`, `decision_source=
"user_defined"`), recorded as feedback. `ask_user=True` (opaque names) shows no fabricated
name. The UI shows the readable form ("Customer Type") via `readableConceptName`; the stored
value stays the internal token form (audit artifact keeps both). Approve (KB), Edit
(free text), Reject are all supported; the candidate is distinguishable from KB concepts
(`decision_kind`, `source`, family badges). Gap (P2): no accumulated candidate-evidence
admission workflow yet (`KnowledgeBaseVersion` model exists; single edits never touch the KB — correct).

## 12. Semantic UI problems (and fixes)

- Fixed: `relationship_evidence` / `value_evidence` rows rendered raw with "(no evidence
  available)" — now labeled and hidden unless applicable.
- Fixed: `description_similarity` not-applicable wording now states the truth ("no column
  description was supplied to the engine").
- Fixed: full ranked candidate block demoted into a collapsed audit `<details>`; primary UI
  shows Prediction / Confidence / Why (evidence list) / clickable Alternatives.
- Fixed: open-discovery proposals displayed with underscores ("customer_type") — now
  title-cased readable names; "Accept proposal" stores the readable form.
- Fixed: exact embedding text + description + domain + table context + model version visible
  in a collapsed "Semantic input (audit)" block.

## 13. Semantic run / versioning

Was: none — re-runs replaced pending predictions in place (approved/edited kept).
Now: every run appends a `semantic_runs` row (`run_number`, `model_version`,
`embedding_representation_version`, `evidence_version`, `kb_fingerprint`, full
`configuration_json` with weights/thresholds/margins, `semantic_input_json`, `predictions_json`,
timestamp). Nothing is overwritten; approved decisions survive as before and are recorded in
the new run with their decision source. APIs: `GET /datasets/{id}/semantic/runs` (history)
and `GET /datasets/{id}/semantic/runs/compare` (S(n-1) vs S(n) per-column diff with changed
flags). UI: collapsible "Compare runs S1 → S2" table. Note (P2): comparison of arbitrary
runs (not just the last two) and KB-version bump as an explicit run trigger remain open.

## 14. Dataset-wide prediction results

Exported per run in `artifacts/semantic/<dataset_id>/<version_id>/semantic_input.json`:
dataset/version/domain/source system/cadence/description, per column the full semantic input
(name, dtype, description, domain, table context, siblings, exact embedding text, model,
representation version) plus prediction, decision, confidence, evidence and top-5 alternatives.
No ground truth is invented; human decisions carry `decision_source`.

Additionally, `artifacts/semantic/<dataset_id>/<version_id>/predictions.json` exports the
**dataset-wide column listing for ALL 16 dataset versions that hold stored predictions**
(column, dtype, description, domain, table context, semantic input text where a snapshot
exists, predicted concept, decision, confidence, full evidence, top-5 alternatives).
Columns without a verified label carry `evaluation: "UNVERIFIED"`; decided ones carry
`evaluation: "USER_APPROVED/EDITED/REJECTED"`.

## 15. Semantic failure analysis

Verified failure modes: (1) decided-before-metadata predictions freeze `applicable=False`
description evidence (root cause of the reported UI complaint — see §1; a re-run creates
fresh evidence for pending columns only); (2) table-name dedup suffixes broke the metadata
join on re-uploads (fixed); (3) opaque names → `ask_user` with no suggestion (correct);
(4) gibberish names are capped at 0.2 name evidence; (5) dense-integer containment in
relationships is dampened (see §19).

## 16. KB inventory

Complete export: [docs/semantic_kb_inventory.md](semantic_kb_inventory.md) and
`artifacts/semantic_kb.json` — every concept with id, name, category, description, aliases,
expected types, full profile expectations (family, role, rule_family, head/entity tokens),
embedding model/version, and the **exact embedding text template**. Concept count and details:
see the generated inventory (seeded from `services/semantic_kb.py`; families + specific concepts).

## 17. KB expansion recommendations

- Missing concepts observed in registered datasets: postal code, order status vocabulary,
  product name, flight number, loyalty tier (DS 28) — several resolve only via family fallback.
- Overlapping concepts: "Identifier" family vs specific `*_id` concepts overlap by design
  (two-pass ranking) but "Email" doubles as family — documented dual-role behavior.
- Weak/missing aliases: `cust_type` has no direct alias (resolved only via expansion);
  `order_id`/`orderid` variants depend on the abbreviation lexicon.
- No duplicate concepts found in the seed (idempotent seeding by name).
- Missing semantic families: URL/JSON/free-text family; temporal-duration family.

## 18. Range-rule recommendation audit

`test_stage06_rules_workflow.py` (numeric-range candidate tests) confirm the recommender
distinguishes observation from constraint: observed min/max produce **candidate** rules for
human approval, never authoritative business constraints; approval flow exists
(`RuleApproval`). Distinction A (observed) / B (candidate) / C (approved) is enforced in the
lifecycle. Remaining gap (P1): the recommendation reason text does not yet literally say
"Observed range only — insufficient evidence for a business constraint" when observed range
is the sole evidence; wording hardening recommended.

## 19. Relationship discovery implementation

`discover_relationships` (persisted candidates, kinds `pk` / `composite_pk` / `relationship`):
PK candidates from profile evidence (`_parent_pk_evidence`: 0.55·uniqueness + 0.25·completeness
+ 0.20·identifier_name_signal, small-sample dampened, ≥0.55 threshold); composite candidates
from profiling with ≥98% uniqueness and no float-measure/free-text members. Pair evaluation
(`_evaluate_pair`) evidence: name_similarity, datatype_compatibility, semantic_compatibility
(final Stage-04 types; neutral when either side unmapped), parent_uniqueness, value_containment,
value_overlap (dense-integer dampening), cardinality_compatibility, child_completeness,
structural_evidence. Orphan statistics (`orphan_distinct_count`, `orphan_distinct_rate`)
are recorded as STATS — **low containment does not eliminate a candidate**; it reports
"potential relationship with likely orphans". Discovery and RI validation remain separate
(approved candidates may later become RI rules; nothing is auto-created).
Gates: name prefilter unless strong PK, min score 0.35, zero-overlap+weak-name exclusion,
top-4 per child column.

## 20. Relationship evidence

Full evidence + stats persisted per candidate in `relationship_candidates.evidence_json` /
`stats_json` (containment, overlap, orphan counts/rates, parent PK stats, semantic type
comparison, dense-integer warnings) and now exported to
`artifacts/relationships/<dataset_id>/<version_id>/relationships.json` (backfilled for all
existing datasets). Human decisions (approve/reject/edit/missed/manual) are kept across
re-runs and recorded in `RelationshipFeedback`.

## 21. Relationship UI

`stages/relationships.tsx` renders candidates as `child → parent` with score, status badges,
per-evidence values and Approve/Reject/Edit actions; manual-key and manual-relationship
addition supported. Audit detail remains in the artifact; no internal model features are shown
as primary UI. Identifier/composite candidates appear as key-candidate rows (kind badges),
not separate fake "cards".

## 22. Architecture gaps

- Quantiles/IQR/MAD/outlier evidence calculated but unconsumed downstream (removal or use — decide).
- Dataset description / source system / cadence recorded but not used as evidence.
- No RI-validation stage consuming approved relationships yet (discovery only).
- KB admission workflow (accumulated evidence → review → version bump) designed
  (`KnowledgeBaseVersion`) but not yet active.
- Legacy decided predictions keep pre-metadata evidence until a human re-decides (by design,
  but a "refresh evidence keeping decision" action would help — P2).

## 23. Recommended P0/P1/P2 fixes

**P0 (done this audit)**: truthful description-evidence wording; table-name join fix;
semantic input + embedding-text snapshots; append-only run records + comparison; artifact
exports (profiling / semantic / relationships / KB); UI evidence cleanup + readable
open-discovery names + clickable alternatives.

**P1**: range-rule reason wording ("Observed range only — insufficient evidence for a
business constraint"); RI-validation stage consuming approved relationships; explicit
run trigger on KB/representation change.

**P2**: KB admission workflow activation; arbitrary-run comparison; evidence refresh action
for decided predictions; consume or drop unconsumed profiling statistics.

---

## Verification

- Backend tests: 303 passed / 2 failed / 1 skipped — the 2 failures
  (`test_end_to_end.py::test_full_pipeline_end_to_end`,
  `test_stage_services.py::test_confidence_requires_evidence_coverage`) are pre-existing
  and identical to the pre-audit baseline.
- Frontend: `npm run build` exit 0.
- Artifacts verified on disk for 16 stored profile versions, 4 relationship datasets,
  and the complete KB (see `artifacts/`).

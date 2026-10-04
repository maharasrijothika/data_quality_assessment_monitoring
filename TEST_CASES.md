# TEST_CASES.md — Manual QA Test Plan

Manually execute these against the real application. Leave `Actual Result`,
`Pass/Fail`, and `Notes` blank while testing; fill them in as you go.

**How to read each case:** `Preconditions` assume a running backend
(`http://127.0.0.1:8000/health` returns healthy) and frontend
(`http://localhost:5173`) unless stated otherwise. UI steps reference the
stage tabs `01 Ingestion … 11 Feedback` exactly as shown in the app.

**Before you start:** this table records what is actually implemented. Features
not implemented are marked `NOT IMPLEMENTED` in the relevant stage sections and
their tests are omitted rather than faked.

| # | Feature | Status |
|---|---|---|
| 1 | Ingestion (CSV/XLS/XLSX/Parquet, folder, validation, dedupe, analyze-before-register) | IMPLEMENTED |
| 2 | Dataset context (description/domain/source/cadence/table+column descriptions, persistence) | IMPLEMENTED |
| 3 | Versioning & fingerprinting (file/schema/content fingerprints, lineage, immutability) | IMPLEMENTED |
| 4 | Profiling (general/completeness/uniqueness/numeric/text/datetime + edge cases) | IMPLEMENTED |
| 5 | Semantic understanding (embedding+feature scoring, evidence, confidence levels, human decisions) | IMPLEMENTED |
| 6 | Metric & rule recommendation (from approved semantics + profile evidence only) | IMPLEMENTED |
| 7 | Rule validation (VALID / NEEDS_REVIEW / INVALID) | IMPLEMENTED |
| 8 | Human approval & safe rule execution (constrained DSL: completeness/uniqueness/validity) | IMPLEMENTED |
| 9 | DQ scoring (per-metric, overall mean, N/A + critical-failure surfacing) | IMPLEMENTED |
| 10 | RCA (evidence-based concentration, correlation-only language) | IMPLEMENTED |
| 11 | Remediation (whitespace normalization only, approval, new immutable version, auto-reassessment) | IMPLEMENTED |
| 12 | Monitoring (KS/PSI/schema drift vs explicit baseline, small-sample honesty) | IMPLEMENTED |
| 13 | Feedback store + offline training (threshold-gated) + model versioning/promotion | IMPLEMENTED |
| 14 | Consistency rules (cross-column) | NOT IMPLEMENTED — no rule type exists; see Stage 06 notes |
| 15 | Referential integrity rules (multi-table FK checks) | NOT IMPLEMENTED — see Stage 06 notes |
| 16 | Weighted scoring endpoint/config | PARTIALLY IMPLEMENTED — service supports `weighted`, no API/UI exposure |
| 17 | Anomaly detection (Isolation Forest etc.) | NOT IMPLEMENTED (deliberately out of scope) |
| 18 | Remediation beyond whitespace normalization | NOT IMPLEMENTED by design (all other corrections are source-system or human edits) |
| 19 | Allowed-value / range rules | NOT IMPLEMENTED (no config UI; spec forbids inferring from data) |
| 20 | Rule editing in the UI | PARTIALLY IMPLEMENTED — API supports `edited_rule_json`, UI exposes Approve/Reject only |

---

## TEST DATA

There is **no committed sample dataset directory** in the repo (only runtime
uploads under `data/raw/datasets/` from previous runs). Create the following
test files locally (e.g. `C:\dq-test-data\`).

**TD-01 `dq_core.csv` — main workhorse (profiling, semantics, rules, execution, scoring, RCA, remediation)**
Exactly 12 data rows; designed so profiler stats and rule failures are predictable.

```csv
customer_id,cust_name,email,age,city,signup_date,revenue
C001,Alice Cooper,alice@example.com,25,Pune,2024-01-15,1200.50
C002,Bob Jones ,bob@example.com,34,Delhi,2024-02-20,950
C003,Carla Gomez,carla@gnail.com,29,Mumbai,2024-03-05,780.25
C004,Dave Miller,   ,41,Chennai,2024-04-10,0
C005,Eve Nair,,31,  Mumbai,2024-05-12,2330.75
C006,   ,fred@example.com,,Pune,2024-06-01,460
C007,Grace Lee,grace@example.com,55,Kolkata,2024-07-23,1120
C008,Hank Pym,hank@example.com,,Delhi,2024-08-30,875.10
C009,Ivy Chen,ivy@example.com,27,Pune,2024-09-14,640.80
C010,Omar Farouk,omar@example.com,38,Mumbai,2024-10-05,1499.99
C011,Parvati Rao,parvati@example.com,45,Pune,2024-11-11,2040.60
C012,Quinn Bell,quinn@example.com,52,Delhi,2024-12-25,0
```

Built-in evidence: `customer_id` = 12 distinct → identifier; `email` has one
empty + one whitespace-only + one malformed (`carla@gnail.com`); `age` has
2 nulls (25,34,29,41,31,55,27,38,45,52 present); `city` has one
whitespace-padded value (`␣␣Mumbai`) — remediation preview material; `revenue`
has outliers 0/0 and 2330.75 vs median ≈ 975; `signup_date` is monotonic.

**TD-02 `semantics_tricky.csv` — semantic edge cases (Stages 05, 06)**

```csv
CUST_REF,cust_nm,cust_type,phone_number,order_dt,amount,currency,xyz_qq,memo
1001,Acme Corp,Retail,9876543210,2024-01-05,150.00,INR,a11,note one
1002,Globex,Corporate,9123456780,2024-02-11,240.50,INR,b22,note two
1003,Initech,Retail,9812345678,2024-03-22,90.25,USD,c33,note three
1004,Umbrella,Corporate,9900112233,2024-04-02,680.00,INR,d44,note four
1005,Wayne Enterprises,Retail,9871234567,2024-05-19,410.75,USD,e55,note five
```

Built-in traps: `cust_type` must **not** map to a name concept; `CUST_REF`
(mixed case + abbreviation) should map to an identifier concept; `xyz_qq` is
gibberish and must resolve to UNKNOWN / insufficient evidence; `cust_nm`
(abbreviated) vs `cust_type` tests lexical-vs-semantic disambiguation.

**TD-03 `dates_bad.csv` — datetime profiling + date parsing evidence**

```csv
event_id,event_date,not_a_date
E1,2024-01-01,hello
E2,2024-02-01,world
E3,15/03/2024,again
E4,2024-04-31,bad-day
E5,2024-05-05,
```

`event_date` mixes ISO and DD/MM/YYYY (mixed formats); `not_a_date` contains
non-dates (parseability evidence ≈ 0).

**TD-04 `dq_core_v2.csv` — changed content for versioning/drift (same schema, different values)**
Copy TD-01, then change: row 1 revenue `1200.50` → `1350.00`, row 5 email
`fred@example.com` → `fred.smith@example.com`, and fix `␣␣Mumbai` → `Mumbai`.
Same 12 rows, same columns, different content → new version, no schema change.

**TD-05 `dq_core_schema_change.csv` — schema change (add one column to TD-01)**
Copy TD-01 and add a `loyalty_tier` column with values `Silver` in every row.
Schema fingerprint must change.

**TD-06 `empty.csv` / TD-07 `corrupted.xlsx`**
- TD-06: create the file but leave it **0 bytes** (right-click → New → Text Document, rename to `empty.csv`, keep size 0 KB).
- TD-07: take any `.xlsx`, open it in a text editor, delete most bytes, save → invalid zip. Name it `corrupted.xlsx`.

**TD-08 `book.xlsx` — multi-sheet workbook**
Create in Excel with **3 sheets** named `customers`, `orders`, `reviews`:

Sheet `customers` (header + 4 rows):
```csv
customer_id,customer_name,email
K001,Nova Ltd,nova@example.com
K002,Orion Inc,orion@example.com
K003,Zephyr,zephyr@example.com
K004,Polaris,polaris@example.com
```
Sheet `orders` (header + 4 rows):
```csv
order_id,customer_id,order_total
T-001,K001,250.00
T-002,K001,180.25
T-003,K002,99.99
T-004,K003,300.00
```
Sheet `reviews` (header + 3 rows):
```csv
review_id,order_id,rating
R-1,T-001,5
R-2,T-002,4
R-3,T-003,3
```
Use: workbook → multiple logical tables; boundaries preserved; one file, three
tables in profiling; also proves per-sheet profiles.

**TD-09 folder `dq_folder\` — multi-file folder upload (3 related tables)**
Create a folder with three CSVs (same content as TD-08's three sheets, saved as
`customers.csv`, `orders.csv`, `reviews.csv`). Use: folder registration;
table-per-file discovery; multiple tables in one dataset.

**TD-10 `unsafe_#name?.csv` — unsafe filename**
Valid CSV content (a copy of TD-06 replaced with 2 rows) but filename containing
characters like `#` and `?`. Use: filename sanitization (`Path(filename).name`
strips directories; characters are preserved or normalized by the browser —
verify the system accepts or clearly rejects it and never stores a path).

**TD-11 `dup_copy.csv` — exact duplicate**
Byte-identical copy of TD-01. Use: duplicate detection (SHA-256 equality).

**TD-12 `large.csv` — volume profiling**
Generate ~20,000 rows (e.g. PowerShell loop or Excel fill) with columns
`id (1..20000 unique), value (random 0–1000), city (5 repeating values)`.
Use: profiling on larger data; UI stays responsive.

**TD-13 `latin1_special.csv` — non-UTF8 encoding**
Save with **ANSI/Windows-1252** encoding (Notepad: Save As → Encoding ANSI):
```csv
product_id,product_name
P1,Café Latte
P2,Strömbäck
P3,Ünïcødé test
```
Use: cp1252 fallback path in `read_table`.

**TD-14 `numeric_edge.csv` — numeric profiling edge cases**

```csv
const_col,mixed_col,outlier_col
5,12,100
5,abc,102
5,13,101
5,,99
5,14,5000
5,15,98
```
`const_col` near-constant; `mixed_col` mixed text/number with null;
`outlier_col` has IQR outliers (5000) for outlier evidence.

> Known limitation to keep in mind while testing: **relationships between
> tables are NOT analyzed** (no FK/candidate-relationship detection — see
> Stage 06 note), so TD-08/TD-09 verify table-boundary preservation and
> per-table profiling, not cross-table rules.

---

## STAGE 01 — DATA INGESTION

**ING-01 · Valid CSV registration**
- Purpose: register a single CSV as dataset V1.
- Preconditions: backend + frontend running; no dataset named "dq-core" yet.
- Input: `TD-01 dq_core.csv`; dataset name `dq-core`.
- Steps: Stage 01 → choose name → Upload file → select TD-01 → Register dataset.
- Expected: success notice with Dataset ID, `V1`, Version ID, Files=1; app auto-advances to Stage 02; `data/raw/datasets/{id}/v1/dq_core.csv` exists on disk; `GET /datasets` lists it with `version_number 1`.

**ING-02 · Valid XLSX**
- Purpose: Excel workbook ingestion.
- Input: `TD-08 book.xlsx`.
- Steps: register as new dataset `book`.
- Expected: success; later profiling (Stage 04) shows **3 tables** (`book` registers one version; profiling renders `customers`, `orders`, `reviews` as separate tables with their own columns).

**ING-03 · Valid XLS (legacy)**
- Purpose: legacy Excel support.
- Input: save TD-08 sheet 1 as `book.xls` (Excel 97-2003).
- Steps: register as `book-legacy`.
- Expected: accepted (`xlrd` engine); profiling shows 1 table.

**ING-04 · Valid Parquet**
- Purpose: Parquet ingestion.
- Input: generate `parquet_sample.parquet` via Python:
  `pandas.DataFrame({"id":[1,2,3],"amount":[10.5,20.0,30.25]}).to_parquet("parquet_sample.parquet")`
- Steps: register as `parquet-sample`.
- Expected: accepted; profiling shows 1 table with columns `id`, `amount`.

**ING-05 · Empty file**
- Purpose: reject 0-byte upload.
- Input: `TD-06 empty.csv`.
- Expected: red notice `Uploaded file is empty.`; no dataset created.

**ING-06 · Unsupported file**
- Purpose: reject non-data formats.
- Input: any `.txt` file.
- Expected: `Unsupported file format: .txt. Supported formats: .csv, .parquet, .xls, .xlsx`; nothing registered.

**ING-07 · Corrupted file**
- Purpose: reject unreadable content.
- Input: `TD-07 corrupted.xlsx`.
- Expected: `Could not read corrupted.xlsx: ...`; nothing registered.

**ING-08 · Oversized file**
- Purpose: enforce the 100 MB limit.
- Input: any supported file padded past 100 MB (e.g. concatenate TD-12 copies), or simulate by temporarily raising size check.
- Expected: `File exceeds the maximum allowed size of 100 MB.`; nothing registered.

**ING-09 · Duplicate upload (single)**
- Purpose: byte-identical detection across datasets.
- Preconditions: ING-01 done.
- Input: `TD-11 dup_copy.csv` (copy of TD-01).
- Steps: select it, click Register dataset.
- Expected: analysis returns "All uploaded files already exist…" (danger notice) listing the original file + version; registration blocked.

**ING-10 · Mixed new + duplicate upload**
- Purpose: analyze-then-register flow keeps new files, skips duplicates.
- Preconditions: ING-01 done.
- Input: select **TD-11 + TD-02** together.
- Expected: warning notice "Upload requires review" with `Duplicate files (1)` (named `dup_copy.csv` + original reference) and `New files (1)` (`semantics_tricky.csv`); button reads "Register 1 new file"; clicking registers only TD-02 as a dataset.

**ING-11 · Folder upload**
- Purpose: folder registration with multiple related tables.
- Input: `TD-09 dq_folder\` via **Upload folder**.
- Expected: all 3 CSVs selected (relative paths shown); registration succeeds with Files=3; profiling later shows 3 tables.

**ING-12 · Multiple related tables in one workbook**
- Purpose: sheet = logical table; boundaries preserved.
- Input: TD-08 (see ING-02).
- Expected: after profiling, `customers`/`orders`/`reviews` have **separate** column sets; no column mixing between sheets.

**ING-13 · Unsafe filename/path**
- Purpose: path traversal + directory stripping.
- Input: `TD-10 unsafe_#name?.csv`; also try renaming a file to `..\evil.csv` (not selectable in browsers — if you cannot select it, mark as covered-by-design: `validate_filename` rejects absolute paths and `..`).
- Expected: either accepted with the stored file flattened to a safe basename inside `v1/`, or a clear rejection; **never** a write outside `data/raw/datasets/{id}/v{n}/`.

**ING-14 · File with missing values**
- Purpose: ingestion tolerates nulls (profiling reports them later).
- Input: TD-01 (has nulls in `email`, `age`, `cust_name`).
- Expected: registration succeeds; profiling shows `null_count > 0` for those columns.

**ING-15 · Unusual column names**
- Purpose: spaces/case/symbols in headers.
- Input: create `weird_cols.csv`:
  ```csv
  Order ID,Total Amount,2nd_col,UPPER FLAG
  1,10.5,x,YES
  2,11.0,y,NO
  ```
- Expected: registered and profiled with exact names (`Order ID`, `2nd_col`, …); semantic stage runs without crashing (may be Unknown — fine).

**ING-16 · Non-UTF8 encoding**
- Purpose: cp1252 fallback.
- Input: TD-13.
- Expected: registered; profiling shows `product_name` with the accented strings readable (`Café Latte`, `Strömbäck`).

---

## STAGE 02 — DATASET CONTEXT

All cases use dataset `dq-core` (ING-01). Context is optional — the app must
never force it.

**CTX-01 · Save full context** — enter Description "Retail customer master with contact info", Domain "Retail", Source system "Salesforce", Update cadence "Monthly", a table description, and a description for `customer_id` ("Primary customer identifier"). Click **Save and continue**. Expected: success notice; app moves to Stage 03; values persist after reload (see CTX-11).

**CTX-02 · Domain only** — clear other fields, set Domain "Finance", save. Expected: success; Stage 03 shows.

**CTX-03 · Source system only** — set Source system "Oracle", save. Expected: success.

**CTX-04 · Update cadence only** — set cadence "Daily", save. Expected: success.

**CTX-05 · Table description only** — set table description, save. Expected: success.

**CTX-06 · Column description only** — set a description on `email`, save. Expected: success. (Column descriptions feed the semantic stage's description-similarity evidence — re-run semantic analysis afterwards and the evidence values may improve.)

**CTX-07 · Optional context** — open a **new** dataset (e.g. `book`) and click Save with everything empty. Expected: saves with nulls (no validation error); all later stages work.

**CTX-08 · Empty context load** — open `parquet-sample` (registered with no context). Expected: Stage 02 loads with empty fields, no crash.

**CTX-09 · Edit existing context** — on `dq-core` change Domain "Retail" → "E-commerce", save. Expected: success; reopening shows "E-commerce" (old value replaced).

**CTX-10 · Save persists to DB** — after any save, restart backend (Ctrl+C, rerun uvicorn) and reload the page. Expected: context still shown (values came from SQLite, not browser state).

**CTX-11 · Reload persistence** — F5 on Stage 02. Expected: all saved values still present.

---

## STAGE 03 — VERSIONING & FINGERPRINTING

**VER-01 · First upload creates V1** — register TD-01 (ING-01). Expected: response shows `version_number 1`, `parent_version_id` null; Stage 03 shows "Parent version: None".

**VER-02 · Fingerprints recorded** — Stage 03 shows non-empty 64-char Schema fingerprint and Dataset content fingerprint. Cross-check via Swagger `GET /datasets/{id}/context` → same values. Independently verify file hash:
```powershell
Get-FileHash data\raw\datasets\<id>\v1\dq_core.csv -Algorithm SHA256
```
This equals the per-file fingerprint shown in the file table (SHA-256).

**VER-03 · Changed content creates V2** — on `dq-core` Stage 01: upload `TD-04 dq_core_v2.csv` **as a new dataset upload via the version endpoint** — from the UI: select TD-04 in Stage 01 of the *same* dataset flow is not available in UI; use Swagger `POST /datasets/{id}/upload` (dataset id of dq-core) with the file. Expected: `version_number 2`, `parent_version_id` = V1's id, **409 + "identical" never triggers** because content differs; Stage 03 (re-open dataset) now shows V2 with parent "Version 1".

**VER-04 · Same content rejected** — repeat the same upload (TD-04 again) on the same dataset. Expected: HTTP 409 `Uploaded content is identical to the current latest version; no new version was created.` (No V3.)

**VER-05 · Schema change detected** — upload TD-05 (adds `loyalty_tier`) via the version endpoint. Expected: V3 created; its `schema_fingerprint` differs from V2's (verify via `GET /datasets/{id}/versions`).

**VER-06 · Raw data remains unchanged** — after VER-03/05: `Get-ChildItem data\raw\datasets\<id>` shows `v1`, `v2`, `v3` folders; open `v1\dq_core.csv` and confirm its bytes still match the original TD-01 (`Get-FileHash` equals VER-02's hash).

**VER-07 · Multiple versions remain accessible** — `GET /datasets/{id}/versions` lists all versions with fingerprints and (if scored) per-version DQ scores; remediation history page also shows the lineage table.

**VER-08 · Folder fingerprint stability** — re-register folder TD-09 into a **second new dataset**, then register the identical folder again. Expected: the second attempt is flagged all-duplicates (same content fingerprint), i.e. the dataset content fingerprint captures the ordered file identities.

**VER-09 · Dataset fingerprint equality across same content** — `GET /datasets/{id}/context` content fingerprint of the folder dataset equals itself across re-runs (deterministic: file names + SHA-256s, stable order). Expected: identical value on every call.

> Added/removed/modified file tracking (per-file add/remove/modify classification) is implemented at the duplicate/file level (file-level fingerprints + duplicate matching), **not** as a dedicated per-version added/removed/modified report — mark VER results accordingly if you exercise it.

---

## STAGE 04 — PROFILING

Use dataset `dq-core` (TD-01). Open Stage 04 (auto-loads; "Generating data profile..."). Then expand the table row ("View columns").

**PRF-01 · General counts** — Expected on dq-core: Tables=1, Total rows=12, Total columns=7. Column list shows all 7 columns with pandas dtypes.

**PRF-02 · NULL counting** — `email`: null_count = 1 (Eve), plus `age`: null_count = 2; `cust_name`: null_count = 1. `null_percentage` matches count/12.

**PRF-03 · Empty strings** — `cust_name` row C006 is `"   "` (whitespace) not empty; add an empty-string column if you want exact 1s: the whitespace-only value is counted under `whitespace_only_count`, not `empty_string_count`. Verify each column's empty-string count is 0 for TD-01 (none present) and whitespace-only = 1 for `cust_name`, 1 for `email` (`"   "` row C004), 0 elsewhere.

**PRF-04 · Distinct + duplicates** — `city`: distinct = 4 (Pune, Delhi, Mumbai, Kolkata) with the padded value counted separately; `customer_id`: distinct 12 (100%); `revenue` has duplicate 0 values → `duplicate_count` ≥ 2, `duplicate_excess_count` ≥ 1.

**PRF-05 · Identifier evidence** — `customer_id` row shows the **"Identifier evidence"** badge (id-like name + 100% distinct). No column may say "Confirmed PK" anywhere in the UI — the language is evidence, not declaration.

**PRF-06 · Numeric stats** — `age` (10 non-null values: 25,34,29,41,31,55,27,38,45,52): Min 25, Max 55, Mean 37.7, Median 35, Std ≈ 9.5, Q25 ≈ 29.5, Q50 35, Q75 ≈ 42.25, IQR ≈ 12.75, MAD ≈ 8, Outliers (IQR) = 0. `revenue`: Outlier count > 0 is possible (0/2330.75 are extremes) — verify count matches an independent pandas calc:
```powershell
python -c "import pandas as pd; d=pd.read_csv(r'data\raw\datasets\<id>\v1\dq_core.csv'); q=d.revenue.quantile([.25,.75]); iqr=q[.75]-q[.25]; lo,hi=q[.25]-1.5*iqr,q[.75]+1.5*iqr; print(((d.revenue<lo)|(d.revenue>hi)).sum())"
```

**PRF-07 · Text stats** — `cust_name`: Min length 9 (Ivy Chen) – Max 12 (Parvati Rao), Mean length ≈ 10.5 (label reads **"Mean length"**, not "Mean"), Median length present; pattern counts exist for `email` (`email_like` ≥ 10).

**PRF-08 · Datetime stats** — `signup_date`: Datetime min 2024-01-15 to max 2024-12-25 (if pandas parsed it as datetime; if typed as string it appears under Text patterns with `date_like` counts — either presentation is acceptable; record which).

**PRF-09 · Edge: all-null column** — create `all_null.csv`:
```csv
id,ghost
1,
2,
```
Register + profile. Expected: `ghost` null_count = 2 (100%), no crash; text stats of zero-length values handled.

**PRF-10 · Edge: one-value column** — TD-14 `const_col`: near_constant flag true, distinct = 1.

**PRF-11 · Edge: all-unique column** — TD-12 `id`: distinct = row count (100%).

**PRF-12 · Edge: all-duplicate column** — create `all_dup.csv`:
```csv
k,v
x,1
x,1
x,1
```
Expected: `k` distinct = 1, duplicates flagged; no division-by-zero crash.

**PRF-13 · Edge: mixed types in one column** — TD-14 `mixed_col`: typed as object; text stats apply, `numeric_like` pattern count reflects parseable values (12,13,14,15).

**PRF-14 · Edge: invalid dates** — TD-03 `event_date`: parseability < 100% (mixed formats), `not_a_date` ≈ 0% parseable; no crash, evidence only.

**PRF-15 · Larger dataset** — TD-12 (~20k rows): profiling completes in seconds; numbers sane (row_count 20000).

**PRF-16 · Profiling does not modify data** — hash `dq_core.csv` before and after running profiling:
```powershell
Get-FileHash data\raw\datasets\<id>\v1\dq_core.csv -Algorithm SHA256
```
Expected: identical hashes; also Stage 03 version data unchanged.

**PRF-17 · Profile persistence** — run profiling, then `GET /datasets/{id}/profiling` again: results identical, and the second call is served with the stored profile (subsequent stages reuse it — see that semantic analyze doesn't re-profile).

---

## STAGE 05 — SEMANTIC UNDERSTANDING

Use dataset `semantics_tricky` (TD-02) for prediction quality; `dq-core` for the workflow. Open Stage 05 → "Run semantic analysis".

**SEM-01 · Obvious identifier** — `CUST_REF`: predicted concept is an identifier concept (Customer ID or similar), confidence Strong or Probable; evidence shows name similarity + datatype + uniqueness signals. (Mixed-case + abbreviation still matches via embedding + alias handling.)

**SEM-02 · Email** — register `dq_core` and run analysis: `email` → concept "Email", Strong, email-like pattern evidence listed.

**SEM-03 · Phone** — `phone_number`: expect a Phone/Contact concept or Unknown/low confidence (KB has no strong phone concept — record actual result; **must not** be forced into a wrong concept).

**SEM-04 · Date** — `order_dt`: date/time concept, datatype compatibility evidence active.

**SEM-05 · Amount/revenue** — `amount` and `revenue`: monetary concept (Revenue/Amount family) or Unknown — record actual; alternatives list should include related concepts, not name-family confusion.

**SEM-06 · Customer name** — `cust_nm`: customer-name concept (abbreviation handled); `cust_name` (dq-core) likewise.

**SEM-07 · customer_type must NOT map to customer_name** — `cust_type`: expected concept is a Customer Type/Segment concept or Unknown — **never** Customer Name. Check the alternatives list: if Customer Name appears, its score must be below the top candidate. (This is the lexical-vs-semantic guardrail test.)

**SEM-08 · Ambiguous column** — `memo` (free text): likely Unknown/Ambiguous with alternatives ≤ threshold; system does not invent a concept.

**SEM-09 · Unknown column** — `xyz_qq`: concept UNKNOWN/None with "insufficient evidence" presentation; **no concept silently forced**. The UI shows "Unknown — insufficient evidence" in the dropdown and the Approve button stays available only with a concept.

**SEM-10 · Column with description** — in Stage 02 of `dq-core`, describe `email` as "Primary contact email address"; re-run semantic analysis. Expected: description-similarity evidence becomes active/non-zero for the Email mapping (value changes vs the no-description run).

**SEM-11 · Column without description** — fresh dataset (no Stage 02 inputs): evidence still includes name/embedding/datatype/profile; description evidence = 0 and does not break scoring.

**SEM-12 · Different naming conventions** — `CUST_REF` (SCREAMING), `cust_nm`, `phone_number`, `order_dt`: each maps sensibly (record per-column result). Consistency: snake_case, CamelCase, and abbreviations are all normalized before matching.

**SEM-13 · Abbreviations** — `cust_nm` vs `cust_type` disambiguation (see SEM-06/07); `CUST_REF` (see SEM-01).

**SEM-14 · Mixed case** — `CUST_REF`, `UPPER FLAG` (weird_cols): no crash; `CUST_REF` matches identifier family.

**SEM-15 · Underscores** — `signup_date`, `order_dt` map through underscore tokenization.

**SEM-16 · Evidence display** — expand any prediction: Evidence list shows named signals (Embedding similarity, Name similarity, Description similarity, Datatype compatibility, Profile compatibility, Context similarity) with numeric values; active signals have checkmarks; zero-signal entries are visibly inactive.

**SEM-17 · Confidence levels honest** — across the two datasets you should see at least two different levels (e.g. Strong for `customer_id`, Unknown for `xyz_qq`). Levels shown: Strong / Probable / Ambiguous / Unknown — never a fake probability percentage presented as calibrated.

**SEM-18 · Alternatives list** — every prediction with a concept shows ranked alternatives with scores (top-5, descending).

**SEM-19 · Approve** — approve `CUST_REF`'s mapping. Expected: badge changes to Approved; the row's decision persists (see SEM-21); re-running analysis keeps this decision (approved mappings are not reset).

**SEM-20 · Reject** — reject `phone_number`'s mapping (if it had one). Expected: Rejected badge; concept cleared; feedback recorded (`GET /datasets/{id}/semantic/feedback` count increases with decision "rejected").

**SEM-21 · Edit** — in the "Confirm concept" dropdown choose a **different** concept for a pending column (e.g. map `cust_type` explicitly to whatever type concept exists, or map `city` to a Geographic concept). Expected: status "edited", chosen concept stored, feedback decision "edited" with both original and corrected concept ids.

**SEM-22 · Persistence** — F5 and reopen the dataset: approvals/edits/rejections still shown; `GET /datasets/{id}/semantic` returns stored statuses (SQLite-backed, not React state).

**SEM-23 · Historical evidence** — approved alias history influences future runs only through the KB admission flow (`POST /kb/aliases/propose` → human admission); there is no automatic promotion from a single approval. Mark as verified-if-exercised via the KB endpoints; not a UI screen.

---

## STAGE 06 — METRIC & RULE RECOMMENDATION

Use `dq-core`. Prereq: profiling done (Stage 04) and **approve** these semantic mappings in Stage 05 first: `customer_id` → Customer ID (or your identifier concept), `email` → Email. Then Stage 06 → "Generate recommendations".

**REC-01 · Identifier → completeness** — Expected: rule "`customer_id` must not be empty" (metric completeness), evidence = approved semantic concept + observed completeness %, risk low, status recommended.

**REC-02 · Identifier → uniqueness (only with evidence)** — `customer_id` (100% distinct): rule "`customer_id` values must be unique" (metric uniqueness), risk medium. If you add a duplicate-laden identifier column (distinct < 99%), **no** uniqueness rule is recommended — verify by testing on TD-02 (`CUST_REF` 5/5 distinct is exactly 100%, so it does get the rule; a column with dupes does not).

**REC-03 · Email → validity** — rule "`email` must be a valid email address" (metric validity), evidence = approved Email concept + observed email-like pattern count.

**REC-04 · No rules without approved semantics** — on a dataset where Stage 05 decisions are all pending/rejected: Generate recommendations → 0 rules with the message "No rules recommended. Approve semantic mappings in stage 05…". (This proves recommendations are gated on human confirmation, not raw name matching.)

**REC-05 · No business ranges from observed min/max** — TD-01 `age` (observed 25–55): **no** rule like "age between 25 and 55" may be generated. Same for `revenue` (observed 0–2330.75). This is the core safety test: the recommender has no range/allowed-value rule types at all — confirm none appear.

**REC-06 · Numeric/date fields produce no implicit rules** — `age`, `revenue`, `signup_date` generate no rules by themselves (no approved identifier/email semantics on them). Expected: nothing recommended for them.

**REC-07 · Rule ranking/coverage display** — multiple recommendations are all shown with metric + risk + validation badges; each card exposes "Show evidence" with the raw `rule_json` DSL and validation issues.

**REC-08 · Consistency rules NOT IMPLEMENTED** — verify honestly: there is no cross-column rule (e.g. `quantity × price = total`) in the DSL. Confirm the recommender never emits one. Mark `NOT IMPLEMENTED` in notes.

**REC-09 · Referential-integrity rules NOT IMPLEMENTED** — on the folder dataset (TD-09) approve identifier semantics on `customers.customer_id` and `orders.customer_id`: no RI/cross-table rule is generated (only per-table completeness/uniqueness if evidence warrants). Mark `NOT IMPLEMENTED` in notes; relationship detection itself is not implemented (no candidate-relationship UI).

**REC-10 · Regeneration replaces pending recommendations** — click "Generate recommendations" twice. Expected: old unapproved recommendations are replaced (no duplicates); already approved rules are kept.

---

## STAGE 07 — RULE VALIDATION

**VAL-01 · Valid rule** — after REC-01..03, click "Validate pending". Expected: `customer_id` completeness and uniqueness rules → **VALID** (no issues); email validity → **VALID** (email-like evidence high).

**VAL-02 · Invalid column** — via Swagger `POST /rules/{id}/approve` is approval; to see INVALID validation use Swagger: `POST /datasets/{id}/recommendations` won't create bad rules, so inject one via the DB or use an edited approval with a non-existent column: `PUT`-style edit through `POST /rules/{id}/approve` with `edited_rule_json = {"type":"completeness","table":"customers","column":"does_not_exist","condition":"not_null"}` — approval is **rejected** with 400 "Edited rule is invalid…" because validation returns INVALID ("Column does_not_exist not found in table customers").

**VAL-03 · Low coverage warning** — on a dataset where an email column has < 50% email-like values (create `bad_emails.csv` with 4 junk of 5 values), approve its Email semantic and generate: validation status **NEEDS_REVIEW** with "Only X/Y observed values look email-like - rule may fail broadly."

**VAL-04 · Uniqueness warning** — uniqueness rule on a column with distinct < 95% → NEEDS_REVIEW with "observed distinct ratio … may fail." (Force by editing an approval onto a dup-heavy column as in VAL-02.)

**VAL-05 · Unsupported type → INVALID** — edited rule `{"type":"range","column":"age","min":18,"max":120}` must fail validation: "Unsupported rule type: range" (the DSL has no range rule — this also re-proves REC-05).

**VAL-06 · Validation ≠ execution** — validating never runs the rule: no `rule_executions` rows appear (`GET /datasets/{id}/executions` count unchanged) after "Validate pending".

**VAL-07 · Statuses displayed** — rule cards show VALID / NEEDS_REVIEW / INVALID badges sourced from the DB (`validation_status`).

---

## STAGE 08 — HUMAN APPROVAL & EXECUTION

**EXE-01 · Approve rule** — Approve the `customer_id` completeness rule. Expected: status Approved; a `RuleApproval` row records the decision (check DB or behavior: rule stays approved after re-generating recommendations).

**EXE-02 · Reject rule** — Reject the uniqueness rule. Expected: status Rejected; excluded from execution.

**EXE-03 · Edit rule (API)** — Swagger `POST /rules/{id}/approve` with `decision: approved` + `edited_rule_json` (e.g. switch the email rule's `check` to `numeric_type` then back). Expected: INVALID edits are refused (see VAL-02); valid edits replace `rule_json` and are logged.

**EXE-04 · Execute approved rules** — click "Execute approved rules (N)". Expected: results section appears with per-rule: Total rows 12, Applicable rows, Passed, Failed, Pass rate, Violation rate; `customer_id` completeness passes 12/12 (100% pass rate).

**EXE-05 · Unapproved rules never execute** — with 0 approved rules the Execute button is disabled; via Swagger `POST /datasets/{id}/execute` returns 400 "No approved rules to execute." Also `execute_rule` refuses non-approved rules at the service layer.

**EXE-06 · Failing rule** — approve the email validity rule (it has 2 failures: empty, whitespace are excluded from validity — the malformed `carla@gnail.com` is the real failure; also C004's `"   "` is missing → excluded). Execute. Expected: Applicable = 11 (non-null emails), Failed ≥ 1 (`carla@gnail.com`), Pass rate < 100%, bounded failure examples shown (row index + JSON of failing row, ≤ 20 examples).

**EXE-07 · Partially applicable rule** — `age` completeness (if approved via edit on a value-bearing column) or the email rule above: applicable < total because nulls are excluded from validity checks. Verify the UI shows Applicable < Total rows.

**EXE-08 · Zero applicable rows** — create `zero_applicable.csv` with one column, all null; approve a completeness rule on it (via semantic edit + recommendation); execute. Expected: applicable_rows = 0, pass_rate 0.0, no crash; scoring later marks the metric not-applicable (N/A).

**EXE-09 · Execution error handling** — delete `data\raw\datasets\<id>\v1\dq_core.csv` (move it aside temporarily) and execute an approved rule: the rule reports status "error" with the message in the results list (`Rule execution failed` path) and the run continues for other rules; restore the file afterwards. (Score never treats the errored rule as passed.)

**EXE-10 · Failure evidence bounded** — for a rule with ≥ 20 failures (use TD-12 `value`-style data or the email dataset), the UI shows at most 20 example rows (`MAX_EVIDENCE_EXAMPLES = 20`), never the full dataset.

**EXE-11 · No arbitrary code** — attempt an edited approval whose `rule_json` contains extra keys (e.g. `"payload": "import os"`): execution ignores unknown keys (DSL is a fixed interpreter: completeness/uniqueness/validity only; no eval anywhere in `rule_execution.py`). Confirm by approving such a rule and executing: behavior identical to the clean rule.

**EXE-12 · Re-execution records new rows** — click Execute twice. Expected: two execution result sets (`GET /executions` count grows); latest results feed scoring.

---

## STAGE 09 — DQ SCORING

Open Stage 08 ("Scoring & RCA" tab). The score loads from the last execution round; use "Recalculate score" after new executions.

**SCR-01 · Metric breakdown present** — after executing the three dq-core rules: metrics `completeness`, `uniqueness` (if approved), `validity` each show score %, applicable records, failed records, rule count.

**SCR-02 · Overall = mean of available metrics** — with completeness 100% and validity ~90.9%: overall ≈ mean of the available metric scores (unweighted). Verify arithmetic against the displayed metric scores.

**SCR-03 · Denominators are applicable rows** — validity: 1 failed / 11 applicable ≈ 90.91% (not 1/12). Completeness: 0/12 = 100%. No N/A confusion.

**SCR-04 · N/A metrics listed** — if a rule executed against an empty-applicable population (EXE-08) or no rule exists for a metric, it appears under "N/A (no applicable records): …" and is excluded from the mean. `excluded_metrics` count reflects it.

**SCR-05 · No applicable metrics** — on a dataset with zero executed rules, `POST /datasets/{id}/scores` returns 400 "No approved rules exist. Approve and execute rules first." (UI shows the warning notice; no fake 100%.)

**SCR-06 · Execution errors never count as passing** — after EXE-09 (file missing), recalculate: the errored rule's metric is not silently scored 100; verify metric scores derive only from real executions and the incident is visible in the run's error count.

**SCR-07 · Multiple rules per metric** — approve two completeness rules (e.g. `customer_id` + `cust_name`) and execute: the completeness metric aggregates by **records**, not by averaging rule percentages: score = (Σapplicable − Σfailed) / Σapplicable. Verify with the displayed per-rule numbers.

**SCR-08 · Critical failures visible despite high overall** — engineer: overall > 90 while validity < 90 (e.g. an email dataset with many malformed values → validity ~40%, completeness 100% → overall ~70; or add more passing rules to lift overall above 90 with validity below 90). Expected: red "Critical rule failures visible below the overall score" notice lists the failing metric(s) — the summary never hides them.

**SCR-09 · Weighted score** — `PARTIALLY IMPLEMENTED`: `compute_dq_score(weighted=True)` exists with default equal weights (1.0 each), but no endpoint/UI exposes weighting. Verify via Python directly if desired; the UI always shows unweighted. Record as partial.

**SCR-10 · Score persists per version** — `GET /datasets/{id}/scores` after reload shows the same stored score (`computed_at` unchanged); Stage 03/09 lineage table shows per-version scores.

---

## STAGE 09b — ROOT CAUSE ANALYSIS (RCA)

Prereq: at least one executed rule with failures (email rule from EXE-06). Click **Run RCA**.

**RCA-01 · Findings only for real failures** — findings list contains one entry per failed execution; rules with 0 failures produce none ("No rule failures to analyze." when nothing failed).

**RCA-02 · Failure grouping** — the email finding shows failure count (1 for dq-core) and violation rate; the analysis re-executes the rule deterministically (same numbers as Stage 08).

**RCA-03 · Pattern concentration** — the finding's "Value patterns" section breaks failed values into `leading or trailing whitespace` / `empty string` / `other` with counts and percentages. For dq-core's email failure the malformed address lands in "other".

**RCA-04 · Category concentration** — "Concentrated in — <column>" sections list top values of other categorical columns among failing rows with percentages (e.g. city distribution of the failing row). With 1 failure the sections may be sparse — use TD-12-style data with many failures for richer output.

**RCA-05 · Correlation ≠ causation language** — the RCA panel shows the language note: percentages "describe observed concentrations among failed rows only. They do not establish causality." **No** "caused by"/"root cause confirmed" wording may appear anywhere in RCA output.

**RCA-06 · No failures → clean state** — on a dataset where all rules passed: `POST /rca` returns `findings: []` + message; UI shows no findings section.

**RCA-07 · Findings persist** — reload: `GET /datasets/{id}/rca` returns stored findings (`RCAFinding` rows), not a transient computation.

> Source-system / batch / time concentration and schema-version correlation: the current implementation concentrates on **value-level and column-level** distributions among failed rows (patterns, categorical concentration, repeated failure values). Source/batch/time slices are NOT implemented — mark accordingly in notes.

---

## STAGE 10 — REMEDIATION & REASSESSMENT

Use `dq-core` (has the `"   "` whitespace values in `cust_name`/`email`/`city`).

**REM-01 · Propose with preview** — Stage 09 → "Propose remediation". Expected: proposal lists each affected column (`cust_name`, `email`, `city`) with affected-row counts, risk "low", safety note, and before → after samples (e.g. `"␣␣Mumbai" → "Mumbai"`).

**REM-02 · Affected-row accounting** — corrections count equals the total changed cells (whitespace-affected); affected rows ≤ corrections (a row can have several corrected cells).

**REM-03 · Approve and apply** — click "Approve and apply". Expected: message "Remediation applied and reassessed. N corrections applied. V2 created." with score before → after and delta.

**REM-04 · New immutable version + lineage** — Version lineage table now shows V2 with parent V1; on disk `data/raw/datasets/<id>/v2/` contains the corrected CSV **under the same stored filename**; `v1/` untouched (hash-verify TD-01 as in PRF-16).

**REM-05 · Automatic reassessment** — the decision response includes `reassessment`: approved rules were re-run on V2 (`failed_rows` per rule) and a new DQ score computed (`after_score`). The email validity failure on `carla@gnail.com` **remains** on V2 — score must not jump to 100.

**REM-06 · Resolved vs remaining** — before/after: whitespace-only failures in completeness rules (if any) resolve; the malformed email remains (remediation never rewrites values semantically). Resolved + Remaining + New should reconcile with the rule counts.

**REM-07 · Score delta sanity** — delta = after − before as displayed; if no rules were approved before remediation, `after_score` is null and the message shows "—" (no fabricated score).

**REM-08 · Reject remediation** — on a fresh proposal: click "Reject". Expected: status "rejected", no new version (`GET /versions` count unchanged), a `RemediationApproval` row records the rejection.

**REM-09 · No unsafe fuzzy corrections** — `carla@gnail.com` must still be exactly `carla@gnail.com` in V2 (open the corrected CSV). The system has **no** email-correction capability by design: confirm no "did you mean" suggestion appears anywhere. This is the negative safety test from the spec.

**REM-10 · Already-clean data** — propose remediation on a dataset with no whitespace issues (e.g. `book`): message "No safe corrections found. Whitespace normalization found nothing to fix…"; approving is a no-op (0 corrections).

**REM-11 · Approved semantics carry forward** — after remediation creates V2, open Stage 05 for `dq-core`: V2's predictions show the previously approved mappings already approved (no re-approval needed); Stage 06 can generate rules for V2 without redoing Stage 05. (Verified behavior: approved semantic decisions are copied to the new version.)

---

## STAGE 11 — MONITORING

Prereq: `dq-core` has ≥ 2 versions (V1 original, V2 from remediation, or V2/V3 from version uploads).

**MON-01 · Score trend / metric trend** — `PARTIALLY IMPLEMENTED` as cross-version numbers: the versions table (Stage 09 remediation screen) shows a DQ score per version — a de-facto trend across versions. There is **no** time-series chart. Record accordingly.

**MON-02 · Run monitoring (default baseline)** — Stage 10 → "Run monitoring". Expected: compares latest version against the immediately previous version; summary message "Compared N columns against V(prev). X column(s) drifted, schema changed/unchanged."

**MON-03 · Schema drift via fingerprint** — on a dataset where a version upload changed the schema (VER-05): Schema badge shows "Changed" and `schema_changes` lists the fingerprint change.

**MON-04 · Numeric drift (KS)** — upload a version where `revenue` values shift strongly (e.g. all revenues × 10) via the version endpoint, then run monitoring: `revenue` row shows method `KS`, statistic, threshold 0.15, and "Drift detected" when the KS statistic exceeds it.

**MON-05 · Categorical drift (PSI)** — create a version where `city` frequencies change (e.g. 90% Pune): `city` row shows method `PSI`, threshold 0.2, drift detected when PSI > 0.2.

**MON-06 · Stable comparison** — remediation-only change (TD-01 → TD-04 whitespace fix): most columns "Stable"; small statistics well under thresholds. (Exact values depend on data; record them.)

**MON-07 · Small-sample honesty** — columns with < 30 non-null values in either version appear under "Insufficient sample" with counts and "minimum 30 required" — **no drift alert** is produced for them. (TD-01's 12-row columns all fall here; expect most columns listed as insufficient — this is correct behavior.)

**MON-08 · Missing baseline** — on a dataset with only V1: run monitoring. Expected: 400 "No baseline version available. Upload a new version or specify baseline_version_id." UI shows the danger notice.

**MON-09 · Baseline selection** — Swagger `POST /datasets/{id}/monitoring` with `{"baseline_version_id": <V1 id>}` compares V1 → latest explicitly; with a mismatched id → 400 "Baseline version does not belong to this dataset."; with baseline == comparison → 400.

**MON-10 · Results persist** — drift rows are stored (`DriftResult`): after reload the same statistic values are shown again when re-run; `methods` block documents why each method exists (KS for numeric, PSI for categorical, schema fingerprints for schema).

**MON-11 · Relationship drift / anomaly detection** — `NOT IMPLEMENTED` (no relationship detection, no Isolation Forest). Confirm the monitoring output contains no fabricated sections for them.

---

## STAGE 12 — FEEDBACK & OFFLINE LEARNING

**FB-01 · Semantic approval feedback** — approve a mapping in Stage 05. Expected: `GET /datasets/{id}/semantic/feedback` shows decision "approved" with dataset/version/column and model name (`all-MiniLM-L6-v2`); Stage 11 summary increments "Approved".

**FB-02 · Semantic rejection** — reject a mapping: feedback decision "rejected", original concept id recorded.

**FB-03 · Semantic edit** — edit a mapping to a different concept: decision "edited" with original + corrected concept ids.

**FB-04 · Rule approval feedback** — approving/rejecting rules records `RuleApproval` rows (see EXE-01/02); visible indirectly through rule statuses persisting across regeneration.

**FB-05 · Remediation decisions recorded** — approve and reject remediation proposals: `RemediationApproval` rows exist (behavioral check: rejected proposals cannot be decided again — 400 "Remediation has already been decided.").

**FB-06 · Feedback persistence** — restart backend, reload Stage 11: counts unchanged (SQLite-backed).

**FB-07 · Training gate: insufficient data** — click "Train candidate model" with < 20 labeled decisions. Expected: message "Need at least 20 labeled feedback examples; found N…" and **no** model created (`GET /models` unchanged).

**FB-08 · Training gate: single class** — with ≥ 20 approvals but **no** rejections/edits: "Feedback contains only one class… Both positive and negative human decisions are required to train."

**FB-09 · Training runs with sufficient data** — accumulate ≥ 20 mixed labels (loop approve/reject across columns/datasets): training returns `trained` with version, held-out accuracy and F1, class counts, and feature weights (transparent logistic reranker); `models/ranking/semantic_reranker_v1.json` appears on disk; model status "candidate".

**FB-10 · Held-out evaluation** — the response metrics include `train_count`/`test_count` (80/20 split) and `accuracy`/`f1` computed on the held-out part only. The registry table shows the held-out accuracy.

**FB-11 · Promotion is explicit** — "Promote" on a candidate: status becomes "promoted", previous promoted version (if any) becomes "archived". Nothing is promoted automatically after training (`status_note`: "It is NOT active until promoted.").

**FB-12 · No live retraining / no self-labeling** — confirm by design: predictions are never used as labels (only human decisions from `semantic_feedback` enter training; rejected/edited/approved decisions map to labels 0/0-or-1/1); no training happens during semantic analysis or rule runs. Re-running analysis after promotion does not change any stored prediction.

---

## CROSS-STAGE END-TO-END TEST

**E2E-01 · Full pipeline on one dataset**

- Preconditions: fresh dataset name `e2e-run`, backend + frontend running, TD-01 available.
- Input: TD-01 `dq_core.csv`.
- Steps and expected results at every hop:

| Step | Action | Expected result |
|---|---|---|
| 1 | Stage 01: register TD-01 as `e2e-run` | Success notice; Dataset ID noted; V1; files=1 |
| 2 | Stage 02: save domain "Retail", source "Salesforce" | Success; stage 02 ✓ |
| 3 | Stage 03: inspect version | V1, no parent, two 64-char fingerprints, file hash row |
| 4 | Stage 04: profile | 12 rows, 7 columns; identifier evidence on `customer_id`; whitespace evidence on `cust_name` |
| 5 | Stage 05: run analysis; approve `customer_id` (identifier) and `email` (Email); reject `city`'s mapping if wrong | Approved badges; feedback counts increment |
| 6 | Stage 06: generate recommendations | ≥ 2 rules (completeness on `customer_id`, validity on `email`; uniqueness on `customer_id` because 100% distinct) |
| 7 | Stage 06: Validate pending | All VALID (email maybe NEEDS_REVIEW only if pattern evidence low — here it is high, expect VALID) |
| 8 | Stage 06: Approve completeness + validity rules; reject uniqueness | Approved badges on 2 rules |
| 9 | Stage 06: Execute approved | Results: completeness 12/12 pass; validity 1 failure (`carla@gnail.com`); bounded evidence rows |
| 10 | Stage 08: open Scoring | Overall = mean(100, ≈90.91) ≈ 95.45%; metric table matches per-rule math; validity < 100 but > 90 → no critical-failure notice (note it) |
| 11 | Stage 08: Run RCA | 1 finding for the email rule; patterns show the malformed address under "other"; language note visible |
| 12 | Stage 09: Propose remediation | Whitespace corrections on `cust_name`/`email`/`city` with previews |
| 13 | Stage 09: Approve and apply | V2 created (parent V1); reassessment re-runs both rules; email failure **remains** (1), completeness clean; after-score ≈ 95.45% |
| 14 | Stage 10: Run monitoring | Baseline V1 → comparison V2; city column may show PSI movement from the whitespace fix; small-sample note if < 30 rows (expected for this dataset) |
| 15 | Stage 11: Feedback review | approved=2, edited=0, rejected=1 (+ any extra decisions); model training still gated (< 20 labels) |
| 16 | Persist everything | Restart backend + browser; reopen `e2e-run`: all stage tabs ✓ per persisted progress; V2 still latest; scores and findings still present |

**E2E-02 · Folder dataset end-to-end (multi-table)**

- Input: TD-09 folder (`customers.csv`, `orders.csv`, `reviews.csv`).
- Steps: register folder → context → profiling → semantic (approve `customer_id`/`order_id` as identifiers) → recommendations → validation → approve → execute → score.
- Expected: **table boundaries preserved** — `customers`, `orders`, `reviews` appear as separate tables in profiling with their own column sets; rules are table-scoped (rule cards show `table_name`); completeness/uniqueness execute per table; scoring aggregates across the executed rules.
- Relationship test: `NOT IMPLEMENTED` — the system does not detect `customers.customer_id → orders.customer_id` or `orders.order_id → reviews.order_id`, does not propose RI rules, and shows no relationship evidence screen. Mark explicitly as N/A in results; do not fabricate relationship output.

---

## UI TESTING

**UI-01 · Stage navigation** — all 11 tabs visible: 01 Ingestion … 11 Feedback; tabs 02+ disabled until a dataset is opened (tooltip "Register or open a dataset first").

**UI-02 · Back buttons** — each stage screen (02–11) has a back control returning to the dataset entry; no dead ends.

**UI-03 · Registered datasets list** — Stage 01 side panel lists every dataset with name, V-number, file/table counts, source system, domain; Refresh button works; >6 datasets reveal a filter box.

**UI-04 · Dataset selection** — clicking a dataset opens Stage 02 with that dataset's context; the active item is highlighted in the list.

**UI-05 · Stage completion markers** — completed stages show a check icon in the tab (persisted state), the current tab is highlighted, and the first incomplete stage is the natural "continue" point.

**UI-06 · Incomplete stages** — a dataset opened mid-pipeline (e.g. after registration only) shows later stages not-done and their data areas in proper empty states (no fake data).

**UI-07 · Refresh (F5)** — on any stage: reload returns to Stage 01 list view (single-page state resets) but the dataset's stage progress and all results are re-fetched from the backend — nothing user-created is lost.

**UI-08 · Browser reopen** — close the tab, reopen `http://localhost:5173`, reopen the dataset: identical state to UI-07 (backend-persisted).

**UI-09 · Empty states** — every stage shows a designed empty state when no data yet (datasets list, semantic predictions before analysis, rules before generation, monitoring before run, feedback before decisions, model registry before training). No layout breakage.

**UI-10 · Loading states** — "Generating data profile...", "Analyzing…", "Executing…", "Comparing…", "Training…" spinners/text appear during work; buttons disable while busy.

**UI-11 · API errors** — stop the backend and refresh: dataset list shows the connection warning; inline error notices appear on each stage's data calls. Restart the backend and press Retry/Refresh — recovery works.

**UI-12 · Validation errors** — duplicate upload 409 detail renders with file names; 400 upload errors (empty/unsupported/oversize) render their exact messages; edited-invalid rule approval shows the backend issues.

**UI-13 · Success messages** — registration, context save, semantic decisions, execution, remediation decisions all show green success notices (text verified in the respective stage tests).

**UI-14 · Warning messages** — "Upload requires review" (mixed duplicates), critical-failure notice in scoring, insufficient-sample notice in monitoring, approval-required notice in remediation.

**UI-15 · Responsive behavior** — narrow the window: layout splits stack; tables scroll horizontally rather than truncating silently; nothing overlaps. (Practical pass/fail at ~1280px and ~900px widths.)

**UI-16 · Table expansion** — Stage 04 "View columns / Hide columns" toggles per table with keyboard support (Enter/Space); Stage 08/09/10 detail disclosures expand/collapse.

**UI-17 · Column-level profiling display** — expanded table shows per-column: dtype, nulls (+%), distinct values (+%), empty strings, whitespace-only, identifier badge, numeric/text/datetime detail lines. Wording uses "Mean length" for text and "Distinct values" (spec-required phrasing).

**UI-18 · Buttons enable/disable correctly** — Execute disabled with 0 approved rules; Validate pending disabled with 0 pending rules; Approve disabled on semantic rows without a concept; Promote shown only for candidates.

**UI-19 · Approval actions everywhere** — semantic Approve/Reject/Edit-select; rule Approve/Reject; remediation Approve/Reject; model Promote. Each persists (see stage tests).

**UI-20 · Theme consistency** — new screens reuse the same design system: identical cards, badges, buttons, tables, notices, spacing, typography as Stages 01–04 (no divergent styling anywhere in 05–11).

**UI-21 · Filters/search** — dataset filter on Stage 01 (>6 datasets) filters by name; empty filter result shows "No datasets match this filter."

---

## PERSISTENCE TEST (mandatory)

For each row: perform the action → close/reopen the browser (or F5) → verify.

| ID | Operation | Verify after reload |
|---|---|---|
| PER-01 | Dataset registered | Listed in Stage 01; version + fingerprint intact |
| PER-02 | Context saved | Values shown again (CTX-10/11) |
| PER-03 | Version uploaded (V2) | Lineage + fingerprints for all versions present |
| PER-04 | Profile generated | Same numbers on re-open (stored profile reused) |
| PER-05 | Semantic approved/edited/rejected | Statuses persist; re-analysis keeps decisions |
| PER-06 | Recommendations generated | Rules listed with statuses (approved rules survive regeneration) |
| PER-07 | Rules approved/rejected | Statuses persist; approved not re-set to recommended |
| PER-08 | Rules executed | Execution results with identical numbers |
| PER-09 | Score computed | Same overall + metric breakdown + computed_at |
| PER-10 | RCA run | Findings identical |
| PER-11 | Remediation applied | V2 lineage, correction counts, before/after scores |
| PER-12 | Monitoring run | Re-run yields same statistics (and DriftResult rows persist) |
| PER-13 | Feedback recorded | Stage 11 counts unchanged |
| PER-14 | Backend restart (not just browser) | Everything above still true (true backend persistence, not browser cache) |

---

## REGRESSION TEST

After manual testing, run the automated suite and record results:

```text
Command:    cd backend; ..\.venv\Scripts\python.exe -m pytest -q
Frontend:   cd frontend; npm run build
```

| Item | Result |
|---|---|
| Command used | |
| Test count | (current: 89 collected) |
| Passed | |
| Failed | |
| Warnings | (current: 1 deprecation warning from testclient/anyio — harmless) |
| Frontend build | (expected: `✓ built in ~1–2s`, exit 0) |

---

## FINAL ACCEPTANCE CHECKLIST

```text
[ ] Backend starts
[ ] Frontend starts
[ ] Database works
[ ] Upload works
[ ] Context works
[ ] Versioning works
[ ] Fingerprinting works
[ ] Profiling works
[ ] Column-level profiling works
[ ] Semantic understanding works
[ ] Metric recommendation works
[ ] Rule recommendation works
[ ] Rule validation works
[ ] Human approval works
[ ] Rule execution works
[ ] DQ scoring works
[ ] RCA works
[ ] Remediation works
[ ] New version created
[ ] Reassessment works
[ ] Monitoring works
[ ] Feedback works
[ ] Automated tests pass
[ ] Frontend build passes
[ ] UI is consistent
[ ] No major placeholder functionality remains
```

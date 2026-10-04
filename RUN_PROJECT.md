# RUN_PROJECT.md — Running the DQ Assessment System on Windows (PowerShell)

This guide was written against the **actual current repository**. Every command,
URL, path, port, and endpoint below was verified against the code on disk.

```text
Intelligent Data Quality Assessment, Metric & Rule Recommendation,
and Monitoring System

backend/   FastAPI + SQLAlchemy + SQLite API
frontend/  React 19 + TypeScript + Vite UI
data/      SQLite database + immutable raw dataset storage (created at runtime)
```

---

## 1. Prerequisites

Install exactly these. Versions below are the ones the project is verified with;
minimums come from `backend/requirements.txt` and `frontend/package.json`.

| Tool | Verified version | Notes |
|---|---|---|
| **Python** | 3.14.6 | 3.11+ required (`requirements.txt` uses `X | None` type syntax) |
| **Node.js** | 22.23.2 (v22 LTS) | 18+ works; Vite 8 needs Node 20.19+ |
| **npm** | 10.9.8 | ships with Node |
| **Git** | any recent | only needed to clone the repo |
| **VS Code** | any recent | recommended editor |

Notes:

- The backend has **no required environment variables**. All configuration is in
  `backend/app/config.py` (file-size limits, semantic model name, confidence
  thresholds, feedback-training minimums) and `backend/app/database.py`
  (SQLite path). Nothing needs to be set before starting.
- The Python packages are **not bundled**. The repo contains `.venv/` locally, but a
  fresh clone must create its own (step 3).
- First semantic analysis run downloads the `all-MiniLM-L6-v2` Sentence-Transformer
  model (~90 MB) from Hugging Face; an internet connection is needed once (or copy
  the local HF cache to the new machine).

---

## 2. Open the project in VS Code

```powershell
cd C:\path\to\DQ-assessment   # the folder containing backend/ and frontend/
code .
```

---

## 3. Backend setup

### Terminal 1 — Backend (keep this terminal open)

```powershell
# 1. Create a virtual environment (skip the first line if .venv already exists)
cd backend
python -m venv ..\.venv

# 2. Activate it
..\ .venv\Scripts\Activate.ps1     # see note below
```

> The activation path is `..\.venv\Scripts\Activate.ps1` (no space after `..\`).
> If PowerShell blocks the script, run once:
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

```powershell
# 3. Install dependencies (verified versions are pinned in requirements.txt)
pip install -r requirements.txt

# 4. Start the API server
..\ .venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8000
# (correct form: ..\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8000)
```

Simplest exact form that works from `backend/` inside the activated venv:

```powershell
cd backend
..\ .venv\Scripts\Activate.ps1   # -> ..\.venv\Scripts\Activate.ps1
python -m uvicorn app.main:app --reload --port 8000
```

What happens on startup:

- `Base.metadata.create_all(bind=engine)` runs automatically — the SQLite
  database **initializes itself** at `data/dq_assessment.db`. There is no
  migration or seed command to run manually.
- The semantic knowledge base (32 canonical concepts) is seeded lazily the first
  time `GET /kb/concepts` is called.
- Leave this terminal running. You should see
  `Uvicorn running on http://127.0.0.1:8000`.

---

## 4. Backend verification

- **Backend base URL:** `http://127.0.0.1:8000`
- **Swagger UI:** `http://127.0.0.1:8000/docs` (verified: HTTP 200)
- **Health endpoint:** `http://127.0.0.1:8000/health` — returns
  `{"status":"healthy","service":"DQ Assessment API"}`
- **`http://127.0.0.1:8000/` returns 404 — this is intentional.** No root route is
  defined; use `/docs` or `/health`.

Verification commands (second terminal or after startup completes):

```powershell
curl.exe -s http://127.0.0.1:8000/health
curl.exe -s http://127.0.0.1:8000/datasets
curl.exe -s http://127.0.0.1:8000/kb/concepts
```

Expected: JSON from all three. `/datasets` returns `{"datasets":[...],"count":N}`
(empty list on a fresh database). `/kb/concepts` seeds and returns 32 concepts.

---

## 5. Frontend setup

### Terminal 2 — Frontend (keep this terminal open)

```powershell
cd frontend
npm install        # only needed on a fresh clone
npm run dev
```

- **Frontend URL:** `http://localhost:5173` (Vite default). Open it in a browser.

---

## 6. Frontend/backend connection

| Item | Actual value | Where it is defined |
|---|---|---|
| Frontend URL | `http://localhost:5173` | Vite dev server |
| Backend URL | `http://127.0.0.1:8000` | uvicorn |
| API base URL in frontend | `http://127.0.0.1:8000` | hardcoded in `frontend/src/api.ts` (`API_BASE_URL`) |
| CORS allow-list on backend | `http://localhost:5173` and `http://127.0.0.1:5173` | `backend/app/main.py` (`CORSMiddleware`) |

How it communicates: the UI calls the REST API with `fetch()` (JSON bodies for
API calls, `multipart/form-data` for uploads). All stage results are read back
from the backend on every page load — stage progress comes from
`GET /datasets/{id}/stages` (persisted in SQLite), so a refresh never loses state.

If you serve the frontend on a different port, CORS will reject requests — either
keep 5173 or add the port to `allow_origins` in `backend/app/main.py`.

---

## 7. Database and storage

All paths are relative to the repository root and are created automatically:

| What | Path | Notes |
|---|---|---|
| SQLite database | `data/dq_assessment.db` | single file; delete it to reset everything |
| Raw immutable uploads | `data/raw/datasets/{dataset_id}/v{version_number}/{filename}` | e.g. `data/raw/datasets/25/v1/customers.csv`; **never modified after write** |
| Remediation result versions | `data/raw/datasets/{dataset_id}/v{n+1}/...` | created by remediation as new immutable versions (V5 = remediation child of V4, etc.) |
| Upload temp area | `data/processed/dq_upload_*`, `dq_analyze_*` | created per request and deleted in `finally` |
| Persisted profiles | inside SQLite (`stored_profiles.profile_json`) | not files on disk |
| Drift results, scores, executions, feedback, KB entries | inside SQLite | |
| Trained model files | `models/ranking/semantic_reranker_v{N}.json` | written only when offline training runs |
| Semantic embedding model | Hugging Face cache (`~/.cache/huggingface`) | `all-MiniLM-L6-v2`, downloaded on first semantic run |
| Reference/lookup data | `data/reference/` | currently empty; reserved |
| Dev server logs (if started with redirection) | `data/uvicorn.log`, `data/vite.log` | only present when started via redirect |
| Knowledge-base version entries | SQLite table `knowledge_base_versions` | alias admission is auditable there |

Raw files are **immutable by design**: no code path in the repository writes to an
existing version directory; corrections always create a new `v{n}` folder.

---

## 8. How to stop the application

- **Backend (Terminal 1):** press `Ctrl+C` in the terminal running uvicorn.
- **Frontend (Terminal 2):** press `Ctrl+C` in the terminal running Vite.

Force-stop from any terminal if needed:

```powershell
netstat -ano | findstr ":8000 :5173"          # find PIDs in LISTENING state
taskkill /PID <backend_pid> /F
taskkill /PID <frontend_pid> /F
```

---

## 9. How to restart

Shortest procedure:

```powershell
# Terminal 1
cd backend
..\ .venv\Scripts\Activate.ps1    # -> ..\.venv\Scripts\Activate.ps1
python -m uvicorn app.main:app --reload --port 8000

# Terminal 2
cd frontend
npm run dev
```

The database and all raw files persist — restarting changes nothing about your
data; the UI will show the same datasets and completed stages as before.

---

## 10. Troubleshooting

Each entry: **Symptom / Cause / Fix**, all observed from this codebase.

### Port 8000 already in use
- **Symptom:** `OSError: [Errno 10048] error while attempting to bind ... address already in use` (backend) or an old API answering on 8000.
- **Cause:** a previous uvicorn instance is still listening (common after a crash with `--reload`).
- **Fix:**
  ```powershell
  netstat -ano | findstr :8000
  taskkill /PID <pid> /F
  ```

### Port 5173 already in use
- **Symptom:** Vite prints `Port 5173 is in use, trying another one...` and the UI is served on 5174.
- **Cause:** another Vite/dev server instance running.
- **Fix:** use the printed port, or kill the old process with `taskkill /PID <pid> /F`. Note CORS only allows 5173 — kill the stale instance rather than using 5174, or add the port to `allow_origins` in `backend/app/main.py`.

### Backend not starting — import errors
- **Symptom:** `ModuleNotFoundError: No module named 'fastapi'` (or `sqlalchemy`, `pandas`...).
- **Cause:** dependencies not installed or wrong interpreter.
- **Fix:** activate `.venv` (prompt shows `(.venv)`) and run `pip install -r requirements.txt` from `backend/`.

### Backend not starting — PowerShell script blocked
- **Symptom:** `...Activate.ps1 cannot be loaded because running scripts is disabled on this system`.
- **Cause:** PowerShell execution policy.
- **Fix:** `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, then retry.

### Frontend not starting — npm install failures
- **Symptom:** `npm run dev` fails with missing `vite` binary or peer-dependency errors.
- **Cause:** interrupted install or Node below 20.19.
- **Fix:** delete `frontend/node_modules` and `frontend/package-lock.json`, confirm `node -v` ≥ 20.19, then `npm install` again.

### Database errors
- **Symptom:** `sqlalchemy.exc.OperationalError: no such table: datasets` or similar.
- **Cause:** the DB file was created by an older schema before the new tables existed, and `create_all` does not alter existing tables.
- **Fix (dev environment):** stop the backend, delete `data/dq_assessment.db`, restart — it re-creates the schema and you re-upload datasets. (Destructive to stored metadata only; raw files in `data/raw/` remain.)

### CORS errors
- **Symptom:** browser console shows `blocked by CORS policy`; every request fails from the UI while `curl` to the API works.
- **Cause:** frontend served on an origin not in the allow-list (e.g. Vite fell back to 5174, or you opened `dist/index.html` via `file://`).
- **Fix:** run the frontend with `npm run dev` so it serves on 5173; do not open the built `dist/index.html` directly. If needed, add your origin to `allow_origins` in `backend/app/main.py`.

### API 404 on `/`
- **Symptom:** `{"detail":"Not Found"}` when opening `http://127.0.0.1:8000/`.
- **Cause:** intentional — the app defines no root route.
- **Fix:** none needed; use `/docs` (Swagger) or `/health`.

### API 404 for a dataset
- **Symptom:** `404 Dataset {id} not found` or `404 No version found for dataset {id}`.
- **Cause:** dataset deleted/never created on this database, or the backend is pointing at a different `data/` folder (e.g. started from a different working copy).
- **Fix:** check `GET /datasets` for actual IDs; make sure only one backend instance and one repo copy are in use.

### Blank frontend page
- **Symptom:** white page, nothing renders.
- **Cause (most common):** backend is not running — App still renders its shell but data calls fail; or a stale JS bundle.
- **Fix:** start the backend first, hard-refresh with `Ctrl+Shift+R`. Check the browser console: the UI reports connection errors inline ("Could not connect to the backend...") once the shell loads.

### Profiling page empty / profiling fails
- **Symptom:** "Profiling could not be loaded" with message `No version found for dataset N` or `Immutable raw file is missing for file ...`.
- **Cause:** dataset registered without a version, or the `data/raw/datasets/{id}/v{n}/` folder was moved/deleted manually.
- **Fix:** restore the raw folder or re-upload the dataset. Profiling reads only from immutable raw files — it never re-asks the browser for data.

### Upload failure — 400 with "Unsupported file format"
- **Symptom:** `Unsupported file format: .txt. Supported formats: .csv, .parquet, .xls, .xlsx`
- **Cause:** the extension is not in `SUPPORTED_EXTENSIONS` (`backend/app/services/validation.py`).
- **Fix:** convert to CSV/XLS/XLSX/Parquet.

### Upload failure — empty or oversized
- **Symptom:** `Uploaded file is empty.` or `File exceeds the maximum allowed size of 100 MB.`
- **Cause:** `validate_file_size` rejects 0-byte files and anything > `MAX_FILE_SIZE_MB` (100).
- **Fix:** check the file; split or compress very large datasets.

### Upload failure — 409 duplicates
- **Symptom:** `One or more uploaded files already exist in the registered file history.` with a duplicate-file list.
- **Cause:** SHA-256 file fingerprint matches an already-registered file.
- **Fix:** this is by design. Use the "Register N new files" flow (analysis step) to register only genuinely new files, or modify the file content.

### Upload failure — 422 unreadable table
- **Symptom:** `Could not read <file>: <pandas error>` or `No readable tables found in <file>`.
- **Cause:** corrupted file, or an Excel workbook with no readable sheet data.
- **Fix:** open the file in Excel/another tool to confirm it is valid; re-export if needed.

### Excel/XLS reading failure
- **Symptom:** `ImportError: Missing optional dependency 'xlrd'` (for `.xls`) or `'openpyxl'` (for `.xlsx`).
- **Cause:** engine packages not installed.
- **Fix:** `pip install openpyxl xlrd` (both are already listed in `requirements.txt`, so this usually means the venv was not used).

### CSV encoding failure
- **Symptom:** mojibake characters (é → Ã©) or, rarely, `UnicodeDecodeError`.
- **Cause:** CSV saved with a legacy Windows encoding.
- **Fix:** none usually required — `read_table` tries `utf-8`, then `cp1252`, then `latin-1` automatically. Prefer saving as UTF-8 for correct special characters.

### Semantic analysis fails or is very slow the first time
- **Symptom:** first `POST /datasets/{id}/semantic/analyze` takes ~10–60 s or shows a Hugging Face download in the log.
- **Cause:** `all-MiniLM-L6-v2` is being downloaded on first use.
- **Fix:** wait; subsequent runs reuse the cached model. No internet is needed after the first run.

### Model not found
- **Symptom:** `models/ranking/` has no JSON file / `GET /models` returns an empty list.
- **Cause:** offline training has never run — it requires ≥ 20 labeled human feedback examples covering both classes.
- **Fix:** this is expected behavior, not an error. Record semantic decisions in Stage 05 until the threshold is met, then use `POST /models/train`.

### Rules page shows "No rules recommended"
- **Symptom:** Stage 06 generates 0 rules.
- **Cause:** by design, rules come only from **human-approved** semantic mappings (identifier/email concepts with supporting profile evidence). Pending or rejected semantics generate nothing.
- **Fix:** approve the relevant identifier/email mappings in Stage 05, then click "Generate recommendations" again.

---

## 11. Build and automated test commands

Verified commands (run from the repo root):

```powershell
# Backend — complete test suite (89 tests, verified passing)
cd backend
..\.venv\Scripts\python.exe -m pytest -q

# Backend — unit groups (examples)
..\.venv\Scripts\python.exe -m pytest tests\test_profiling.py -q
..\.venv\Scripts\python.exe -m pytest tests\test_quality_services.py -q
..\.venv\Scripts\python.exe -m pytest tests\test_stage_services.py -q

# Backend — full pipeline integration test (upload → scores in one flow)
..\.venv\Scripts\python.exe -m pytest tests\test_end_to_end.py -q

# Frontend — type check + production build (verified passing)
cd ..\frontend
npm run build        # runs "tsc -b && vite build"

# Frontend — lint
npm run lint
```

Test-suite layout (all under `backend/tests/`):

| File | Tests | Covers |
|---|---|---|
| `test_upload.py` | 14 | ingestion validation, fingerprints, duplicate detection |
| `test_profiling.py` | 20 | all profiler statistics and edge cases |
| `test_stage_services.py` | 13 | semantic stage, execution, scoring, RCA, monitoring |
| `test_quality_services.py` | 13 | rule execution engine, scoring math, RCA language, remediation |
| `test_dataset_context.py` | 7 | context read/update/persistence |
| `test_upload.py` / `test_storage.py` | 14 / 5 | storage, filenames, immutability |
| `test_versioning.py`, `test_version_snapshot.py`, `test_version_proposal.py`, `test_version_confirmation.py` | 5 + 3 + 2 + 3 | versions, lineage, fingerprints |
| `test_dataset_matching.py` | 3 | duplicate file matching |
| `test_end_to_end.py` | 1 | full pipeline: upload → context → profile → semantic → rules → approval → execution → score → RCA → remediation → reassessment |

There is no separate frontend unit-test runner configured; `npm run build`
(including `tsc -b`) is the frontend verification gate.

# Intelligent Data Quality Assessment, Metric & Rule Recommendation, and Monitoring System

An end-to-end Data Quality platform: upload a dataset, profile it, understand
column semantics with ML, recommend DQ metrics and rules with human approval,
execute rules safely, score quality, analyze root causes, remediate safely
into new versions, monitor drift, and feed human decisions back into offline
model training.

```
Upload → Context → Version/Fingerprint → Profiling → Semantic Understanding →
Metric & Rule Recommendation → Validation → Human Approval → Rule Execution →
DQ Scoring → RCA → Remediation → Reassessment → Monitoring → Feedback →
Offline Learning
```

## Architecture

Modular monolith (FastAPI + SQLite + React/Vite). No microservices, queues or
cloud infrastructure.

```
                 React UI (Vite, TypeScript)
                          │
                  FastAPI backend
                          │
   ┌──────────────────────┼──────────────────────┐
   │ Dataset layer        │ Intelligence layer   │ Quality layer
   │ - ingestion          │ - semantic engine    │ - rule engine (DSL)
   │ - versioning         │ - knowledge base     │ - validation
   │ - fingerprinting     │ - feedback store     │ - execution
   │ - immutable storage  │ - offline learning   │ - scoring
   └──────────────────────┼──────────────────────┘
                          │
              RCA → Remediation → Reassessment
                          │
                     Monitoring (drift)
```

### Key architectural principles

1. **Observation ≠ Recommendation ≠ Business rule.** The profiler reports what
   the data looks like; the recommender suggests rules with evidence; only
   human approval turns a recommendation into a business rule that executes.
2. **Immutable versions.** Raw files are stored once per version under
   `data/raw/datasets/{dataset_id}/v{n}/` and never modified. Remediation
   creates a new version with a parent link.
3. **Safe rule execution.** Rules are a constrained JSON DSL interpreted by
   the engine — never `eval()` or generated code.
4. **Evidence, not fabrication.** Semantic confidence comes from actual
   similarity evidence; scores come from actual rule executions; RCA reports
   observed concentrations ("concentrated in", "associated with") without
   causal claims. "Insufficient evidence" and "insufficient data" are valid
   outputs.
5. **Human-in-the-loop.** Semantic mappings, rules, remediation, KB aliases
   and model promotion all require explicit human decisions, which are
   recorded as versioned feedback.

## Setup

### Backend

```bash
cd backend
python -m venv ../.venv          # or use an existing virtualenv
../.venv/Scripts/pip install -r requirements.txt   # Windows
# ../.venv/bin/pip install -r requirements.txt     # Linux/macOS

PYTHONPATH=. ../.venv/Scripts/python -m uvicorn app.main:app --port 8000
```

The database (`data/dq_assessment.db`) is created automatically on startup.
The semantic knowledge base (32 canonical concepts) is seeded on first use of
`GET /kb/concepts`, and concept embeddings are generated with
`all-MiniLM-L6-v2` (downloaded once from Hugging Face).

### Frontend

```bash
cd frontend
npm install
npm run dev        # http://localhost:5173
```

`npm run build` must pass for production use.

## Stages (UI navigation)

| # | Stage | Backend |
|---|-------|---------|
| 01 | Ingestion | `POST /datasets/upload`, `POST /datasets/upload/analyze` |
| 02 | Context | `GET/PUT /datasets/{id}/context` |
| 03 | Version & Fingerprint | `GET /datasets/{id}/context`, `POST /datasets/{id}/upload` (new version) |
| 04 | Profiling | `GET /datasets/{id}/profiling` (persisted in `stored_profiles`) |
| 05 | Semantic Understanding | `POST /datasets/{id}/semantic/analyze`, `GET /datasets/{id}/semantic`, `POST .../semantic/{pred}/decision` |
| 06-08 | Rules: recommend, validate, approve, execute | `POST /datasets/{id}/recommendations`, `POST /rules/{id}/validate`, `POST /rules/{id}/approve`, `POST /datasets/{id}/execute` |
| 08b | DQ Scoring & RCA | `POST/GET /datasets/{id}/scores`, `POST/GET /datasets/{id}/rca` |
| 09 | Remediation & Reassessment | `POST /datasets/{id}/remediation/propose`, `POST .../remediation/{rid}/decision` |
| 10 | Monitoring | `POST /datasets/{id}/monitoring` |
| 11 | Feedback & Offline Learning | `GET /datasets/{id}/semantic/feedback`, `POST /models/train`, `POST /models/promote` |

Stage completion is persisted in the backend (`GET /datasets/{id}/stages`);
the UI reads it so progress survives refresh.

## Rule DSL

Only these rule types are executable; anything else is rejected:

```json
{ "type": "completeness", "table": "customers", "column": "customer_id",
  "condition": "not_null", "treat_empty_string_as_null": true,
  "treat_whitespace_as_null": true }

{ "type": "uniqueness", "table": "orders", "columns": ["order_id"],
  "ignore_nulls": true }

{ "type": "validity", "table": "customers", "column": "email",
  "check": "email_syntax" }
```

`validity.check` supports `email_syntax`, `numeric_type`, `date_parseable`.
Business ranges, allowed-value lists and cross-column rules are deliberately
**not** inferred from data; they require business configuration (future work:
a configuration surface to add `range` / `allowed_values` rules safely).

Execution results include total/applicable/passed/failed rows, pass and
violation rates, and bounded failure examples (max 20 rows) — never a copy of
the dataset.

## Scoring

- Each metric (completeness, uniqueness, validity, …) aggregates its approved
  rules by applicable records, so rules covering different populations
  contribute proportionally.
- Overall score = mean of available metric scores (weighted mode available in
  `compute_dq_score`).
- N/A metrics and critical failures (score < 90) are listed explicitly; the
  UI surfaces critical failures so the overall score cannot hide them.

## RCA language

RCA output uses observational language ("concentrated in", "observed
alongside") with an explicit note that percentages describe failed rows only
and do not establish causality.

## Monitoring / drift

- Baseline is explicit (default: previous version of the same dataset).
- Numeric columns: two-sample Kolmogorov–Smirnov statistic (threshold 0.15).
- Categorical columns: Population Stability Index (threshold 0.2).
- Schema: schema-fingerprint comparison.
- Columns with fewer than 30 non-null values in either version are reported
  as `insufficient_sample` rather than producing misleading alerts.

## Offline learning

- Training labels are **human decisions only** (approved / edited / rejected
  semantic mappings). Model predictions are never used as labels.
- Training requires ≥ 20 labeled examples covering both classes; otherwise it
  reports `insufficient_data` without training.
- Trained reranker candidates (logistic regression over the semantic evidence
  features) are held-out-evaluated, stored under `models/ranking/`, versioned
  in the model registry, and only activated via explicit promotion.

## Knowledge base admission

New aliases are proposed (`POST /kb/aliases/propose`), accumulate evidence,
and become permanent only after explicit human admission
(`POST /kb/aliases/{id}/admission`). All changes are auditable rows in
`knowledge_base_versions`.

## Testing

```bash
cd backend
../.venv/Scripts/python -m pytest tests/ -q
```

89 tests cover ingestion, versioning, storage, dataset matching, context,
profiling, the execution engine, validation, scoring, semantic stage,
offline learning, monitoring, remediation and a complete end-to-end
pipeline test (`tests/test_end_to_end.py`).

## Known limitations

- Single-user, no authentication (internship scope).
- Rule types cover completeness / uniqueness / basic validity; referential
  integrity, consistency and business-configured ranges are architected
  (metric dimension model) but not yet exposed end-to-end.
- Candidate PK/FK relationship detection (containment evidence, human-approved
  RI rules) is designed in the spec but not yet implemented.
- Excel output of remediation loses multi-sheet structure (one file per table
  is written as a single-sheet workbook); CSV round-trips exactly.
- Drift monitoring compares the first file of a version; multi-file versions
  compare per first table only.
- Offline model training currently reranks semantic candidates using stored
  evidence features; integrating the promoted weights into live retrieval is
  the next step.

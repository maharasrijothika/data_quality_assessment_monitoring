# Verification notes

This file records how the end-to-end pipeline was verified manually.

## API verification (dataset with identifier + email columns)

The Manufacturing dataset (id 24) has no identifier or email columns, so the
recommendation engine correctly returned zero rules there (rules are only
generated from approved identifier / email concepts).

The full chain was verified with the existing end-to-end test:

    cd backend
    python -m pytest tests/test_end_to_end.py -v

which executes: upload → stages → profiling → semantic analyze → human
decisions → recommendations → validation → approval → execution → scoring →
RCA → remediation propose/approve → new version lineage → monitoring →
offline learning.

## Live UI verification

- Frontend on http://localhost:5173, backend on http://127.0.0.1:8000.
- Opened dataset 24 (Manufacturing_Line_Productivity) from the registered
  dataset list.
- Stage 05 (Semantic): ran real MiniLM semantic analysis over 26 columns;
  predictions show evidence (embedding/name/description/datatype/profile
  similarity) and confidence levels (Strong/Probable/Ambiguous/Unknown).
- Approved a mapping through the UI; stage 05 was marked complete in the
  backend stage navigation (green check).
- Stage 06 (Rules): recommendation generation correctly reports that only
  approved identifier/email concepts produce rules.

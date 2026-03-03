# Engineering Roadmap

This project started as a notebook-first workflow. The first engineering pass now adds package structure, config, and runnable scripts.

## Phase 1 (Completed)

- Package scaffold under `src/crisispulse`
- Data preprocessing module extracted from notebook
- Embedding module extracted from notebook
- Topic-model metric and factory modules extracted from notebook
- Drift detection module extracted from notebook
- Script entry points for prepare/embed/tune/monitor

## Phase 2 (Recommended Next)

- Move remaining model-evaluation and merge logic from `main.ipynb` into:
  - `src/crisispulse/topic_model/evaluate.py` (completed)
  - `src/crisispulse/topic_model/merge.py` (completed)
- Add `scripts/run_eval.py` and `scripts/run_merge.py` (completed)
- Replace notebook direct function definitions with module imports (in progress)

## Phase 3 (Production Hardening)

- Add logging across scripts (`logging` module + structured output)
- Add unit tests for metrics and drift utilities
- Add integration smoke tests with tiny fixture datasets
- Add CI workflow for lint + tests
- Optional: add DVC for large artifacts

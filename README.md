# CrisisPulse - Incremental BERTopic Brand Crisis Detection

CrisisPulse is a script-first, reproducible NLP pipeline for tracking evolving risk signals in brand-related social data. The project combines GPU-accelerated topic modeling with staged model fitting (baseline, delta, merge), semantic drift monitoring, and optional local LLM-based topic labeling.

The workflow is designed for production-style iteration rather than notebook-only experimentation: each phase is modularized under `src/`, executable via `scripts/`, and validated through lint/test/CI gates. Outputs include topic summaries, topic-change diagnostics, daily drift alerts, and reporting artifacts that support both technical evaluation and stakeholder communication.

## What Changed

- Added a proper Python package under `src/crisispulse/`
- Added pipeline scripts under `scripts/`
- Added central config in `configs/base.yaml`
- Added packaging metadata in `pyproject.toml`
- Added repository hygiene via `.gitignore`

## Repository Layout

```text
crisispulse/
├─ configs/
│  └─ base.yaml
├─ scripts/
│  ├─ run_prepare.py
│  ├─ run_embed.py
│  ├─ run_tune.py
│  ├─ run_eval.py
│  ├─ run_merge.py
│  ├─ run_topic_summary.py
│  ├─ run_topic_change.py
│  ├─ run_prepare_monitor_inputs.py
│  ├─ run_monitor.py
│  ├─ run_monitor_report.py
│  └─ run_label_topics.py
├─ src/
│  └─ crisispulse/
│     ├─ config.py
│     ├─ data/preprocessing.py
│     ├─ embeddings/e5.py
│     ├─ topic_model/{metrics.py,objective.py,evaluate.py,merge.py,analysis.py,labeling.py}
│     └─ monitoring/{drift.py,report.py}
└─ environment.yaml
```

## Environment Setup (WSL2 + GPU)

Use the existing Conda environment file as the source of truth for RAPIDS + CUDA compatibility:

```bash
conda env create -f environment.yaml
conda activate rapids
```

Note: `environment.yaml` is the recommended setup path.

## Install Package (Editable)

From project root:

```bash
pip install -e .
```

For dev tooling:

```bash
pip install -e .[dev]
```

## Run Pipeline Stages

All scripts default to `configs/base.yaml`.

### 1) Prepare cleaned/split datasets

```bash
python scripts/run_prepare.py --config configs/base.yaml
```

### 2) Generate embeddings

```bash
python scripts/run_embed.py --config configs/base.yaml
```

### 3) Tune baseline BERTopic model

```bash
python scripts/run_tune.py --config configs/base.yaml --trials 50 --study-name baseline_tuning_v2
```

Notes:

- Tuning is resumable via SQLite storage under `artifacts/baseline_model/`.
- If interrupted, rerun the same command and it continues from the existing study.

### 4) Evaluate baseline model

```bash
python scripts/run_eval.py --config configs/base.yaml
```

### 5) Build merged models

```bash
python scripts/run_merge.py --config configs/base.yaml --tau 0.95 --delta-trials 25 --test-fit-ratio 0.70
```

`run_merge.py` now trains the second delta model on the chronological `TEST_fit` slice only
to avoid leakage from future monitor-period data.

### 6) Run daily monitoring

First, prepare TEST monitor inputs (same methodology as notebook: chronological split + merged model topic assignment):

```bash
python scripts/run_prepare_monitor_inputs.py \
  --config configs/base.yaml \
  --model-path artifacts/merge_model/models/merged_final_model \
  --test-csv data/data_combined_clean_test.csv \
  --test-embeddings artifacts/embeddings/embeddings_test.npy \
  --split-ratio 0.70 \
  --out-dir artifacts/online_model
```

Then run monitoring:

```bash
python scripts/run_monitor.py \
  --config configs/base.yaml \
  --monitor-csv artifacts/online_model/all_non_out_monitor.csv \
  --reference-csv artifacts/merge_model/summaries/merged_final/all_docs_with_topics.csv \
  --monitor-embeddings artifacts/online_model/embeddings_test_monitor_non_out.npy
```

### 7) Generate topic summaries (before/after snapshots)

```bash
python scripts/run_topic_summary.py \
  --config configs/base.yaml \
  --model-path artifacts/baseline_model/model_core \
  --docs-csv data/data_combined_clean_test.csv \
  --embeddings-npy artifacts/embeddings/embeddings_test.npy \
  --out-dir artifacts/merge_model/summaries/baseline_test

python scripts/run_topic_summary.py \
  --config configs/base.yaml \
  --model-path artifacts/merge_model/models/merged_final_model \
  --docs-csv data/data_combined_clean_test.csv \
  --embeddings-npy artifacts/embeddings/embeddings_test.npy \
  --out-dir artifacts/merge_model/summaries/merged_final
```

### 8) Analyze topic change between two snapshots

```bash
python scripts/run_topic_change.py \
  --before-csv artifacts/merge_model/summaries/baseline_test/all_docs_with_topics.csv \
  --after-csv artifacts/merge_model/summaries/merged_final/all_docs_with_topics.csv \
  --topic-info-csv artifacts/merge_model/topic_info_snapshots/merged_final.csv \
  --out-dir artifacts/merge_model/topic_change/final_vs_baseline \
  --focus-sentiment negative \
  --top-n 10
```

### 9) Build monitoring report artifacts

```bash
python scripts/run_monitor_report.py \
  --monitor-results-pkl artifacts/online_model/monitor_results.pkl \
  --reference-csv artifacts/merge_model/summaries/merged_final/all_docs_with_topics.csv \
  --topic-info-csv artifacts/merge_model/topic_info_snapshots/merged_final.csv \
  --out-dir artifacts/online_model/report
```

### 10) Optional: Add local LLM labels post-hoc

Run this only after topic models are finalized, to avoid repeated LLM calls during tuning.

```bash
python scripts/run_label_topics.py \
  --topic-info-csv artifacts/merge_model/topic_info_snapshots/merged_final.csv \
  --model-path /path/to/your-model.gguf \
  --output-csv artifacts/merge_model/topic_info_snapshots/merged_final_labeled.csv
```

This step uses cache-based labeling to avoid recomputing labels for unchanged topics.

## Notebook Strategy

The repository is script-first for reproducibility. If you use notebooks for exploration,
keep them thin and import reusable logic from `src/crisispulse/`.

## Data and Artifact Policy

By default, generated artifacts and model binaries are ignored by Git:

- `artifacts/`
- `models/`
- large model files (`*.gguf`)

Commit only reproducible code/config/docs, not generated outputs.

## Reproducibility

For a complete script-by-script rerun recipe, see `docs/reproducibility.md`.

For release gating and publish readiness, see `docs/release_checklist_v0.1.0.md`.

# GitHub Repo Description and Abstract

## Repository Description (short)

Incremental BERTopic pipeline for brand crisis detection with RAPIDS, PyTorch embeddings, merge-aware topic evolution, and drift monitoring.

## Repository Abstract (README/GitHub About)

Incremental BERTopic Brand Crisis Detection is a script-first, reproducible NLP pipeline for tracking evolving risk signals in brand-related social data. The project combines GPU-accelerated topic modeling with staged model fitting (baseline, delta, merge), semantic drift monitoring, and optional local LLM-based topic labeling.

The workflow is designed for production-style iteration rather than notebook-only experimentation: each phase is modularized under `src/`, executable via `scripts/`, and validated through lint/test/CI gates. Outputs include topic summaries, topic-change diagnostics, daily drift alerts, and reporting artifacts that support both technical evaluation and stakeholder communication.

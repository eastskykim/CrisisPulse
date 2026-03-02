# v0.1.0 Release Checklist

## Quality Gates

- [ ] `ruff check .`
- [ ] `black --check .`
- [ ] `pytest -q`

## Pipeline Smoke Validation

- [ ] `python scripts/run_prepare.py --config configs/base.yaml`
- [ ] `python scripts/run_embed.py --config configs/base.yaml`
- [ ] `python scripts/run_tune.py --config configs/base.yaml --trials 1 --study-name release_smoke_v010`
- [ ] `python scripts/run_eval.py --config configs/base.yaml`
- [ ] `python scripts/run_merge.py --config configs/base.yaml --tau 0.95 --delta-trials 1 --test-fit-ratio 0.70`
- [ ] `python scripts/run_prepare_monitor_inputs.py --config configs/base.yaml --out-dir artifacts/online_model`
- [ ] `python scripts/run_monitor.py --config configs/base.yaml --monitor-csv artifacts/online_model/all_non_out_monitor.csv --reference-csv artifacts/merge_model/summaries/merged_final/all_docs_with_topics.csv --monitor-embeddings artifacts/online_model/embeddings_test_monitor_non_out.npy --out artifacts/online_model/monitor_results.pkl`
- [ ] `python scripts/run_monitor_report.py --monitor-results-pkl artifacts/online_model/monitor_results.pkl --reference-csv artifacts/merge_model/summaries/merged_final/all_docs_with_topics.csv --topic-info-csv artifacts/merge_model/topic_info_snapshots/merged_final.csv --out-dir artifacts/online_model/report`

## Optional Labeling Validation

- [ ] `python scripts/run_label_topics.py --topic-info-csv artifacts/merge_model/topic_info_snapshots/merged_final.csv --model-path models/llm/zephyr-7b-alpha.Q4_K_M.gguf --output-csv artifacts/merge_model/topic_info_snapshots/merged_final_labeled.csv --max-topics 20`

## Publish Hygiene

- [ ] Ensure no generated outputs are staged (`artifacts/`, `models/`, large files)
- [ ] Confirm project name and package metadata are generalized
- [ ] Confirm no sponsor-identifying strings remain in tracked files
- [ ] Confirm `LICENSE` and CI workflow are present
- [ ] Tag release as `v0.1.0`

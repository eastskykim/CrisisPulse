# Reproducibility Guide

This project is organized as a script-first pipeline so runs can be reproduced without notebooks.

## 1) Environment

```bash
conda env create -f environment.yaml
conda activate rapids
pip install -e .[dev]
```

## 2) End-to-End Command Order

```bash
python scripts/run_prepare.py --config configs/base.yaml
python scripts/run_embed.py --config configs/base.yaml
python scripts/run_tune.py --config configs/base.yaml --trials 50 --study-name baseline_tuning_v2
python scripts/run_eval.py --config configs/base.yaml
python scripts/run_merge.py --config configs/base.yaml --tau 0.95 --delta-trials 25 --test-fit-ratio 0.70
python scripts/run_prepare_monitor_inputs.py --config configs/base.yaml --out-dir artifacts/online_model
python scripts/run_monitor.py --config configs/base.yaml --monitor-csv artifacts/online_model/all_non_out_monitor.csv --reference-csv artifacts/merge_model/summaries/merged_final/all_docs_with_topics.csv --monitor-embeddings artifacts/online_model/embeddings_test_monitor_non_out.npy --out artifacts/online_model/monitor_results.pkl
python scripts/run_monitor_report.py --monitor-results-pkl artifacts/online_model/monitor_results.pkl --reference-csv artifacts/merge_model/summaries/merged_final/all_docs_with_topics.csv --topic-info-csv artifacts/merge_model/topic_info_snapshots/merged_final.csv --out-dir artifacts/online_model/report
```

## 3) Optional Post-hoc LLM Labels

```bash
python scripts/run_label_topics.py --topic-info-csv artifacts/merge_model/topic_info_snapshots/merged_final.csv --model-path models/llm/zephyr-7b-alpha.Q4_K_M.gguf --output-csv artifacts/merge_model/topic_info_snapshots/merged_final_labeled.csv
```

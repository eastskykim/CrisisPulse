from __future__ import annotations

import argparse
import pickle

import numpy as np
import pandas as pd

from crisispulse.config import load_config
from crisispulse.monitoring.drift import (
    compute_negative_prevalence,
    compute_sentiment_distribution,
    compute_topic_distribution,
    run_daily_monitoring,
    slice_by_day,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run daily semantic drift monitoring.")
    parser.add_argument("--config", default="configs/base.yaml", help="Path to YAML config")
    parser.add_argument(
        "--monitor-csv",
        required=True,
        help="CSV file with monitor-period records and topic assignments",
    )
    parser.add_argument(
        "--reference-csv",
        required=True,
        help="CSV file used to build reference distributions",
    )
    parser.add_argument(
        "--monitor-embeddings",
        required=True,
        help="NPY embeddings for monitor period rows",
    )
    parser.add_argument(
        "--out",
        default="artifacts/online_model/monitor_results.pkl",
        help="Output pickle file",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    monitor_df = pd.read_csv(args.monitor_csv)
    reference_df = pd.read_csv(args.reference_csv)
    monitor_embeddings = np.load(args.monitor_embeddings)

    batches = slice_by_day(monitor_df)
    emb_by_day = {}
    for date_str, df_day in batches.items():
        idx = df_day.index.to_numpy() - df_day.index.min()
        emb_by_day[date_str] = monitor_embeddings[idx]

    ref_topic_dist = compute_topic_distribution(reference_df["topic_id"].tolist())
    ref_sent_dist = compute_sentiment_distribution(reference_df)
    ref_neg_topic_dist = compute_topic_distribution(
        reference_df[reference_df["sentiment"] == "negative"]["topic_id"].tolist()
    )
    ref_neg_prev = compute_negative_prevalence(reference_df)
    ref_neg_std = (reference_df["sentiment"] == "negative").std()

    results = run_daily_monitoring(
        daily_batches=batches,
        daily_embeddings=emb_by_day,
        reference_topic_dist=ref_topic_dist,
        reference_sentiment_dist=ref_sent_dist,
        reference_neg_topic_dist=ref_neg_topic_dist,
        reference_neg_prevalence=ref_neg_prev,
        reference_neg_std=ref_neg_std,
    )

    out_path = cfg.paths.project_root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        pickle.dump(results, f)
    print(f"Saved monitor results -> {out_path}")


if __name__ == "__main__":
    main()

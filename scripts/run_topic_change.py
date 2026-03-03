from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from crisispulse.topic_model.analysis import (
    compare_topic_distributions,
    select_top_new_topics,
    select_top_topic_changes,
    summarize_topic_stability_filtered,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare topic distributions before/after merge.")
    parser.add_argument("--before-csv", required=True, help="CSV of non-outlier docs before merge")
    parser.add_argument("--after-csv", required=True, help="CSV of non-outlier docs after merge")
    parser.add_argument("--topic-info-csv", required=True, help="topic_info CSV for after model")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument("--growth-threshold", type=float, default=0.10)
    parser.add_argument(
        "--focus-sentiment",
        default="",
        help="Optional filter: negative|neutral|positive",
    )
    parser.add_argument("--top-n", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    before_df = pd.read_csv(args.before_csv)
    after_df = pd.read_csv(args.after_csv)
    topic_info_df = pd.read_csv(args.topic_info_csv)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    topic_change_df = compare_topic_distributions(before_df, after_df)
    topic_change_df.to_csv(out / "topic_change.csv", index=False)

    focus = args.focus_sentiment.strip() or None
    metrics_df, persisting_df, new_df = summarize_topic_stability_filtered(
        topic_change_df,
        growth_threshold=args.growth_threshold,
        focus_sentiment=focus,
    )
    metrics_df.to_csv(out / "stability_metrics.csv", index=False)
    persisting_df.to_csv(out / "persisting_topics.csv", index=False)
    new_df.to_csv(out / "new_topics.csv", index=False)

    top_changed_df = select_top_topic_changes(persisting_df, topic_info_df, top_n=args.top_n)
    top_changed_df.to_csv(out / "top_changed_topics.csv", index=False)

    top_new_df = select_top_new_topics(new_df, topic_info_df, top_n=args.top_n)
    top_new_df.to_csv(out / "top_new_topics.csv", index=False)

    print("Topic change analysis complete.")
    print(f"  topic_change_rows={topic_change_df.shape[0]}")
    print(f"  persisting_rows={persisting_df.shape[0]}")
    print(f"  new_rows={new_df.shape[0]}")


if __name__ == "__main__":
    main()

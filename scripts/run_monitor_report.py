from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import pandas as pd

from crisispulse.monitoring.drift import (
    compute_topic_distribution,
)
from crisispulse.monitoring.report import (
    build_hybrid_alert_table,
    plot_jsd_trajectory,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create monitoring report artifacts from drift results."
    )
    parser.add_argument(
        "--monitor-results-pkl", required=True, help="Pickle from scripts/run_monitor.py"
    )
    parser.add_argument(
        "--reference-csv",
        required=True,
        help="CSV used to build reference topic distribution (must contain topic_id)",
    )
    parser.add_argument(
        "--topic-info-csv",
        default="",
        help="Optional topic_info CSV to include representation text in top deltas",
    )
    parser.add_argument(
        "--out-dir", default="artifacts/online_model/report", help="Output directory"
    )
    parser.add_argument("--alert-floor", type=float, default=0.20)
    parser.add_argument("--window-size", type=int, default=5)
    parser.add_argument("--neg-z-threshold", type=float, default=2.5)
    parser.add_argument("--min-neg-share", type=float, default=0.25)
    parser.add_argument("--top-n", type=int, default=5)
    return parser.parse_args()


def _collect_alert_topic_deltas(
    results: list[dict],
    alerts_df: pd.DataFrame,
    ref_topic_dist: pd.Series,
    topic_info_df: pd.DataFrame | None,
    top_n: int,
    min_neg_share: float,
) -> pd.DataFrame:
    topic_label_map: dict[int, str] = {}
    if topic_info_df is not None:
        source_col = "Representation" if "Representation" in topic_info_df.columns else "Name"
        if source_col in topic_info_df.columns:
            topic_label_map = {
                int(r["Topic"]): str(r[source_col])
                for _, r in topic_info_df.iterrows()
                if int(r["Topic"]) >= 0
            }

    rows: list[dict] = []
    alert_dates = set(pd.to_datetime(alerts_df[alerts_df["hybrid_alert"]]["date"]).dt.date)
    for result in results:
        if result["date"] not in alert_dates:
            continue

        day_df = result["df_with_topics"]
        delta = result["topic_dist"].subtract(ref_topic_dist, fill_value=0.0)
        delta = delta.drop(index=-1, errors="ignore")
        delta = delta.reindex(delta.abs().sort_values(ascending=False).index)

        added = 0
        for topic_id, diff in delta.items():
            topic_df = day_df[day_df["topic_id"] == topic_id]
            if topic_df.empty:
                continue
            neg_share = float((topic_df["sentiment"] == "negative").mean())
            if neg_share < min_neg_share:
                continue

            rows.append(
                {
                    "date": result["date"],
                    "topic_id": int(topic_id),
                    "delta": float(diff),
                    "ref_freq": float(ref_topic_dist.get(topic_id, 0.0)),
                    "curr_freq": float(result["topic_dist"].get(topic_id, 0.0)),
                    "neg_share": neg_share,
                    "label": topic_label_map.get(int(topic_id), ""),
                }
            )
            added += 1
            if added >= top_n:
                break

    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with Path(args.monitor_results_pkl).open("rb") as f:
        results = pickle.load(f)

    reference_df = pd.read_csv(args.reference_csv)
    ref_topic_dist = compute_topic_distribution(reference_df["topic_id"].tolist())

    alerts_df = build_hybrid_alert_table(
        results=results,
        alert_floor=args.alert_floor,
        window_size=args.window_size,
        neg_z_threshold=args.neg_z_threshold,
    )
    alerts_df.to_csv(out_dir / "hybrid_alerts.csv", index=False)

    plot_jsd_trajectory(
        results=results,
        output_path=out_dir / "jsd_trajectory.png",
        alert_floor=max(0.10, args.alert_floor),
        window_size=max(1, args.window_size),
    )

    topic_info_df = pd.read_csv(args.topic_info_csv) if args.topic_info_csv.strip() else None
    top_deltas_df = _collect_alert_topic_deltas(
        results=results,
        alerts_df=alerts_df,
        ref_topic_dist=ref_topic_dist,
        topic_info_df=topic_info_df,
        top_n=args.top_n,
        min_neg_share=args.min_neg_share,
    )
    top_deltas_df.to_csv(out_dir / "top_topic_deltas_alerts.csv", index=False)

    print("Monitor report complete.")
    print(f"  alerts={int(alerts_df['hybrid_alert'].sum())}")
    print(f"  alert_topic_rows={top_deltas_df.shape[0]}")
    print(f"  output_dir={out_dir}")


if __name__ == "__main__":
    main()

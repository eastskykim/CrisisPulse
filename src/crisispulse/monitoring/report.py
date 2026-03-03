from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def build_hybrid_alert_table(
    results: list[dict],
    alert_floor: float = 0.20,
    window_size: int = 5,
    neg_z_threshold: float = 2.5,
) -> pd.DataFrame:
    all_jsd = [r["jsd_topic"] for r in results]
    rows: list[dict] = []

    for i, result in enumerate(results):
        window = all_jsd[max(0, i - window_size) : i]
        jsd_threshold = (
            alert_floor if len(window) == 0 else max(alert_floor, np.percentile(window, 95))
        )

        neg_z = result.get("neg_zscore", None)
        neg_alert = (neg_z is not None) and (neg_z > neg_z_threshold)
        hybrid_alert = (result["jsd_topic"] > jsd_threshold) and neg_alert

        rows.append(
            {
                "date": result["date"],
                "jsd_topic": float(result["jsd_topic"]),
                "jsd_threshold": float(jsd_threshold),
                "jsd_sentiment": float(result["jsd_sentiment"]),
                "jsd_neg_topic": result.get("jsd_neg_topic", np.nan),
                "neg_prevalence": result.get("neg_prevalence", np.nan),
                "neg_zscore": neg_z if neg_z is not None else np.nan,
                "hybrid_alert": bool(hybrid_alert),
            }
        )

    return pd.DataFrame(rows)


def plot_jsd_trajectory(
    results: list[dict],
    output_path: Path,
    alert_floor: float = 0.10,
    window_size: int = 3,
    min_neg_samples: int = 10,
) -> None:
    dates = [r["date"] for r in results]
    jsd_topic = [r["jsd_topic"] for r in results]
    jsd_sent = [r["jsd_sentiment"] for r in results]
    jsd_neg = [r.get("jsd_neg_topic", None) for r in results]
    neg_counts = [
        (
            (r["df_with_topics"]["sentiment"] == "negative").sum()
            if r.get("df_with_topics") is not None
            else 0
        )
        for r in results
    ]

    thresholds = []
    for i in range(len(jsd_topic)):
        window = jsd_topic[max(0, i - window_size) : i]
        threshold = alert_floor if len(window) == 0 else max(alert_floor, np.percentile(window, 95))
        thresholds.append(threshold)

    plt.figure(figsize=(11, 4))
    plt.plot(dates, jsd_topic, marker="o", label="Topic JSD")
    plt.plot(dates, jsd_sent, marker="o", label="Sentiment JSD")
    plt.plot(dates, thresholds, linestyle="--", color="gray", label="Dynamic Alert Threshold")
    plt.plot(dates, jsd_neg, linestyle="-", color="tab:red", alpha=0.3, label="_nolegend_")

    for i, (d, jsd, count) in enumerate(zip(dates, jsd_neg, neg_counts)):
        if jsd is None:
            continue
        label = "Neg-topic JSD" if i == 0 else "_nolegend_"
        if count < min_neg_samples:
            plt.plot(
                d,
                jsd,
                marker="o",
                linestyle="None",
                markersize=6,
                markerfacecolor="none",
                markeredgecolor="tab:red",
                label=label,
            )
        else:
            plt.plot(
                d, jsd, marker="o", linestyle="None", markersize=6, color="tab:red", label=label
            )

    plt.title("Daily Jensen-Shannon Divergence (vs Reference)")
    plt.ylabel("JSD")
    plt.xticks(rotation=45)
    plt.legend(loc="upper left")
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()

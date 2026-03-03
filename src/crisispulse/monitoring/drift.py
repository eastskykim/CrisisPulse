from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from tqdm import tqdm


def slice_by_day(df: pd.DataFrame, timestamp_col: str = "created_time") -> dict[str, pd.DataFrame]:
    out = df.copy()
    out[timestamp_col] = pd.to_datetime(out[timestamp_col])
    out["date"] = out[timestamp_col].dt.date
    grouped = out.groupby("date")
    return {str(date): group.drop(columns=["date"]) for date, group in grouped}


def compute_sentiment_distribution(df: pd.DataFrame) -> pd.Series:
    return df["sentiment"].value_counts(normalize=True)


def compute_topic_distribution(topic_ids: list[int] | pd.Series) -> pd.Series:
    topics = pd.Series(topic_ids)
    return topics[topics >= 0].value_counts(normalize=True)


def compute_negative_prevalence(df: pd.DataFrame) -> float:
    return float((df["sentiment"] == "negative").mean())


def jsd_between(p: pd.Series, q: pd.Series) -> float:
    all_keys = p.index.union(q.index)
    p_full = p.reindex(all_keys, fill_value=0)
    q_full = q.reindex(all_keys, fill_value=0)
    return float(jensenshannon(p_full, q_full) ** 2)


def detect_drift(
    batch_df: pd.DataFrame,
    batch_emb: np.ndarray,
    reference_topic_dist: pd.Series,
    reference_sentiment_dist: pd.Series,
    reference_neg_topic_dist: pd.Series | None = None,
    reference_neg_prevalence: float | None = None,
    reference_neg_std: float | None = None,
    alert_threshold_jsd: float = 0.05,
) -> dict:
    df_result = batch_df.copy()
    topics_final = df_result["topic_id"].to_list()

    topic_dist = compute_topic_distribution(topics_final)
    sentiment_dist = compute_sentiment_distribution(df_result[df_result["topic_id"] >= 0])
    neg_prevalence = compute_negative_prevalence(df_result)

    jsd_topic = jsd_between(topic_dist, reference_topic_dist)
    jsd_sent = jsd_between(sentiment_dist, reference_sentiment_dist)

    jsd_neg_topic = None
    if reference_neg_topic_dist is not None:
        neg_topics_today = df_result[df_result["sentiment"] == "negative"]["topic_id"]
        neg_topic_dist_today = compute_topic_distribution(neg_topics_today)
        jsd_neg_topic = jsd_between(neg_topic_dist_today, reference_neg_topic_dist)

    neg_z = None
    if reference_neg_prevalence is not None and reference_neg_std is not None:
        neg_z = (neg_prevalence - reference_neg_prevalence) / reference_neg_std

    alerts: list[str] = []
    if jsd_topic > alert_threshold_jsd:
        alerts.append(f"High topic divergence (JSD={jsd_topic:.3f})")
    if jsd_sent > alert_threshold_jsd:
        alerts.append(f"High sentiment divergence (JSD={jsd_sent:.3f})")
    if jsd_neg_topic is not None and jsd_neg_topic > alert_threshold_jsd:
        alerts.append(f"High negative-topic divergence (JSD={jsd_neg_topic:.3f})")

    return {
        "date": df_result["created_time"].min().date(),
        "jsd_topic": jsd_topic,
        "jsd_sentiment": jsd_sent,
        "jsd_neg_topic": jsd_neg_topic,
        "neg_prevalence": neg_prevalence,
        "neg_zscore": neg_z,
        "alerts": alerts,
        "topic_dist": topic_dist,
        "sentiment_dist": sentiment_dist,
        "df_with_topics": df_result,
        "topic_ids": topics_final,
        "embeddings": batch_emb,
    }


def run_daily_monitoring(
    daily_batches: dict[str, pd.DataFrame],
    daily_embeddings: dict[str, np.ndarray],
    reference_topic_dist: pd.Series,
    reference_sentiment_dist: pd.Series,
    reference_neg_topic_dist: pd.Series | None = None,
    reference_neg_prevalence: float | None = None,
    reference_neg_std: float | None = None,
) -> list[dict]:
    results: list[dict] = []
    for date_str in tqdm(sorted(daily_batches)):
        df_day = daily_batches[date_str]
        emb_day = daily_embeddings[date_str]
        result = detect_drift(
            batch_df=df_day,
            batch_emb=emb_day,
            reference_topic_dist=reference_topic_dist,
            reference_sentiment_dist=reference_sentiment_dist,
            reference_neg_topic_dist=reference_neg_topic_dist,
            reference_neg_prevalence=reference_neg_prevalence,
            reference_neg_std=reference_neg_std,
        )
        results.append(result)
    return results

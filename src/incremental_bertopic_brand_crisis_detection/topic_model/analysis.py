from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_DOC_COLS = ["sprout_guid", "created_time", "sentiment", "deidentified_text", "brand"]


def summarize_topics_by_index(
    model,
    topics: np.ndarray,
    combined_clean_df: pd.DataFrame,
    emb: np.ndarray,
    artifacts_dir: Path,
    top_n_words: int = 10,
    min_docs_per_topic: int = 10,
    top_k_reps: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    topics = np.asarray(topics)
    if len(combined_clean_df) != len(emb) or len(combined_clean_df) != len(topics):
        raise ValueError("Length mismatch between input dataframe, embeddings, and topics")

    missing = [c for c in REQUIRED_DOC_COLS if c not in combined_clean_df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = combined_clean_df.reset_index(drop=True).copy()
    df["topic_id"] = topics

    df_in = df[df["topic_id"] >= 0].copy()
    size_by_topic = df_in.groupby("topic_id").size().rename("n_docs").to_frame()
    keep_ids = size_by_topic.index[size_by_topic["n_docs"] >= min_docs_per_topic].tolist()
    df_kept = df_in[df_in["topic_id"].isin(keep_ids)].copy()

    sent_pivot = (
        df_kept.groupby("topic_id")["sentiment"]
        .value_counts(normalize=True)
        .unstack(fill_value=0.0)
        .reset_index()
    )
    for col in ["negative", "neutral", "positive"]:
        if col not in sent_pivot.columns:
            sent_pivot[col] = 0.0

    brand_pivot = (
        df_kept.groupby("topic_id")["brand"]
        .value_counts(normalize=True)
        .unstack(fill_value=0.0)
        .reset_index()
    )
    brand_cols = [c for c in brand_pivot.columns if c != "topic_id"]
    brand_pivot = brand_pivot.rename(columns={c: f"brand_{c}" for c in brand_cols})

    def topic_top_words(topic_id: int, k: int = top_n_words) -> str:
        pairs = model.get_topic(int(topic_id)) or []
        return ", ".join([w for (w, _) in pairs[:k]]) if pairs else ""

    sent_pivot["top_words"] = sent_pivot["topic_id"].apply(
        lambda t: topic_top_words(t, k=top_n_words)
    )

    sent_pivot = sent_pivot.merge(size_by_topic, left_on="topic_id", right_index=True, how="left")
    total_non_out = float((topics >= 0).sum())
    sent_pivot["share_of_non_outlier_docs"] = sent_pivot["n_docs"] / max(1.0, total_non_out)
    sent_pivot = sent_pivot.merge(brand_pivot, on="topic_id", how="left")

    emb_df = pd.DataFrame(emb).assign(topic_id=topics)
    centroids = emb_df[emb_df["topic_id"] >= 0].groupby("topic_id").mean()

    def cosine_distances_to(vec: np.ndarray, mat: np.ndarray) -> np.ndarray:
        vec_norm = np.linalg.norm(vec)
        mat_norm = np.linalg.norm(mat, axis=1)
        denom = vec_norm * mat_norm
        denom[denom == 0] = 1e-12
        sims = (mat @ vec) / denom
        return 1.0 - sims

    rep_rows: list[dict] = []
    for topic_id in keep_ids:
        idxs = df.index[df["topic_id"] == topic_id].to_numpy()
        if len(idxs) == 0:
            continue

        centroid = centroids.loc[topic_id].to_numpy()
        mat = emb[idxs, :]
        dists = cosine_distances_to(centroid, mat)
        order = np.argsort(dists)[: min(top_k_reps, len(idxs))]
        chosen = idxs[order]

        for rank, i in enumerate(chosen, start=1):
            rep_rows.append(
                {
                    "topic_id": topic_id,
                    "rank": rank,
                    "cosine_dist_to_centroid": float(dists[order[rank - 1]]),
                    "sprout_guid": df.at[i, "sprout_guid"],
                    "created_time": df.at[i, "created_time"],
                    "sentiment": df.at[i, "sentiment"],
                    "brand": df.at[i, "brand"],
                    "deidentified_text": df.at[i, "deidentified_text"],
                }
            )

    rep_docs_df = pd.DataFrame(rep_rows).sort_values(["topic_id", "rank"]).reset_index(drop=True)
    topic_summary_df = sent_pivot.sort_values(
        ["negative", "n_docs"], ascending=[False, False]
    ).reset_index(drop=True)
    all_non_out_df = df[df["topic_id"] >= 0].copy()

    topic_summary_df.to_csv(artifacts_dir / "topic_summary.csv", index=False)
    rep_docs_df.to_csv(artifacts_dir / "representative_docs.csv", index=False)
    all_non_out_df.to_csv(artifacts_dir / "all_docs_with_topics.csv", index=False)

    return topic_summary_df, rep_docs_df, all_non_out_df


def compare_topic_distributions(df_before: pd.DataFrame, df_after: pd.DataFrame) -> pd.DataFrame:
    def _agg(df: pd.DataFrame, label: str) -> pd.DataFrame:
        size = df.groupby("topic_id").size().rename(f"n_{label}").to_frame()

        sent_pivot = (
            df.groupby("topic_id")["sentiment"].value_counts(normalize=True).unstack(fill_value=0.0)
        )
        for col in ["negative", "neutral", "positive"]:
            if col not in sent_pivot.columns:
                sent_pivot[col] = 0.0

        sent_pivot = sent_pivot.rename(
            columns={
                "negative": f"sent_{label}_negative",
                "neutral": f"sent_{label}_neutral",
                "positive": f"sent_{label}_positive",
            }
        )
        out = size.join(sent_pivot, how="left")

        if "brand" in df.columns:
            brand_pivot = (
                df.groupby("topic_id")["brand"].value_counts(normalize=True).unstack(fill_value=0.0)
            )
            brand_cols = sorted(brand_pivot.columns)
            brand_pivot = brand_pivot[brand_cols].rename(
                columns={b: f"brand_{label}_{b}" for b in brand_cols}
            )
            out = out.join(brand_pivot, how="left")

        return out.reset_index()

    agg_before = _agg(df_before, label="before")
    agg_after = _agg(df_after, label="after")
    merged = agg_before.merge(agg_after, on="topic_id", how="outer").fillna(0.0)

    merged["n_delta"] = merged["n_after"] - merged["n_before"]
    for col in ["negative", "neutral", "positive"]:
        merged[f"sent_delta_{col}"] = merged[f"sent_after_{col}"] - merged[f"sent_before_{col}"]

    brand_before_cols = [c for c in merged.columns if c.startswith("brand_before_")]
    for col in brand_before_cols:
        brand_name = col.replace("brand_before_", "")
        after_col = f"brand_after_{brand_name}"
        if after_col not in merged.columns:
            merged[after_col] = 0.0
        merged[f"brand_delta_{brand_name}"] = merged[after_col] - merged[col]

    return merged


def summarize_topic_stability_filtered(
    df: pd.DataFrame,
    growth_threshold: float = 0.10,
    focus_sentiment: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_persisting = df[df["n_before"] > 0].copy()
    df_new = df[df["n_before"] == 0].copy()
    if df_persisting.empty:
        raise ValueError("No persisting topics found (all topics have n_before=0)")

    df_work = df_persisting.copy()
    if focus_sentiment is not None:
        focus = focus_sentiment.lower()
        valid = {"negative", "neutral", "positive"}
        if focus not in valid:
            raise ValueError(f"focus_sentiment must be one of {valid} or None")
        dom_after = df_work[
            ["sent_after_negative", "sent_after_neutral", "sent_after_positive"]
        ].idxmax(axis=1)
        df_work = df_work[dom_after == f"sent_after_{focus}"]
        if df_work.empty:
            raise ValueError(
                f"No persisting topics where AFTER-merge dominant sentiment is '{focus_sentiment}'"
            )

    num_persisting = len(df_work)
    sent_cols = [c for c in df.columns if c.startswith("sent_delta_")]
    avg_sent_shift = float(df_work[sent_cols].abs().mean().mean())
    max_sent_shift = float(df_work[sent_cols].abs().max().max())

    avg_neg_shift = float(df_work["sent_delta_negative"].abs().mean())
    avg_neu_shift = float(df_work["sent_delta_neutral"].abs().mean())
    avg_pos_shift = float(df_work["sent_delta_positive"].abs().mean())

    brand_delta_cols = [c for c in df.columns if c.startswith("brand_delta_")]
    avg_brand_shift = (
        float(df_work[brand_delta_cols].abs().mean().mean()) if brand_delta_cols else np.nan
    )

    before_cols = [c for c in df.columns if c.startswith("brand_before_")]
    after_cols = [c for c in df.columns if c.startswith("brand_after_")]
    if before_cols and after_cols:
        brands_before = {c.replace("brand_before_", "") for c in before_cols}
        brands_after = {c.replace("brand_after_", "") for c in after_cols}
        brands = sorted(brands_before.intersection(brands_after))
        before_mat = df_work[[f"brand_before_{b}" for b in brands]].to_numpy()
        after_mat = df_work[[f"brand_after_{b}" for b in brands]].to_numpy()
        stable_dom_count = int((before_mat.argmax(axis=1) == after_mat.argmax(axis=1)).sum())
        stable_dom_str = f"{stable_dom_count}/{num_persisting}"
    else:
        stable_dom_str = np.nan

    n_before = df_work["n_before"].replace(0, np.nan)
    rel_delta = df_work["n_delta"] / n_before
    small_growth_count = int((rel_delta.abs() < growth_threshold).sum())
    small_growth_str = f"{small_growth_count}/{num_persisting}"

    if df_new.empty:
        new_neg = new_neu = new_pos = new_total = 0
    else:
        dom_new = df_new[
            ["sent_after_negative", "sent_after_neutral", "sent_after_positive"]
        ].idxmax(axis=1)
        new_neg = int((dom_new == "sent_after_negative").sum())
        new_neu = int((dom_new == "sent_after_neutral").sum())
        new_pos = int((dom_new == "sent_after_positive").sum())
        new_total = len(df_new)

    metrics_data = {
        "metric": [
            "avg_abs_sentiment_shift",
            "max_abs_sentiment_shift",
            "avg_abs_sentiment_shift_negative",
            "avg_abs_sentiment_shift_neutral",
            "avg_abs_sentiment_shift_positive",
            "avg_abs_brand_shift",
            "topics_with_stable_dominant_brand",
            "topics_with_small_doc_change",
        ],
        "value": [
            avg_sent_shift,
            max_sent_shift,
            avg_neg_shift,
            avg_neu_shift,
            avg_pos_shift,
            avg_brand_shift,
            stable_dom_str,
            small_growth_str,
        ],
    }

    if focus_sentiment is None:
        metrics_data["metric"].extend(
            [
                "num_new_topics_negative",
                "num_new_topics_neutral",
                "num_new_topics_positive",
                "num_new_topics_total",
            ]
        )
        metrics_data["value"].extend([new_neg, new_neu, new_pos, new_total])
    else:
        pick = {"negative": new_neg, "neutral": new_neu, "positive": new_pos}
        metrics_data["metric"].append("num_new_topics")
        metrics_data["value"].append(pick[focus_sentiment.lower()])

    metrics = pd.DataFrame(metrics_data)
    return metrics, df_work, df_new


def select_top_topic_changes(
    df_persisting: pd.DataFrame,
    topic_info_df: pd.DataFrame,
    top_n: int = 5,
) -> pd.DataFrame:
    df_sorted = df_persisting.reindex(
        df_persisting["n_delta"].abs().sort_values(ascending=False).index
    )
    top_topics = df_sorted.head(top_n).copy()
    merged = top_topics.merge(
        topic_info_df[["Topic", "Representation"]],
        left_on="topic_id",
        right_on="Topic",
        how="left",
    )
    cols = [
        "topic_id",
        "n_before",
        "n_after",
        "n_delta",
        "sent_after_negative",
        "sent_after_neutral",
        "sent_after_positive",
        "Representation",
    ]
    return merged[cols]


def select_top_new_topics(
    df_new: pd.DataFrame, topic_info_df: pd.DataFrame, top_n: int = 5
) -> pd.DataFrame:
    if df_new.empty:
        return pd.DataFrame(
            columns=[
                "topic_id",
                "n_after",
                "sent_after_negative",
                "sent_after_neutral",
                "sent_after_positive",
                "Representation",
            ]
        )
    top_new = df_new.sort_values("n_after", ascending=False).head(top_n).copy()
    merged = top_new.merge(
        topic_info_df[["Topic", "Representation"]],
        left_on="topic_id",
        right_on="Topic",
        how="left",
    )
    cols = [
        "topic_id",
        "n_after",
        "sent_after_negative",
        "sent_after_neutral",
        "sent_after_positive",
        "Representation",
    ]
    return merged[cols]

from __future__ import annotations

import pickle
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from incremental_bertopic_brand_crisis_detection.topic_model.compat import (
    ensure_llama_cpp_importable,
)

ensure_llama_cpp_importable()

from bertopic import BERTopic
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity

from incremental_bertopic_brand_crisis_detection.topic_model.metrics import (
    coherence_cv,
    compute_score_j,
    get_top_words,
    outlier_rate,
    redundancy_jaccard,
    topic_diversity,
)
from incremental_bertopic_brand_crisis_detection.topic_model.objective import (
    TopicModelConfig,
    TopicModelFactory,
)

TOP_N_WORDS = 10
OUT_ALPHA = 0.7


def _topic_config_from_dict(params: dict[str, Any]) -> TopicModelConfig:
    raw_ngram = params["ngram"]
    if isinstance(raw_ngram, str):
        ngram = (1, 1) if raw_ngram == "1,1" else (1, 2)
    else:
        ngram = tuple(raw_ngram)

    return TopicModelConfig(
        ngram=ngram,
        min_df=float(params["min_df"]),
        max_df=float(params["max_df"]),
        nn=int(params["nn"]),
        nc=int(params["nc"]),
        min_dist=float(params["min_dist"]),
        mcs=int(params["mcs"]),
        csm=str(params["csm"]),
    )


def _save_topic_info(model: BERTopic, out_dir: Path, split_name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    model.get_topic_info().to_csv(out_dir / f"topic_info_{split_name}.csv", index=False)


def _save_topic_assignments(topics: list[int] | np.ndarray, out_dir: Path, split_name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"topic": topics}).to_csv(
        out_dir / f"topic_assignments_{split_name}.csv", index=False
    )


def eval_split_pre(
    model: BERTopic,
    docs: list[str],
    emb: np.ndarray,
    top_n: int,
    out_reduce_threshold: float,
    topics_base: list[int] | np.ndarray | None = None,
) -> tuple[np.ndarray, float, float, float, int, float, float, float, int]:
    topics = (
        np.asarray(topics_base)
        if topics_base is not None
        else np.asarray(model.transform(docs, embeddings=emb)[0])
    )

    out_before = outlier_rate(topics)
    topics_reassigned = np.asarray(
        model.reduce_outliers(
            docs,
            topics,
            strategy="embeddings",
            threshold=out_reduce_threshold,
            embeddings=emb,
        )
    )
    out_after = outlier_rate(topics_reassigned)

    non_out = [t for t in topics_reassigned if t != -1]
    if non_out:
        cnts = Counter(non_out)
        mega = max(cnts.values()) / sum(cnts.values())
        num_topics = len(cnts)
    else:
        mega, num_topics = 1.0, 0

    coh_pre = coherence_cv(model, docs, top_n)
    top_words_pre = get_top_words(model, top_n)
    div_pre = topic_diversity(top_words_pre, top_n)
    red_pre = redundancy_jaccard(model, top_n)

    return (
        topics_reassigned,
        out_before,
        out_after,
        mega,
        num_topics,
        coh_pre,
        div_pre,
        red_pre,
        len(top_words_pre),
    )


def centroids(emb: np.ndarray, topics: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame(emb)
    df["topic"] = topics
    return df[df.topic >= 0].groupby("topic").mean()


def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    if np.allclose(a, 0, atol=1e-12) or np.allclose(b, 0, atol=1e-12):
        return np.nan
    return float(cosine_similarity(a.reshape(1, -1), b.reshape(1, -1))[0, 0])


def per_topic_cosines(
    emb_a: np.ndarray,
    topics_a: np.ndarray,
    emb_b: np.ndarray,
    topics_b: np.ndarray,
) -> np.ndarray:
    c_a = centroids(emb_a, topics_a)
    c_b = centroids(emb_b, topics_b)
    common = c_a.index.intersection(c_b.index)
    if len(common) == 0:
        return np.empty((0,), dtype=float)

    sims = []
    for topic_id in common:
        sim = cos_sim(c_a.loc[topic_id].values, c_b.loc[topic_id].values)
        if not np.isnan(sim):
            sims.append(sim)
    return np.asarray(sims, dtype=float)


def summarize_sims(sims: np.ndarray) -> dict[str, float | int]:
    if sims.size == 0:
        return {"mean": np.nan, "median": np.nan, "p05": np.nan, "p95": np.nan, "n_matched": 0}
    return {
        "mean": float(np.mean(sims)),
        "median": float(np.median(sims)),
        "p05": float(np.percentile(sims, 5)),
        "p95": float(np.percentile(sims, 95)),
        "n_matched": int(sims.size),
    }


def support_stats(topics_arr: np.ndarray, min_support: int = 5) -> dict[str, float | int]:
    cnt = Counter([t for t in topics_arr if t != -1])
    if not cnt:
        return {
            "n_topics": 0,
            "min": np.nan,
            "median": np.nan,
            "p05": np.nan,
            "p95": np.nan,
            "n_lt5": 0,
        }
    vals = np.asarray(list(cnt.values()), dtype=int)
    return {
        "n_topics": int(vals.size),
        "min": int(vals.min()),
        "median": float(np.median(vals)),
        "p05": float(np.percentile(vals, 5)),
        "p95": float(np.percentile(vals, 95)),
        "n_lt5": int((vals < min_support).sum()),
    }


def evaluate_baseline(
    factory: TopicModelFactory,
    best_params: dict[str, Any],
    docs_train: list[str],
    emb_train: np.ndarray,
    docs_valid: list[str],
    emb_valid: np.ndarray,
    docs_test: list[str],
    emb_test: np.ndarray,
    out_reduce_threshold: float,
) -> tuple[dict[str, Any], BERTopic, np.ndarray, np.ndarray, np.ndarray]:
    model_cfg = _topic_config_from_dict(best_params)
    model = factory.build_model(model_cfg)

    topics_train_raw, _ = model.fit_transform(docs_train, embeddings=emb_train)
    topics_train_raw = np.asarray(topics_train_raw)
    topics_train_after = np.asarray(
        model.reduce_outliers(
            docs_train,
            topics_train_raw,
            strategy="embeddings",
            threshold=out_reduce_threshold,
            embeddings=emb_train,
        )
    )
    model.update_topics(docs_train, topics=topics_train_after)

    coh_train = coherence_cv(model, docs_train, TOP_N_WORDS)
    div_train = topic_diversity(get_top_words(model, TOP_N_WORDS), TOP_N_WORDS)
    red_train = redundancy_jaccard(model, TOP_N_WORDS)
    out_train_before = outlier_rate(topics_train_raw)
    out_train_after = outlier_rate(topics_train_after)

    train_non_out = [t for t in topics_train_after if t != -1]
    if train_non_out:
        train_counts = Counter(train_non_out)
        mega_train = max(train_counts.values()) / sum(train_counts.values())
        n_topics_train = len(train_counts)
    else:
        mega_train = 1.0
        n_topics_train = 0

    out_blend_train = (OUT_ALPHA * out_train_after) + ((1 - OUT_ALPHA) * out_train_before)
    j_train = compute_score_j(coh_train, div_train, out_blend_train, mega_train)

    topics_valid_raw, _ = model.transform(docs_valid, embeddings=emb_valid)
    topics_test_raw, _ = model.transform(docs_test, embeddings=emb_test)

    (
        topics_valid_after,
        out_valid_before,
        out_valid_after,
        mega_valid,
        n_topics_valid,
        coh_valid,
        div_valid,
        red_valid,
        _,
    ) = eval_split_pre(
        model,
        docs_valid,
        emb_valid,
        TOP_N_WORDS,
        out_reduce_threshold,
        topics_base=np.asarray(topics_valid_raw),
    )
    (
        topics_test_after,
        out_test_before,
        out_test_after,
        mega_test,
        n_topics_test,
        coh_test,
        div_test,
        red_test,
        _,
    ) = eval_split_pre(
        model,
        docs_test,
        emb_test,
        TOP_N_WORDS,
        out_reduce_threshold,
        topics_base=np.asarray(topics_test_raw),
    )

    out_blend_valid = (OUT_ALPHA * out_valid_after) + ((1 - OUT_ALPHA) * out_valid_before)
    out_blend_test = (OUT_ALPHA * out_test_after) + ((1 - OUT_ALPHA) * out_test_before)
    j_valid = compute_score_j(coh_valid, div_valid, out_blend_valid, mega_valid)
    j_test = compute_score_j(coh_test, div_test, out_blend_test, mega_test)

    sims_tv_pre = summarize_sims(
        per_topic_cosines(emb_train, topics_train_raw, emb_valid, np.asarray(topics_valid_raw))
    )
    sims_tt_pre = summarize_sims(
        per_topic_cosines(emb_train, topics_train_raw, emb_test, np.asarray(topics_test_raw))
    )
    sims_tv_post = summarize_sims(
        per_topic_cosines(emb_train, topics_train_after, emb_valid, topics_valid_after)
    )
    sims_tt_post = summarize_sims(
        per_topic_cosines(emb_train, topics_train_after, emb_test, topics_test_after)
    )

    metrics = {
        "coh_train": coh_train,
        "div_train": div_train,
        "redundancy_jaccard_train": red_train,
        "outlier_train_before": out_train_before,
        "outlier_train_after": out_train_after,
        "mega_train": mega_train,
        "num_topics_train": n_topics_train,
        "J_train": float(j_train),
        "coh_valid": coh_valid,
        "div_valid": div_valid,
        "redundancy_jaccard_valid": red_valid,
        "outlier_valid_before": out_valid_before,
        "outlier_valid_after": out_valid_after,
        "mega_valid": mega_valid,
        "num_topics_valid": n_topics_valid,
        "J_valid": float(j_valid),
        "coh_test": coh_test,
        "div_test": div_test,
        "redundancy_jaccard_test": red_test,
        "outlier_test_before": out_test_before,
        "outlier_test_after": out_test_after,
        "mega_test": mega_test,
        "num_topics_test": n_topics_test,
        "J_test": float(j_test),
        "topic_stability_train_valid_pre": sims_tv_pre["mean"],
        "topic_stability_train_test_pre": sims_tt_pre["mean"],
        "topic_stability_train_valid_post": sims_tv_post["mean"],
        "topic_stability_train_test_post": sims_tt_post["mean"],
        "valid_support_topics": support_stats(topics_valid_after)["n_topics"],
        "test_support_topics": support_stats(topics_test_after)["n_topics"],
    }

    return metrics, model, topics_train_after, topics_valid_after, topics_test_after


def run_eval_pipeline(
    factory: TopicModelFactory,
    study_path: Path,
    vectorizer_path: Path,
    docs_train: list[str],
    emb_train: np.ndarray,
    docs_valid: list[str],
    emb_valid: np.ndarray,
    docs_test: list[str],
    emb_test: np.ndarray,
    out_reduce_threshold: float,
    out_dir: Path,
) -> dict[str, Any]:
    with study_path.open("rb") as f:
        study = pickle.load(f)
    best_params = study.best_trial.params

    base_vectorizer = joblib.load(vectorizer_path)
    factory.default_vectorizer = base_vectorizer

    metrics, model, topics_train, topics_valid, topics_test = evaluate_baseline(
        factory=factory,
        best_params=best_params,
        docs_train=docs_train,
        emb_train=emb_train,
        docs_valid=docs_valid,
        emb_valid=emb_valid,
        docs_test=docs_test,
        emb_test=emb_test,
        out_reduce_threshold=out_reduce_threshold,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "topic_info").mkdir(parents=True, exist_ok=True)
    (out_dir / "topic_assignments").mkdir(parents=True, exist_ok=True)

    _save_topic_info(model, out_dir / "topic_info", "train")
    _save_topic_info(model, out_dir / "topic_info", "valid")
    _save_topic_info(model, out_dir / "topic_info", "test")

    _save_topic_assignments(topics_train, out_dir / "topic_assignments", "train")
    _save_topic_assignments(topics_valid, out_dir / "topic_assignments", "valid")
    _save_topic_assignments(topics_test, out_dir / "topic_assignments", "test")

    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(out_dir / "baseline_eval_metrics.csv", index=False)

    model_dir = out_dir / "model_core"
    model.save(str(model_dir))

    with (out_dir / "baseline_topics.pkl").open("wb") as f:
        pickle.dump(
            {
                "topics_train_after": topics_train,
                "topics_valid_after": topics_valid,
                "topics_test_after": topics_test,
                "metrics": metrics,
            },
            f,
        )

    return metrics


def default_stop_words() -> list[str]:
    custom_stops = {"falcons", "marta", "waffle", "house", "waffle_house", "wafflehouse"}
    return list(ENGLISH_STOP_WORDS.union(custom_stops))

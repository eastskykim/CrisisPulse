from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd

from incremental_bertopic_brand_crisis_detection.topic_model.compat import (
    ensure_llama_cpp_importable,
)

ensure_llama_cpp_importable()

from bertopic import BERTopic

from incremental_bertopic_brand_crisis_detection.topic_model.evaluate import TOP_N_WORDS
from incremental_bertopic_brand_crisis_detection.topic_model.metrics import (
    coherence_cv,
    compute_score_j,
    get_top_words,
    outlier_rate,
    topic_diversity,
)
from incremental_bertopic_brand_crisis_detection.topic_model.objective import (
    TopicModelConfig,
    TopicModelFactory,
)

OUT_ALPHA = 0.7


def _topic_cfg_from_best(best_params: dict[str, Any], mcs: int, csm: str) -> TopicModelConfig:
    raw_ngram = best_params["ngram"]
    if isinstance(raw_ngram, str):
        ngram = (1, 1) if raw_ngram == "1,1" else (1, 2)
    else:
        ngram = tuple(raw_ngram)

    return TopicModelConfig(
        ngram=ngram,
        min_df=float(best_params["min_df"]),
        max_df=float(best_params["max_df"]),
        nn=int(best_params["nn"]),
        nc=int(best_params["nc"]),
        min_dist=float(best_params["min_dist"]),
        mcs=int(mcs),
        csm=str(csm),
    )


def fit_slice_with_params(
    factory: TopicModelFactory,
    cfg: TopicModelConfig,
    docs_slice: list[str],
    emb_slice: np.ndarray,
    out_reduce_threshold: float,
) -> tuple[BERTopic, np.ndarray, dict[str, float]]:
    model = factory.build_model(cfg)
    topics_raw, _ = model.fit_transform(docs_slice, embeddings=emb_slice)
    topics_raw = np.asarray(topics_raw)

    out_before = outlier_rate(topics_raw)
    topics_post = np.asarray(
        model.reduce_outliers(
            docs_slice,
            topics_raw,
            strategy="embeddings",
            threshold=out_reduce_threshold,
            embeddings=emb_slice,
        )
    )
    out_after = outlier_rate(topics_post)

    non_out = [t for t in topics_post if t != -1]
    if non_out:
        values, counts = np.unique(non_out, return_counts=True)
        _ = values
        mega = float(counts.max() / counts.sum())
        num_topics = int(len(counts))
    else:
        mega = 1.0
        num_topics = 0

    coh = coherence_cv(model, docs_slice, top_n=TOP_N_WORDS)
    div = topic_diversity(get_top_words(model, top_n=TOP_N_WORDS), n=TOP_N_WORDS)
    out_blend = (OUT_ALPHA * out_after) + ((1.0 - OUT_ALPHA) * out_before)
    score = compute_score_j(coh, div, out_blend, mega)

    model.update_topics(docs_slice, topics=topics_post)
    metrics = {
        "J": float(score),
        "coherence": float(coh),
        "diversity": float(div),
        "outlier_before": float(out_before),
        "outlier_after": float(out_after),
        "mega_cluster_share": float(mega),
        "num_topics": float(num_topics),
    }
    return model, topics_post, metrics


def tune_delta_on_valid(
    factory: TopicModelFactory,
    best_params: dict[str, Any],
    docs_valid: list[str],
    emb_valid: np.ndarray,
    out_reduce_threshold: float,
    n_trials: int = 25,
) -> dict[str, Any]:
    def objective(trial: optuna.Trial) -> float:
        mcs = trial.suggest_int("mcs", 10, 50, step=5)
        csm = trial.suggest_categorical("csm", ["leaf"])
        cfg = _topic_cfg_from_best(best_params, mcs=mcs, csm=csm)
        _, _, metrics = fit_slice_with_params(
            factory=factory,
            cfg=cfg,
            docs_slice=docs_valid,
            emb_slice=emb_valid,
            out_reduce_threshold=out_reduce_threshold,
        )
        for k, v in metrics.items():
            trial.set_user_attr(k, v)
        return metrics["J"]

    study = optuna.create_study(study_name="delta_hdbscan_tuning", direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return {
        "best_params": study.best_trial.params,
        "best_value": float(study.best_trial.value),
        "study": study,
    }


def build_merged_models(
    baseline_model: BERTopic,
    delta_valid: BERTopic,
    delta_test_fit: BERTopic,
    tau: float = 0.95,
) -> tuple[BERTopic, BERTopic]:
    merged_valid = BERTopic.merge_models([baseline_model, delta_valid], min_similarity=tau)
    merged_final = BERTopic.merge_models([merged_valid, delta_test_fit], min_similarity=tau)
    return merged_valid, merged_final


def save_model_snapshots(
    out_dir: Path,
    baseline_model: BERTopic,
    delta_valid: BERTopic,
    delta_test_fit: BERTopic,
    merged_valid: BERTopic,
    merged_final: BERTopic,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    models_dir = out_dir / "models"
    info_dir = out_dir / "topic_info_snapshots"
    models_dir.mkdir(parents=True, exist_ok=True)
    info_dir.mkdir(parents=True, exist_ok=True)

    baseline_model.save(str(models_dir / "baseline_model"))
    delta_valid.save(str(models_dir / "delta_valid_model"))
    delta_test_fit.save(str(models_dir / "delta_test_fit_model"))
    merged_valid.save(str(models_dir / "merged_valid_model"))
    merged_final.save(str(models_dir / "merged_final_model"))

    baseline_model.get_topic_info().to_csv(info_dir / "baseline.csv", index=False)
    delta_valid.get_topic_info().to_csv(info_dir / "delta_valid.csv", index=False)
    delta_test_fit.get_topic_info().to_csv(info_dir / "delta_test_fit.csv", index=False)
    merged_valid.get_topic_info().to_csv(info_dir / "merged_valid.csv", index=False)
    merged_final.get_topic_info().to_csv(info_dir / "merged_final.csv", index=False)


def save_merge_metadata(
    out_dir: Path,
    tau: float,
    delta_tuning: dict[str, Any],
    metrics_valid: dict[str, float],
    metrics_test_fit: dict[str, float],
) -> None:
    payload = {
        "tau": tau,
        "delta_tuning_best": delta_tuning.get("best_params", {}),
        "delta_tuning_best_value": delta_tuning.get("best_value", np.nan),
        "delta_valid_metrics": metrics_valid,
        "delta_test_fit_metrics": metrics_test_fit,
    }
    with (out_dir / "merge_metadata.pkl").open("wb") as f:
        pickle.dump(payload, f)
    pd.DataFrame([payload]).to_json(out_dir / "merge_metadata.json", orient="records", indent=2)

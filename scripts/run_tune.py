from __future__ import annotations

import argparse
import os
import pickle
import time
from collections import Counter

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer

from incremental_bertopic_brand_crisis_detection.config import load_config
from incremental_bertopic_brand_crisis_detection.embeddings.e5 import load_docs_and_embeddings
from incremental_bertopic_brand_crisis_detection.topic_model.compat import (
    ensure_llama_cpp_importable,
)
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune BERTopic baseline with Optuna.")
    parser.add_argument("--config", default="configs/base.yaml", help="Path to YAML config")
    parser.add_argument("--trials", type=int, default=20, help="Number of Optuna trials")
    parser.add_argument(
        "--trial-max-minutes",
        type=float,
        default=15.0,
        help="Prune trial if it exceeds this wall-clock budget",
    )
    parser.add_argument(
        "--study-name",
        default="baseline_tuning_v2",
        help="Optuna study name",
    )
    parser.add_argument(
        "--storage",
        default="",
        help="Optuna storage URL (default: sqlite in artifacts/baseline_model)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    ensure_llama_cpp_importable()
    from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance

    docs_train, emb_train = load_docs_and_embeddings(
        "train", cfg.paths.data_dir, cfg.paths.artifacts_dir / "embeddings"
    )
    docs_valid, emb_valid = load_docs_and_embeddings(
        "valid", cfg.paths.data_dir, cfg.paths.artifacts_dir / "embeddings"
    )

    representation_model = [KeyBERTInspired(), MaximalMarginalRelevance()]

    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedding_model = SentenceTransformer(cfg.embedding.model_name, device=device)

    custom_stops = {"falcons", "marta", "waffle", "house", "waffle_house", "wafflehouse"}
    rep_stop = list(ENGLISH_STOP_WORDS.union(custom_stops))

    factory = TopicModelFactory(
        representation_model=representation_model,
        embedding_model=embedding_model,
        stop_words=rep_stop,
    )

    baseline_artifacts_dir = cfg.paths.artifacts_dir / "baseline_model"
    baseline_models_dir = cfg.paths.models_dir / "baseline_model"
    baseline_artifacts_dir.mkdir(parents=True, exist_ok=True)
    baseline_models_dir.mkdir(parents=True, exist_ok=True)
    (baseline_artifacts_dir / "vectorizer").mkdir(parents=True, exist_ok=True)

    storage_url = args.storage.strip()
    if not storage_url:
        storage_url = (
            f"sqlite:///{(baseline_artifacts_dir / f'optuna_{args.study_name}.db').as_posix()}"
        )

    def objective(trial: optuna.Trial) -> float:
        start = time.time()

        def check_timeout(stage: str) -> None:
            elapsed_min = (time.time() - start) / 60.0
            if elapsed_min > args.trial_max_minutes:
                trial.set_user_attr("pruned_stage", stage)
                trial.set_user_attr("duration_sec", time.time() - start)
                raise optuna.exceptions.TrialPruned()

        ngram_choice = trial.suggest_categorical("ngram", ["1,1", "1,2"])
        ngram = (1, 1) if ngram_choice == "1,1" else (1, 2)
        model_cfg = TopicModelConfig(
            ngram=ngram,
            min_df=trial.suggest_float("min_df", 0.003, 0.01, log=True),
            max_df=trial.suggest_float("max_df", 0.85, 0.95, step=0.03),
            nn=trial.suggest_int("nn", 20, 100, step=10),
            nc=trial.suggest_int("nc", 20, 70, step=20),
            min_dist=trial.suggest_float("min_dist", 0.15, 0.4, step=0.05),
            mcs=trial.suggest_int("mcs", 30, 100, step=10),
            csm=trial.suggest_categorical("csm", ["leaf"]),
        )

        model = factory.build_model(model_cfg)
        topics_train, _ = model.fit_transform(docs_train, embeddings=emb_train)
        check_timeout("fit_transform_train")
        if np.all(np.asarray(topics_train) == -1):
            raise optuna.exceptions.TrialPruned()

        topics_valid, _ = model.transform(docs_valid, embeddings=emb_valid)
        check_timeout("transform_valid")
        topics_valid_reassigned = model.reduce_outliers(
            docs_valid,
            topics_valid,
            strategy="embeddings",
            threshold=cfg.pipeline.outlier_reduce_threshold,
            embeddings=emb_valid,
        )
        check_timeout("reduce_outliers_valid")

        out_before = outlier_rate(topics_valid)
        out_after = outlier_rate(topics_valid_reassigned)

        non_out = [t for t in topics_valid_reassigned if t != -1]
        if non_out:
            counts = Counter(non_out)
            mega = max(counts.values()) / sum(counts.values())
        else:
            mega = 1.0

        coh = coherence_cv(model, docs_valid, top_n=TOP_N_WORDS)
        top_words = get_top_words(model, top_n=TOP_N_WORDS)
        div = topic_diversity(top_words, n=TOP_N_WORDS)
        red = redundancy_jaccard(model, top_n=TOP_N_WORDS)

        out_blend = (OUT_ALPHA * out_after) + ((1 - OUT_ALPHA) * out_before)
        score = compute_score_j(coh, div, out_blend, mega)

        trial.set_user_attr("coh_valid", coh)
        trial.set_user_attr("div_valid", div)
        trial.set_user_attr("red_valid", red)
        trial.set_user_attr("outlier_valid_before", out_before)
        trial.set_user_attr("outlier_valid_after", out_after)
        trial.set_user_attr("mega_valid", mega)
        trial.set_user_attr("duration_sec", time.time() - start)
        return score

    study = optuna.create_study(
        study_name=args.study_name,
        direction="maximize",
        storage=storage_url,
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0),
    )
    before_n = len(study.trials)
    target_n = before_n + args.trials

    def progress_callback(st: optuna.Study, tr: optuna.Trial) -> None:
        completed = len(st.trials)
        try:
            best = st.best_value
        except ValueError:
            best = float("nan")
        dur = tr.user_attrs.get("duration_sec", float("nan"))
        print(
            f"[Progress] completed={completed}/{target_n} "
            f"trial={tr.number} state={tr.state.name} "
            f"value={tr.value} duration_sec={dur:.1f} best={best:.6f}"
        )

    while len(study.trials) < target_n:
        remaining = target_n - len(study.trials)
        study.optimize(
            objective,
            n_trials=remaining,
            n_jobs=1,
            show_progress_bar=True,
            gc_after_trial=True,
            callbacks=[progress_callback],
        )

    study_file = baseline_models_dir / "optuna_study.pkl"
    with study_file.open("wb") as f:
        pickle.dump(study, f)

    results_df = study.trials_dataframe()
    user_attrs_df = pd.DataFrame([{"trial": t.number, **t.user_attrs} for t in study.trials])
    results_df = results_df.merge(
        user_attrs_df, left_on="number", right_on="trial", how="left"
    ).drop(columns=["trial"])
    results_df.to_csv(baseline_artifacts_dir / "optuna_study_results.csv", index=False)

    best = study.best_trial.params

    best_ngram = best["ngram"]
    if isinstance(best_ngram, str):
        ngram_tuple = (1, 1) if best_ngram == "1,1" else (1, 2)
    elif isinstance(best_ngram, list):
        ngram_tuple = tuple(best_ngram)
    else:
        ngram_tuple = best_ngram

    base_vectorizer = CountVectorizer(
        stop_words=rep_stop,
        ngram_range=ngram_tuple,
        min_df=best["min_df"],
        max_df=best["max_df"],
        token_pattern=r"(?u)\b[a-z]{3,}\b",
    )
    base_vectorizer.fit(docs_train)
    joblib.dump(base_vectorizer, baseline_artifacts_dir / "vectorizer" / "vectorizer_baseline.pkl")

    print("Best params:", best)


if __name__ == "__main__":
    main()

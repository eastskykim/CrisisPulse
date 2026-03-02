from __future__ import annotations

import argparse
import pickle

import joblib
import torch
from sentence_transformers import SentenceTransformer

from incremental_bertopic_brand_crisis_detection.config import load_config
from incremental_bertopic_brand_crisis_detection.embeddings.e5 import load_docs_and_embeddings
from incremental_bertopic_brand_crisis_detection.topic_model.compat import (
    ensure_llama_cpp_importable,
)
from incremental_bertopic_brand_crisis_detection.topic_model.evaluate import default_stop_words
from incremental_bertopic_brand_crisis_detection.topic_model.merge import (
    _topic_cfg_from_best,
    build_merged_models,
    fit_slice_with_params,
    save_merge_metadata,
    save_model_snapshots,
    tune_delta_on_valid,
)
from incremental_bertopic_brand_crisis_detection.topic_model.objective import TopicModelFactory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit delta models and build merged BERTopic models."
    )
    parser.add_argument("--config", default="configs/base.yaml", help="Path to YAML config")
    parser.add_argument("--tau", type=float, default=0.95, help="Topic merge similarity threshold")
    parser.add_argument(
        "--delta-trials", type=int, default=25, help="Optuna trials for delta tuning"
    )
    parser.add_argument(
        "--test-fit-ratio",
        type=float,
        default=0.70,
        help="Chronological ratio used to build TEST_fit from TEST for delta training",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    ensure_llama_cpp_importable()
    from bertopic import BERTopic
    from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance

    emb_dir = cfg.paths.artifacts_dir / "embeddings"
    docs_train, emb_train = load_docs_and_embeddings("train", cfg.paths.data_dir, emb_dir)
    docs_valid, emb_valid = load_docs_and_embeddings("valid", cfg.paths.data_dir, emb_dir)
    docs_test, emb_test = load_docs_and_embeddings("test", cfg.paths.data_dir, emb_dir)

    if not (0.0 < args.test_fit_ratio < 1.0):
        raise ValueError("--test-fit-ratio must be between 0 and 1")

    test_fit_idx = int(len(docs_test) * args.test_fit_ratio)
    docs_test_fit = docs_test[:test_fit_idx]
    emb_test_fit = emb_test[:test_fit_idx]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedding_model = SentenceTransformer(cfg.embedding.model_name, device=device)
    representation_model = [KeyBERTInspired(), MaximalMarginalRelevance()]

    factory = TopicModelFactory(
        representation_model=representation_model,
        embedding_model=embedding_model,
        stop_words=default_stop_words(),
    )
    factory.default_vectorizer = joblib.load(
        cfg.paths.artifacts_dir / "baseline_model" / "vectorizer" / "vectorizer_baseline.pkl"
    )

    with (cfg.paths.models_dir / "baseline_model" / "optuna_study.pkl").open("rb") as f:
        study = pickle.load(f)
    best_params = study.best_trial.params

    baseline_model_path = cfg.paths.artifacts_dir / "baseline_model" / "model_core"
    if baseline_model_path.exists():
        baseline_model = BERTopic.load(str(baseline_model_path))
    else:
        base_cfg = _topic_cfg_from_best(best_params, mcs=best_params["mcs"], csm=best_params["csm"])
        baseline_model, topics_train_post, _ = fit_slice_with_params(
            factory=factory,
            cfg=base_cfg,
            docs_slice=docs_train,
            emb_slice=emb_train,
            out_reduce_threshold=cfg.pipeline.outlier_reduce_threshold,
        )
        baseline_model.update_topics(docs_train, topics=topics_train_post)

    delta_tuning = tune_delta_on_valid(
        factory=factory,
        best_params=best_params,
        docs_valid=docs_valid,
        emb_valid=emb_valid,
        out_reduce_threshold=cfg.pipeline.outlier_reduce_threshold,
        n_trials=args.delta_trials,
    )
    delta_best = delta_tuning["best_params"]

    delta_cfg = _topic_cfg_from_best(best_params, mcs=delta_best["mcs"], csm=delta_best["csm"])
    delta_valid_model, _, metrics_valid = fit_slice_with_params(
        factory=factory,
        cfg=delta_cfg,
        docs_slice=docs_valid,
        emb_slice=emb_valid,
        out_reduce_threshold=cfg.pipeline.outlier_reduce_threshold,
    )
    delta_test_fit_model, _, metrics_test_fit = fit_slice_with_params(
        factory=factory,
        cfg=delta_cfg,
        docs_slice=docs_test_fit,
        emb_slice=emb_test_fit,
        out_reduce_threshold=cfg.pipeline.outlier_reduce_threshold,
    )

    merged_valid, merged_final = build_merged_models(
        baseline_model=baseline_model,
        delta_valid=delta_valid_model,
        delta_test_fit=delta_test_fit_model,
        tau=args.tau,
    )

    out_dir = cfg.paths.artifacts_dir / "merge_model"
    save_model_snapshots(
        out_dir=out_dir,
        baseline_model=baseline_model,
        delta_valid=delta_valid_model,
        delta_test_fit=delta_test_fit_model,
        merged_valid=merged_valid,
        merged_final=merged_final,
    )
    save_merge_metadata(
        out_dir=out_dir,
        tau=args.tau,
        delta_tuning=delta_tuning,
        metrics_valid=metrics_valid,
        metrics_test_fit=metrics_test_fit,
    )

    print("Merge stage complete.")
    print(f"  tau={args.tau}")
    print(f"  test_fit_ratio={args.test_fit_ratio}")
    print(f"  test_fit_size={len(docs_test_fit)} / test_size={len(docs_test)}")
    print(f"  delta best={delta_best}")
    print(f"  delta valid J={metrics_valid['J']:.4f}")
    print(f"  delta test-fit J={metrics_test_fit['J']:.4f}")


if __name__ == "__main__":
    main()

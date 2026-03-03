from __future__ import annotations

import argparse

import torch
from sentence_transformers import SentenceTransformer

from crisispulse.config import load_config
from crisispulse.embeddings.e5 import load_docs_and_embeddings
from crisispulse.topic_model.compat import (
    ensure_llama_cpp_importable,
)
from crisispulse.topic_model.evaluate import (
    default_stop_words,
    run_eval_pipeline,
)
from crisispulse.topic_model.objective import TopicModelFactory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate baseline model on train/valid/test.")
    parser.add_argument("--config", default="configs/base.yaml", help="Path to YAML config")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    ensure_llama_cpp_importable()
    from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance

    emb_dir = cfg.paths.artifacts_dir / "embeddings"
    docs_train, emb_train = load_docs_and_embeddings("train", cfg.paths.data_dir, emb_dir)
    docs_valid, emb_valid = load_docs_and_embeddings("valid", cfg.paths.data_dir, emb_dir)
    docs_test, emb_test = load_docs_and_embeddings("test", cfg.paths.data_dir, emb_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedding_model = SentenceTransformer(cfg.embedding.model_name, device=device)
    representation_model = [KeyBERTInspired(), MaximalMarginalRelevance()]

    factory = TopicModelFactory(
        representation_model=representation_model,
        embedding_model=embedding_model,
        stop_words=default_stop_words(),
    )

    out_dir = cfg.paths.artifacts_dir / "baseline_model"
    metrics = run_eval_pipeline(
        factory=factory,
        study_path=cfg.paths.models_dir / "baseline_model" / "optuna_study.pkl",
        vectorizer_path=cfg.paths.artifacts_dir
        / "baseline_model"
        / "vectorizer"
        / "vectorizer_baseline.pkl",
        docs_train=docs_train,
        emb_train=emb_train,
        docs_valid=docs_valid,
        emb_valid=emb_valid,
        docs_test=docs_test,
        emb_test=emb_test,
        out_reduce_threshold=cfg.pipeline.outlier_reduce_threshold,
        out_dir=out_dir,
    )

    print("Baseline evaluation complete.")
    print(f"  J_train={metrics['J_train']:.4f}")
    print(f"  J_valid={metrics['J_valid']:.4f}")
    print(f"  J_test={metrics['J_test']:.4f}")


if __name__ == "__main__":
    main()

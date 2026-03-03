from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from crisispulse.config import load_config
from crisispulse.topic_model.analysis import (
    summarize_topics_by_index,
)
from crisispulse.topic_model.compat import (
    ensure_llama_cpp_importable,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build per-topic summary and representative docs.")
    parser.add_argument("--config", default="configs/base.yaml", help="Path to YAML config")
    parser.add_argument("--model-path", required=True, help="Path to BERTopic model directory")
    parser.add_argument("--docs-csv", required=True, help="Path to cleaned docs CSV")
    parser.add_argument("--embeddings-npy", required=True, help="Path to embeddings NPY")
    parser.add_argument(
        "--topics-csv",
        default="",
        help="Optional topic assignments CSV with 'topic' or 'topic_id' column",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Output directory for topic summary CSVs",
    )
    parser.add_argument("--min-docs-per-topic", type=int, default=10)
    parser.add_argument("--top-k-reps", type=int, default=5)
    parser.add_argument("--top-n-words", type=int, default=10)
    parser.add_argument(
        "--outlier-threshold",
        type=float,
        default=-1.0,
        help="Override outlier reduction threshold; negative value uses config",
    )
    return parser.parse_args()


def _load_topics_csv(path: str) -> np.ndarray:
    df = pd.read_csv(path)
    if "topic" in df.columns:
        return df["topic"].to_numpy()
    if "topic_id" in df.columns:
        return df["topic_id"].to_numpy()
    raise ValueError("topics-csv must contain 'topic' or 'topic_id' column")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    ensure_llama_cpp_importable()
    from bertopic import BERTopic

    docs_df = pd.read_csv(args.docs_csv)
    emb = np.load(args.embeddings_npy)
    model = BERTopic.load(args.model_path)

    if args.topics_csv.strip():
        topics = _load_topics_csv(args.topics_csv)
    else:
        docs = docs_df["deidentified_text"].fillna("").tolist()
        topics_raw, _ = model.transform(docs, embeddings=emb)
        threshold = (
            cfg.pipeline.outlier_reduce_threshold
            if args.outlier_threshold < 0
            else float(args.outlier_threshold)
        )
        topics = np.asarray(
            model.reduce_outliers(
                docs,
                topics_raw,
                strategy="embeddings",
                threshold=threshold,
                embeddings=emb,
            )
        )

    topic_summary_df, rep_docs_df, all_non_out_df = summarize_topics_by_index(
        model=model,
        topics=np.asarray(topics),
        combined_clean_df=docs_df,
        emb=emb,
        artifacts_dir=cfg.paths.project_root / args.out_dir,
        top_n_words=args.top_n_words,
        min_docs_per_topic=args.min_docs_per_topic,
        top_k_reps=args.top_k_reps,
    )

    print("Topic summary complete.")
    print(f"  topics_kept={topic_summary_df.shape[0]}")
    print(f"  representative_rows={rep_docs_df.shape[0]}")
    print(f"  all_non_out_rows={all_non_out_df.shape[0]}")


if __name__ == "__main__":
    main()

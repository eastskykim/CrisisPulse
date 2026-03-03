from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from crisispulse.topic_model.labeling import (
    label_topics_with_llama_cpp,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate LLM topic labels from topic_info CSV.")
    parser.add_argument(
        "--topic-info-csv",
        default="artifacts/merge_model/topic_info_snapshots/merged_final.csv",
        help="Path to topic_info CSV",
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="Path to local GGUF model file",
    )
    parser.add_argument(
        "--output-csv",
        default="artifacts/merge_model/topic_info_snapshots/merged_final_labeled.csv",
        help="Output labeled CSV",
    )
    parser.add_argument(
        "--cache-path",
        default="artifacts/merge_model/llm_labels_cache.json",
        help="Cache file for already-generated labels",
    )
    parser.add_argument(
        "--min-count", type=int, default=25, help="Only label topics with at least this count"
    )
    parser.add_argument(
        "--max-topics", type=int, default=200, help="Maximum number of topics to label"
    )
    parser.add_argument("--n-gpu-layers", type=int, default=-1, help="llama.cpp n_gpu_layers")
    parser.add_argument("--n-ctx", type=int, default=2048, help="llama.cpp context length")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    topic_df = pd.read_csv(args.topic_info_csv)

    labeled = label_topics_with_llama_cpp(
        topic_info_df=topic_df,
        model_path=Path(args.model_path),
        cache_path=Path(args.cache_path),
        min_count=args.min_count,
        max_topics=args.max_topics,
        n_gpu_layers=args.n_gpu_layers,
        n_ctx=args.n_ctx,
    )
    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    labeled.to_csv(out_path, index=False)

    print(f"Saved labeled topics -> {out_path}")
    print(f"Generated new labels: {labeled.attrs.get('generated_count', 0)}")
    print(f"Cache hits: {labeled.attrs.get('cache_hits', 0)}")


if __name__ == "__main__":
    main()

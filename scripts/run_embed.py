from __future__ import annotations

import argparse

from crisispulse.config import load_config
from crisispulse.embeddings.e5 import (
    encode_and_save,
    load_embedding_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate embeddings for train/valid/test splits.")
    parser.add_argument("--config", default="configs/base.yaml", help="Path to YAML config")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    model = load_embedding_model(
        model_name=cfg.embedding.model_name,
        max_seq_length=cfg.embedding.max_seq_length,
    )
    embeddings_dir = cfg.paths.artifacts_dir / "embeddings"

    for split in ("train", "valid", "test"):
        encode_and_save(
            model=model,
            split_name=split.upper(),
            input_csv=cfg.paths.data_dir / f"data_combined_clean_{split}.csv",
            output_npy=embeddings_dir / f"embeddings_{split}.npy",
            prefix=cfg.embedding.prefix,
            batch_size=cfg.embedding.batch_size,
            normalize_embeddings=cfg.embedding.normalize_embeddings,
        )


if __name__ == "__main__":
    main()

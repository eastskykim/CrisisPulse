from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from crisispulse.config import load_config
from crisispulse.topic_model.compat import (
    ensure_llama_cpp_importable,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare TEST_monitor inputs for drift monitoring."
    )
    parser.add_argument("--config", default="configs/base.yaml", help="Path to YAML config")
    parser.add_argument(
        "--model-path",
        default="artifacts/merge_model/models/merged_final_model",
        help="Path to merged-final BERTopic model",
    )
    parser.add_argument(
        "--test-csv",
        default="data/data_combined_clean_test.csv",
        help="Path to cleaned TEST CSV",
    )
    parser.add_argument(
        "--test-embeddings",
        default="artifacts/embeddings/embeddings_test.npy",
        help="Path to TEST embeddings NPY",
    )
    parser.add_argument(
        "--split-ratio",
        type=float,
        default=0.70,
        help="Chronological split ratio for TEST_fit vs TEST_monitor",
    )
    parser.add_argument(
        "--out-dir",
        default="artifacts/online_model",
        help="Output directory for monitor inputs",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    ensure_llama_cpp_importable()
    from bertopic import BERTopic

    test_df = pd.read_csv(cfg.paths.project_root / args.test_csv)
    emb_test = np.load(cfg.paths.project_root / args.test_embeddings)

    if len(test_df) != emb_test.shape[0]:
        raise ValueError(
            f"TEST length mismatch: docs={len(test_df)} embeddings={emb_test.shape[0]}"
        )

    test_df = test_df.copy()
    test_df["created_time"] = pd.to_datetime(test_df["created_time"])
    test_df = test_df.sort_values("created_time").reset_index(drop=True)

    n_test = len(test_df)
    split_idx = int(n_test * args.split_ratio)

    test_fit_df = test_df.iloc[:split_idx].copy().reset_index(drop=True)
    test_monitor_df = test_df.iloc[split_idx:].copy().reset_index(drop=True)

    emb_test_fit = emb_test[:split_idx]
    emb_test_monitor = emb_test[split_idx:]

    docs_monitor = test_monitor_df["deidentified_text"].fillna("").tolist()

    model = BERTopic.load(str(cfg.paths.project_root / args.model_path))
    topics_raw, _ = model.transform(docs_monitor, embeddings=emb_test_monitor)
    topics_final = np.asarray(
        model.reduce_outliers(
            docs_monitor,
            topics_raw,
            strategy="embeddings",
            threshold=cfg.pipeline.outlier_reduce_threshold,
            embeddings=emb_test_monitor,
        )
    )

    test_monitor_with_topics = test_monitor_df.copy()
    test_monitor_with_topics["topic_id"] = topics_final

    non_out_mask = topics_final >= 0
    all_non_out_monitor = test_monitor_with_topics.loc[non_out_mask].copy().reset_index(drop=True)
    emb_test_monitor_non_out = emb_test_monitor[non_out_mask]

    out_dir = cfg.paths.project_root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    test_fit_df.to_csv(out_dir / "test_fit.csv", index=False)
    test_monitor_with_topics.to_csv(out_dir / "test_monitor_with_topics.csv", index=False)
    all_non_out_monitor.to_csv(out_dir / "all_non_out_monitor.csv", index=False)

    np.save(out_dir / "embeddings_test_fit.npy", emb_test_fit)
    np.save(out_dir / "embeddings_test_monitor.npy", emb_test_monitor)
    np.save(out_dir / "embeddings_test_monitor_non_out.npy", emb_test_monitor_non_out)

    print("Prepared monitor inputs.")
    print(f"  test_total={n_test}")
    print(f"  test_fit={len(test_fit_df)}")
    print(f"  test_monitor={len(test_monitor_with_topics)}")
    print(f"  test_monitor_non_out={len(all_non_out_monitor)}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse

import pandas as pd

from crisispulse.config import load_config
from crisispulse.data.preprocessing import (
    expand_nested_columns,
    load_brand_sources,
    prepare_df,
    split_time_series_df,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare cleaned and split datasets.")
    parser.add_argument("--config", default="configs/base.yaml", help="Path to YAML config")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    data_dir = cfg.paths.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)

    raw_sources = load_brand_sources(data_dir)
    cleaned_frames: list[pd.DataFrame] = []
    for brand, df in raw_sources.items():
        expanded = expand_nested_columns(df)
        expanded["brand"] = brand
        cleaned = prepare_df(expanded, min_tokens=cfg.pipeline.min_tokens)
        cleaned.to_csv(data_dir / f"data_{brand}_clean.csv", index=False)
        cleaned_frames.append(cleaned)

    combined = pd.concat(cleaned_frames, ignore_index=True)
    combined["created_time"] = pd.to_datetime(combined["created_time"])
    combined = combined.sort_values("created_time").reset_index(drop=True)

    train_df, valid_df, test_df = split_time_series_df(combined)
    combined.to_csv(data_dir / "data_combined_clean.csv", index=False)
    train_df.to_csv(data_dir / "data_combined_clean_train.csv", index=False)
    valid_df.to_csv(data_dir / "data_combined_clean_valid.csv", index=False)
    test_df.to_csv(data_dir / "data_combined_clean_test.csv", index=False)

    print("Prepared datasets:")
    print(f"  train={len(train_df):,} rows")
    print(f"  valid={len(valid_df):,} rows")
    print(f"  test={len(test_df):,} rows")


if __name__ == "__main__":
    main()

from __future__ import annotations

import html
import re
from pathlib import Path

import emoji
import pandas as pd

COLS_NEEDED = ["sprout_guid", "created_time", "sentiment", "deidentified_text", "brand"]


def split_hashtag(tag: str) -> str:
    text = tag.replace("_", " ")
    text = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text)
    return text.lower()


def clean_text(value: str) -> str:
    if not isinstance(value, str):
        return ""

    text = html.unescape(value)

    if re.match(r"^\s*RT\b[^A-Za-z0-9]*.*\|\s*QT", text, flags=re.IGNORECASE):
        return ""
    if re.match(r"^\s*RT[^A-Za-z0-9]+", text, flags=re.IGNORECASE):
        return ""

    if re.search(r"\|\s*QT:", text, flags=re.IGNORECASE):
        text = re.sub(
            r"\s*(?:\|\s*QT\b:?\s*|QT\b\s*\|\s*)[\s\S]*$",
            "",
            text,
            flags=re.IGNORECASE,
        )

    text = emoji.replace_emoji(text, " ")
    text = re.sub(r"<\s*/?\s*[A-Za-z_][A-Za-z0-9_]*\s*>", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\b[a-z]{2,6}://\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#([A-Za-z0-9_]+)", lambda m: f" {split_hashtag(m.group(1))} ", text)
    text = re.sub(r"([:;|/\\\-,.])\1{1,}", r"\1", text)
    text = re.sub(r"\s[:;|/\\\-,.]+\s", " ", text)
    text = re.sub(r"^[^\"\w']+|[^\"\w']+$", " ", text)
    text = re.sub(r"[\t\r\n]+", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.lower().strip()


def prepare_df(df: pd.DataFrame, min_tokens: int = 5) -> pd.DataFrame:
    required = set(COLS_NEEDED)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    cleaned = df.loc[:, COLS_NEEDED].copy()
    cleaned["deidentified_text"] = cleaned["deidentified_text"].fillna("").map(clean_text)
    cleaned = cleaned[cleaned["deidentified_text"].str.len() > 0]

    if min_tokens > 0:
        token_pattern = re.compile(r"\b[a-z]{3,}\b")
        cleaned = cleaned[
            cleaned["deidentified_text"].map(lambda s: len(token_pattern.findall(s)) >= min_tokens)
        ]

    cleaned = cleaned.drop_duplicates(subset=["deidentified_text"], keep="first")
    return cleaned[COLS_NEEDED].reset_index(drop=True)


def split_time_series_df(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ordered = df.copy()
    ordered["created_time"] = pd.to_datetime(ordered["created_time"])
    ordered = ordered.sort_values("created_time").reset_index(drop=True)

    n_rows = len(ordered)
    train_end = int(0.7 * n_rows)
    valid_end = int(0.85 * n_rows)
    return ordered.iloc[:train_end], ordered.iloc[train_end:valid_end], ordered.iloc[valid_end:]


def load_brand_sources(data_dir: Path) -> dict[str, pd.DataFrame]:
    raw_dir = data_dir / "raw"
    source_dir = raw_dir if raw_dir.exists() else data_dir
    return {
        "falcons": pd.read_json(source_dir / "georgiatech_summer2025_falcons.json"),
        "marta": pd.read_json(source_dir / "georgiatech_summer2025_marta.json"),
        "wafflehouse": pd.read_json(source_dir / "georgiatech_summer2025_waffle_house.json"),
    }


def expand_nested_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "listening_metadata" in out.columns:
        meta = out["listening_metadata"].apply(pd.Series)
        out = out.drop(columns=["listening_metadata"]).join(meta)
    if "metrics" in out.columns:
        metrics = out["metrics"].apply(pd.Series)
        out = out.drop(columns=["metrics"]).join(metrics)
    return out

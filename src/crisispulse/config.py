from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class PathsConfig:
    project_root: Path
    data_dir: Path
    artifacts_dir: Path
    models_dir: Path


@dataclass(frozen=True)
class EmbeddingConfig:
    model_name: str
    batch_size: int
    max_seq_length: int
    normalize_embeddings: bool
    prefix: str


@dataclass(frozen=True)
class PipelineConfig:
    random_seed: int
    min_tokens: int
    outlier_reduce_threshold: float


@dataclass(frozen=True)
class AppConfig:
    paths: PathsConfig
    embedding: EmbeddingConfig
    pipeline: PipelineConfig


def _resolve(base_dir: Path, value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else (base_dir / p).resolve()


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f)

    base_dir = config_path.parent.parent

    paths = raw["paths"]
    embedding = raw["embedding"]
    pipeline = raw["pipeline"]

    return AppConfig(
        paths=PathsConfig(
            project_root=_resolve(base_dir, paths["project_root"]),
            data_dir=_resolve(base_dir, paths["data_dir"]),
            artifacts_dir=_resolve(base_dir, paths["artifacts_dir"]),
            models_dir=_resolve(base_dir, paths["models_dir"]),
        ),
        embedding=EmbeddingConfig(
            model_name=embedding["model_name"],
            batch_size=int(embedding["batch_size"]),
            max_seq_length=int(embedding["max_seq_length"]),
            normalize_embeddings=bool(embedding["normalize_embeddings"]),
            prefix=embedding["prefix"],
        ),
        pipeline=PipelineConfig(
            random_seed=int(pipeline["random_seed"]),
            min_tokens=int(pipeline["min_tokens"]),
            outlier_reduce_threshold=float(pipeline["outlier_reduce_threshold"]),
        ),
    )

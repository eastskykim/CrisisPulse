from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer


def load_embedding_model(model_name: str, max_seq_length: int) -> SentenceTransformer:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(
        model_name,
        device=device,
        model_kwargs={"dtype": torch.float16, "trust_remote_code": True},
    )
    model.max_seq_length = max_seq_length
    return model


def encode_and_save(
    model: SentenceTransformer,
    split_name: str,
    input_csv: Path,
    output_npy: Path,
    text_column: str = "deidentified_text",
    prefix: str = "passage: ",
    batch_size: int = 1024,
    normalize_embeddings: bool = True,
) -> None:
    texts = pd.read_csv(input_csv)[text_column].fillna("").tolist()
    payload = [f"{prefix}{t}" for t in texts]
    emb = model.encode(
        payload,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=normalize_embeddings,
        device=model.device,
    ).astype("float32")
    output_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_npy, emb)
    print(f"[Saved] {split_name} embeddings -> {output_npy}")


def load_docs_and_embeddings(
    split: str,
    data_dir: Path,
    embeddings_dir: Path,
) -> tuple[list[str], np.ndarray]:
    text_path = data_dir / f"data_combined_clean_{split}.csv"
    emb_path = embeddings_dir / f"embeddings_{split}.npy"

    docs = pd.read_csv(text_path)["deidentified_text"].fillna("").tolist()
    emb = np.load(emb_path).astype("float32")

    mask = np.array([len(t.strip()) > 0 for t in docs])
    docs = [t for t, keep in zip(docs, mask) if keep]
    emb = emb[mask]

    if emb.ndim != 2:
        raise ValueError(f"{split} embeddings must be 2D. Got {emb.shape}")
    if emb.shape[0] != len(docs):
        raise ValueError(f"{split} mismatch: {len(docs)} docs vs {emb.shape[0]} embeddings")
    return docs, emb

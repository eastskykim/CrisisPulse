from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pandas as pd


def _parse_representation(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = ast.literal_eval(text)
                if isinstance(parsed, list):
                    return [str(x).strip() for x in parsed if str(x).strip()]
            except Exception:
                pass
        return [w.strip() for w in text.split(",") if w.strip()]
    return []


def _topic_fingerprint(words: list[str], model_id: str, prompt_version: str = "v1") -> str:
    payload = json.dumps(
        {
            "model": model_id,
            "prompt_version": prompt_version,
            "words": words,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_cache(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_cache(path: Path, payload: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def _build_prompt(words: list[str]) -> str:
    word_text = ", ".join(words[:12])
    return (
        "You are labeling a BERTopic topic from social posts. "
        "Create one concise label (2-5 words), title case, no punctuation.\n\n"
        f"Top words: {word_text}\n\n"
        "Return only the label."
    )


def label_topics_with_llama_cpp(
    topic_info_df: pd.DataFrame,
    model_path: Path,
    cache_path: Path,
    min_count: int = 25,
    max_topics: int | None = None,
    n_gpu_layers: int = -1,
    n_ctx: int = 2048,
    temperature: float = 0.0,
    max_tokens: int = 24,
) -> pd.DataFrame:
    from llama_cpp import Llama

    work = topic_info_df.copy()
    if "Topic" not in work.columns or "Representation" not in work.columns:
        raise ValueError("topic_info_df must contain Topic and Representation columns")

    if "Count" in work.columns:
        work = work[(work["Topic"] != -1) & (work["Count"] >= min_count)].copy()
        work = work.sort_values("Count", ascending=False)
    else:
        work = work[work["Topic"] != -1].copy()

    if max_topics is not None:
        work = work.head(max_topics).copy()

    llm = Llama(
        model_path=str(model_path),
        chat_format="zephyr",
        n_gpu_layers=n_gpu_layers,
        n_ctx=n_ctx,
        verbose=False,
    )
    model_id = model_path.name

    cache = _load_cache(cache_path)
    generated = 0
    labels: list[str] = []

    for _, row in work.iterrows():
        words = _parse_representation(row["Representation"])
        fingerprint = _topic_fingerprint(words, model_id=model_id)
        cached = cache.get(fingerprint)
        if cached:
            labels.append(cached)
            continue

        prompt = _build_prompt(words)
        result = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": "You generate concise topic labels."},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        label = result["choices"][0]["message"]["content"].strip().splitlines()[0]
        cache[fingerprint] = label
        labels.append(label)
        generated += 1

    _save_cache(cache_path, cache)
    out = work.copy()
    out["llm_label"] = labels
    out.attrs["generated_count"] = generated
    out.attrs["cache_hits"] = len(out) - generated
    return out

from __future__ import annotations

import re
from collections import Counter

import numpy as np
from gensim import corpora
from gensim.models.coherencemodel import CoherenceModel


def build_coherence_inputs(model, docs: list[str], top_n: int, topic_ids: list[int] | None = None):
    vec = model.vectorizer_model
    if hasattr(vec, "build_analyzer"):
        analyze = vec.build_analyzer()
        tokens_per_doc = [analyze(d) for d in docs]
    elif hasattr(vec, "build_tokenizer"):
        tokenize = vec.build_tokenizer()
        tokens_per_doc = [tokenize(d) for d in docs]
    else:
        rx = re.compile(r"(?u)\b[a-z]{3,}\b")
        tokens_per_doc = [rx.findall(str(d).lower()) for d in docs]

    tokens_per_doc = [tokens for tokens in tokens_per_doc if tokens]
    if len(tokens_per_doc) < 5:
        return [], tokens_per_doc, None, None

    dictionary = corpora.Dictionary(tokens_per_doc)
    if len(dictionary) < 5:
        return [], tokens_per_doc, dictionary, []

    corpus = [dictionary.doc2bow(tokens) for tokens in tokens_per_doc]
    if topic_ids is None:
        tinfo = model.get_topic_info()
        topic_ids = [int(t) for t in tinfo["Topic"].tolist() if int(t) >= 0]

    topics_tokens = []
    for topic_id in topic_ids:
        topic_words = model.get_topic(topic_id) or []
        terms: list[str] = []
        for word, _ in topic_words[:top_n]:
            w = str(word).strip().lower()
            if w.isalpha() and len(w) >= 3 and w in dictionary.token2id:
                terms.append(w)
        if len(terms) >= 2:
            topics_tokens.append(terms)
    return topics_tokens, tokens_per_doc, dictionary, corpus


def coherence_cv(model, docs: list[str], top_n: int) -> float:
    topics_tokens, tokens_per_doc, dictionary, corpus = build_coherence_inputs(model, docs, top_n)
    if not topics_tokens or dictionary is None or not corpus:
        return 0.0
    try:
        cm = CoherenceModel(
            topics=topics_tokens,
            texts=tokens_per_doc,
            corpus=corpus,
            dictionary=dictionary,
            coherence="c_v",
            processes=1,
        )
        return float(cm.get_coherence())
    except Exception:
        return 0.0


def get_top_words(model, top_n: int) -> list[list[str]]:
    topics = model.get_topics()
    cleaned: list[list[str]] = []
    for topic_id, words in topics.items():
        if topic_id == -1 or not words:
            continue
        word_list = [str(w).strip().lower() for w, _ in words[:top_n] if str(w).isalpha()]
        if len(set(word_list)) >= 2:
            cleaned.append(word_list)
    return cleaned


def topic_diversity(top_words: list[list[str]], n: int) -> float:
    trimmed = [words[:n] for words in top_words if words]
    total = len(trimmed) * n if trimmed else 0
    unique_words = len({w for words in trimmed for w in words})
    return unique_words / max(1, total)


def outlier_rate(topics: list[int] | np.ndarray) -> float:
    return float(np.mean(np.asarray(topics) == -1))


def mega_cluster_share(topics: list[int] | np.ndarray) -> float:
    non_out = [t for t in topics if t != -1]
    if not non_out:
        return 1.0
    counts = Counter(non_out)
    return float(max(counts.values()) / sum(counts.values()))


def redundancy_jaccard(model, top_n: int) -> float:
    sets = [set(words) for words in get_top_words(model, top_n)]
    if len(sets) < 2:
        return 0.0

    def jacc(a: set[str], b: set[str]) -> float:
        return len(a & b) / max(1, len(a | b))

    sims = [jacc(a, b) for i, a in enumerate(sets) for b in sets[i + 1 :]]
    return float(np.mean(sims)) if sims else 0.0


def compute_score_j(
    coherence: float,
    diversity: float,
    outlier_blend: float,
    mega_share: float,
    w_coh: float = 0.35,
    w_div: float = 0.15,
    w_out: float = 0.20,
    w_mega: float = 0.30,
) -> float:
    return (
        (w_coh * coherence)
        + (w_div * diversity)
        - (w_out * outlier_blend)
        + (w_mega * (1 - mega_share))
    )

from __future__ import annotations

from dataclasses import dataclass

from incremental_bertopic_brand_crisis_detection.topic_model.compat import (
    ensure_llama_cpp_importable,
)

ensure_llama_cpp_importable()

from bertopic import BERTopic
from cuml.cluster import HDBSCAN
from cuml.manifold import UMAP
from sklearn.feature_extraction.text import CountVectorizer


@dataclass(frozen=True)
class TopicModelConfig:
    ngram: tuple[int, int]
    min_df: float
    max_df: float
    nn: int
    nc: int
    min_dist: float
    mcs: int
    csm: str


class TopicModelFactory:
    def __init__(
        self,
        representation_model,
        embedding_model,
        stop_words: list[str] | None = None,
        default_vectorizer=None,
    ):
        self.representation_model = representation_model
        self.embedding_model = embedding_model
        self.stop_words = stop_words or []
        self.default_vectorizer = default_vectorizer

    def build_model(self, config: TopicModelConfig, vectorizer_model=None) -> BERTopic:
        umap_model = UMAP(
            n_neighbors=config.nn,
            n_components=config.nc,
            min_dist=config.min_dist,
            metric="cosine",
            random_state=42,
        )
        hdbscan_model = HDBSCAN(
            min_cluster_size=config.mcs,
            min_samples=config.mcs,
            metric="euclidean",
            cluster_selection_method=config.csm,
            cluster_selection_epsilon=0,
            prediction_data=True,
        )
        if vectorizer_model is None:
            vectorizer_model = self.default_vectorizer or CountVectorizer(
                stop_words=self.stop_words,
                ngram_range=config.ngram,
                min_df=config.min_df,
                max_df=config.max_df,
                token_pattern=r"(?u)\b[a-z]{3,}\b",
            )

        return BERTopic(
            embedding_model=self.embedding_model,
            representation_model=self.representation_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model,
            nr_topics="auto",
            calculate_probabilities=False,
            verbose=False,
        )

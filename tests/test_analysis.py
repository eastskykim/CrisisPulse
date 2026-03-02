from __future__ import annotations

import pandas as pd

from incremental_bertopic_brand_crisis_detection.topic_model.analysis import (
    compare_topic_distributions,
    summarize_topic_stability_filtered,
)


def test_compare_topic_distributions_has_expected_columns() -> None:
    before = pd.DataFrame(
        {
            "topic_id": [0, 0, 1, 1],
            "sentiment": ["negative", "neutral", "positive", "neutral"],
            "brand": ["a", "a", "b", "b"],
        }
    )
    after = pd.DataFrame(
        {
            "topic_id": [0, 0, 2],
            "sentiment": ["negative", "negative", "positive"],
            "brand": ["a", "a", "b"],
        }
    )

    out = compare_topic_distributions(before, after)
    assert "n_before" in out.columns
    assert "n_after" in out.columns
    assert "n_delta" in out.columns
    assert "sent_delta_negative" in out.columns


def test_summarize_topic_stability_returns_metrics() -> None:
    df = pd.DataFrame(
        {
            "topic_id": [0, 1, 2],
            "n_before": [10, 12, 0],
            "n_after": [11, 8, 4],
            "n_delta": [1, -4, 4],
            "sent_before_negative": [0.4, 0.2, 0.0],
            "sent_before_neutral": [0.4, 0.4, 0.0],
            "sent_before_positive": [0.2, 0.4, 0.0],
            "sent_after_negative": [0.5, 0.3, 0.8],
            "sent_after_neutral": [0.3, 0.3, 0.1],
            "sent_after_positive": [0.2, 0.4, 0.1],
            "sent_delta_negative": [0.1, 0.1, 0.8],
            "sent_delta_neutral": [-0.1, -0.1, 0.1],
            "sent_delta_positive": [0.0, 0.0, 0.1],
            "brand_before_a": [0.7, 0.1, 0.0],
            "brand_before_b": [0.3, 0.9, 0.0],
            "brand_after_a": [0.6, 0.2, 0.0],
            "brand_after_b": [0.4, 0.8, 1.0],
            "brand_delta_a": [-0.1, 0.1, 0.0],
            "brand_delta_b": [0.1, -0.1, 1.0],
        }
    )

    metrics, persisting, new_topics = summarize_topic_stability_filtered(df)
    assert not metrics.empty
    assert len(persisting) == 2
    assert len(new_topics) == 1

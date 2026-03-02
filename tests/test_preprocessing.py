from __future__ import annotations

import pandas as pd

from incremental_bertopic_brand_crisis_detection.data.preprocessing import clean_text, prepare_df


def test_clean_text_removes_rt_prefix() -> None:
    assert clean_text("RT @user this is a retweet") == ""


def test_prepare_df_keeps_required_columns() -> None:
    df = pd.DataFrame(
        {
            "sprout_guid": ["1"],
            "created_time": ["2025-01-01"],
            "sentiment": ["negative"],
            "deidentified_text": ["This is a sample post with enough words"],
            "brand": ["marta"],
        }
    )
    out = prepare_df(df, min_tokens=3)
    assert list(out.columns) == [
        "sprout_guid",
        "created_time",
        "sentiment",
        "deidentified_text",
        "brand",
    ]
    assert len(out) == 1

from __future__ import annotations

import pandas as pd

from src.eval.recommendation_eval import evaluate_recommendations


def test_evaluate_recommendations_basic_metrics():
    split = pd.DataFrame(
        [
            {"ClientID": 1, "ProductID": 10, "label": 1},
            {"ClientID": 1, "ProductID": 11, "label": 0},
            {"ClientID": 1, "ProductID": 12, "label": 1},
            {"ClientID": 2, "ProductID": 20, "label": 0},
            {"ClientID": 2, "ProductID": 21, "label": 1},
            {"ClientID": 2, "ProductID": 22, "label": 0},
        ]
    )
    recs = pd.DataFrame(
        [
            {"ClientID": 1, "ProductID": 10, "rank": 1, "hybrid_score": 0.9},
            {"ClientID": 1, "ProductID": 11, "rank": 2, "hybrid_score": 0.8},
            {"ClientID": 2, "ProductID": 20, "rank": 1, "hybrid_score": 0.9},
            {"ClientID": 2, "ProductID": 21, "rank": 2, "hybrid_score": 0.8},
        ]
    )

    m = evaluate_recommendations(recs, split, k=2, score_col="hybrid_score")
    assert m["clients"] == 2.0
    assert 0.0 <= m["precision@2"] <= 1.0
    assert 0.0 <= m["recall@2"] <= 1.0
    assert 0.0 <= m["map@2"] <= 1.0
    assert 0.0 <= m["ndcg@2"] <= 1.0
    assert 0.0 <= m["hitrate@2"] <= 1.0


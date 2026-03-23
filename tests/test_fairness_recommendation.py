from __future__ import annotations

import pandas as pd

from src.fairness.audit import fairness_report
from src.fairness.rerank import rerank_for_exposure_balance


def _mock_recommendations() -> pd.DataFrame:
    # Group A mostly gets Cat1, Group B mostly gets Cat2 before fairness reranking.
    return pd.DataFrame(
        [
            {"ClientID": 1, "ProductID": 101, "Category": "Cat1", "hybrid_score": 0.95},
            {"ClientID": 1, "ProductID": 102, "Category": "Cat1", "hybrid_score": 0.90},
            {"ClientID": 1, "ProductID": 103, "Category": "Cat2", "hybrid_score": 0.30},
            {"ClientID": 2, "ProductID": 101, "Category": "Cat1", "hybrid_score": 0.94},
            {"ClientID": 2, "ProductID": 102, "Category": "Cat1", "hybrid_score": 0.88},
            {"ClientID": 2, "ProductID": 104, "Category": "Cat2", "hybrid_score": 0.29},
            {"ClientID": 3, "ProductID": 201, "Category": "Cat2", "hybrid_score": 0.96},
            {"ClientID": 3, "ProductID": 202, "Category": "Cat2", "hybrid_score": 0.89},
            {"ClientID": 3, "ProductID": 203, "Category": "Cat1", "hybrid_score": 0.35},
            {"ClientID": 4, "ProductID": 201, "Category": "Cat2", "hybrid_score": 0.93},
            {"ClientID": 4, "ProductID": 202, "Category": "Cat2", "hybrid_score": 0.87},
            {"ClientID": 4, "ProductID": 204, "Category": "Cat1", "hybrid_score": 0.33},
        ]
    )


def _mock_clients() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"ClientID": 1, "ClientGender": "A"},
            {"ClientID": 2, "ClientGender": "A"},
            {"ClientID": 3, "ClientGender": "B"},
            {"ClientID": 4, "ClientGender": "B"},
        ]
    )


def test_fairness_report_outputs_expected_shapes():
    recs = _mock_recommendations()
    clients = _mock_clients()
    exp, cat, summary = fairness_report(recs, clients, group_col="ClientGender")

    assert {"ClientGender", "exposure_rows", "exposure_share", "avg_score"}.issubset(exp.columns)
    assert {"ClientGender", "category_l1_distance"}.issubset(cat.columns)
    assert {"max_gap", "min_max_ratio", "mean_category_l1"}.issubset(summary.keys())
    assert len(exp) == 2


def test_reranking_reduces_category_parity_distance():
    recs = _mock_recommendations()
    clients = _mock_clients()

    # Baseline top-2 lists.
    base = (
        recs.sort_values(["ClientID", "hybrid_score"], ascending=[True, False])
        .groupby("ClientID", as_index=False)
        .head(2)
    )
    _, _, pre = fairness_report(base, clients, group_col="ClientGender")

    reranked = rerank_for_exposure_balance(
        recommendations=recs,
        client_groups=clients,
        group_col="ClientGender",
        top_k=2,
        lambda_fairness=1.2,
        score_col="hybrid_score",
        category_col="Category",
    )
    _, _, post = fairness_report(
        reranked,
        clients,
        group_col="ClientGender",
        score_col="fair_score",
    )

    assert post["mean_category_l1"] <= pre["mean_category_l1"]

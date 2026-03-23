from __future__ import annotations

import pandas as pd

from src.fairness.audit import attach_group_labels, category_exposure_matrix


def _category_lift_table(
    recommendations: pd.DataFrame,
    group_col: str,
    category_col: str,
) -> pd.DataFrame:
    """
    Build a per-(group, category) lift:
      lift = global_share(category) - group_share(category)
    Positive lift means the category is underexposed for that group.
    """
    mat = category_exposure_matrix(recommendations, group_col=group_col, category_col=category_col)
    if mat.empty:
        return pd.DataFrame(columns=[group_col, category_col, "lift"])

    global_share = (
        mat.groupby(category_col)["n"].sum()
        / max(float(mat["n"].sum()), 1.0)
    ).rename("global_share")
    g = mat.merge(global_share, on=category_col, how="left", validate="m:1")
    g["lift"] = g["global_share"] - g["exposure_share"]
    return g[[group_col, category_col, "lift"]]


def rerank_for_exposure_balance(
    recommendations: pd.DataFrame,
    client_groups: pd.DataFrame,
    group_col: str,
    top_k: int = 10,
    lambda_fairness: float = 0.25,
    score_col: str = "hybrid_score",
    category_col: str = "Category",
    eligible_groups: list[str] | None = None,
) -> pd.DataFrame:
    """
    Re-rank recommendations using a category-exposure fairness lift by group.
    The adjustment is:
      fair_score = hybrid_score + lambda_fairness * lift(group, category)
    """
    if "ClientID" not in recommendations.columns:
        raise ValueError("recommendations must contain ClientID")
    if "ProductID" not in recommendations.columns:
        raise ValueError("recommendations must contain ProductID")
    if score_col not in recommendations.columns:
        raise ValueError(f"recommendations must contain '{score_col}'")
    if category_col not in recommendations.columns:
        raise ValueError(f"recommendations must contain '{category_col}'")

    with_groups = attach_group_labels(recommendations, client_groups, group_col=group_col)
    top = (
        with_groups.sort_values(["ClientID", score_col], ascending=[True, False])
        .groupby("ClientID", as_index=False)
        .head(top_k)
        .copy()
    )
    if eligible_groups is not None:
        top = top[top[group_col].isin(set(eligible_groups))].copy()

    lifts = _category_lift_table(top, group_col=group_col, category_col=category_col)
    out = with_groups.merge(
        lifts,
        on=[group_col, category_col],
        how="left",
        validate="m:1",
    )
    out["lift"] = out["lift"].fillna(0.0)
    out["fair_score"] = out[score_col] + (float(lambda_fairness) * out["lift"])

    reranked = (
        out.sort_values(["ClientID", "fair_score"], ascending=[True, False])
        .groupby("ClientID", as_index=False)
        .head(top_k)
        .copy()
    )
    reranked["rank"] = reranked.groupby("ClientID").cumcount() + 1
    return reranked

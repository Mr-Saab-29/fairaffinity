from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def attach_group_labels(
    recommendations: pd.DataFrame,
    client_groups: pd.DataFrame,
    group_col: str,
) -> pd.DataFrame:
    """Attach demographic group labels to recommendation rows."""
    if "ClientID" not in recommendations.columns:
        raise ValueError("recommendations must contain ClientID")
    if group_col in recommendations.columns:
        out = recommendations.copy()
        out[group_col] = out[group_col].fillna("UNKNOWN")
        return out
    if "ClientID" not in client_groups.columns:
        raise ValueError("client_groups must contain ClientID")
    if group_col not in client_groups.columns:
        raise ValueError(f"client_groups must contain group column '{group_col}'")

    out = recommendations.merge(
        client_groups[["ClientID", group_col]].drop_duplicates("ClientID"),
        on="ClientID",
        how="left",
        validate="m:1",
    )
    out[group_col] = out[group_col].fillna("UNKNOWN")
    return out


def _safe_ratio(numer: float, denom: float, default: float = 0.0) -> float:
    if denom == 0:
        return default
    return float(numer / denom)


def exposure_by_group(
    recommendations: pd.DataFrame,
    group_col: str,
    item_col: str = "ProductID",
    score_col: str = "hybrid_score",
) -> pd.DataFrame:
    """
    Compute exposure and quality proxies by demographic group.
    Exposure here is row-count share in top-k recommendations.
    """
    if group_col not in recommendations.columns:
        raise ValueError(f"recommendations must contain '{group_col}'")

    total_rows = max(len(recommendations), 1)
    g = (
        recommendations.groupby(group_col, dropna=False)
        .agg(
            exposure_rows=(item_col, "count"),
            unique_items=(item_col, "nunique"),
            avg_score=(score_col, "mean"),
            median_score=(score_col, "median"),
            clients=("ClientID", "nunique"),
        )
        .reset_index()
        .sort_values(group_col)
    )
    g["exposure_share"] = g["exposure_rows"] / total_rows
    g["rows_per_client"] = g.apply(
        lambda r: _safe_ratio(float(r["exposure_rows"]), float(r["clients"]), default=0.0),
        axis=1,
    )
    return g


def category_exposure_matrix(
    recommendations: pd.DataFrame,
    group_col: str,
    category_col: str = "Category",
) -> pd.DataFrame:
    """Return group x category exposure-share table."""
    if group_col not in recommendations.columns:
        raise ValueError(f"recommendations must contain '{group_col}'")
    if category_col not in recommendations.columns:
        raise ValueError(f"recommendations must contain '{category_col}'")

    mat = (
        recommendations.groupby([group_col, category_col], dropna=False)
        .size()
        .rename("n")
        .reset_index()
    )
    totals = mat.groupby(group_col, dropna=False)["n"].sum().rename("group_total")
    mat = mat.merge(totals, on=group_col, how="left", validate="m:1")
    mat["exposure_share"] = mat["n"] / mat["group_total"].replace(0, np.nan)
    mat["exposure_share"] = mat["exposure_share"].fillna(0.0)
    return mat


def parity_metrics(group_exposure: pd.DataFrame, share_col: str = "exposure_share") -> Dict[str, float]:
    """Simple disparity metrics across groups."""
    if group_exposure.empty:
        return {"max_gap": 0.0, "min_max_ratio": 1.0}
    vals = group_exposure[share_col].astype(float).to_numpy()
    mx = float(np.max(vals))
    mn = float(np.min(vals))
    return {
        "max_gap": float(mx - mn),
        "min_max_ratio": _safe_ratio(mn, mx, default=0.0 if mx > 0 else 1.0),
    }


def category_parity_distance(
    category_exposure: pd.DataFrame,
    group_col: str,
    category_col: str = "Category",
    share_col: str = "exposure_share",
) -> pd.DataFrame:
    """
    For each group, compute L1 distance to global category exposure distribution.
    Lower means category exposure is closer to overall baseline.
    """
    if category_exposure.empty:
        return pd.DataFrame(columns=[group_col, "category_l1_distance"])

    global_dist = (
        category_exposure.groupby(category_col)["n"].sum()
        / max(float(category_exposure["n"].sum()), 1.0)
    )
    rows = []
    for grp, gdf in category_exposure.groupby(group_col, dropna=False):
        dist = gdf.set_index(category_col)[share_col]
        aligned = pd.DataFrame({"global": global_dist, "group": dist}).fillna(0.0)
        l1 = float(np.abs(aligned["global"] - aligned["group"]).sum())
        rows.append({group_col: grp, "category_l1_distance": l1})
    return pd.DataFrame(rows)


def fairness_report(
    recommendations: pd.DataFrame,
    client_groups: pd.DataFrame,
    group_col: str,
    item_col: str = "ProductID",
    category_col: str = "Category",
    score_col: str = "hybrid_score",
) -> tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    """
    Return:
      1) exposure_by_group dataframe
      2) category parity dataframe (L1 distances by group)
      3) summary dict with key fairness metrics
    """
    with_groups = attach_group_labels(recommendations, client_groups, group_col=group_col)
    exp = exposure_by_group(
        with_groups, group_col=group_col, item_col=item_col, score_col=score_col
    )
    cat_matrix = category_exposure_matrix(
        with_groups, group_col=group_col, category_col=category_col
    )
    cat_parity = category_parity_distance(cat_matrix, group_col=group_col, category_col=category_col)
    summary = parity_metrics(exp, share_col="exposure_share")
    summary["mean_category_l1"] = float(cat_parity["category_l1_distance"].mean()) if not cat_parity.empty else 0.0
    return exp, cat_parity, summary

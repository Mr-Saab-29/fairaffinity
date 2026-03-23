from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from joblib import load

from src.fairness.audit import fairness_report
from src.fairness.rerank import rerank_for_exposure_balance
from src.utils.io_helpers import load_interactions

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
REPORTS = PROC / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)


def _load_split(name: str) -> pd.DataFrame:
    path = PROC / "model" / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Split file not found: {path}")
    return pd.read_parquet(path)


def _load_ranked_features() -> list[str] | None:
    path = REPORTS / "feature_importance_lgbm.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "feature" not in df.columns or df.empty:
        return None
    return df["feature"].tolist()


def _load_best_k(default: int | None = None) -> int | None:
    path = REPORTS / "feature_selection_best_k.json"
    if not path.exists():
        return default
    try:
        meta = json.loads(path.read_text())
        k = int(meta.get("best_k", 0))
        return k if k > 0 else default
    except Exception:
        return default


def _select_features(df: pd.DataFrame) -> list[str]:
    ranked = _load_ranked_features()
    best_k = _load_best_k()
    if ranked and best_k:
        cols = [c for c in ranked[:best_k] if c in df.columns]
        if cols:
            return cols
    if ranked:
        cols = [c for c in ranked if c in df.columns]
        if cols:
            return cols
    drop = {
        "ClientID",
        "ProductID",
        "StoreID",
        "label",
        "target",
        "SaleTransactionDate",
        "txn_date",
    }
    return [c for c in df.columns if c not in drop and pd.api.types.is_numeric_dtype(df[c])]


@dataclass
class HybridWeights:
    affinity: float = 0.60
    collaborative: float = 0.20
    content: float = 0.15
    popularity: float = 0.05


def _safe_normalize(series: pd.Series) -> pd.Series:
    lo = float(series.min()) if len(series) else 0.0
    hi = float(series.max()) if len(series) else 0.0
    if hi <= lo:
        return pd.Series(np.zeros(len(series)), index=series.index, dtype=float)
    return (series - lo) / (hi - lo)


def _load_split_cutoff(split_name: str) -> pd.Timestamp | None:
    p = REPORTS / "label_sampling_summary.csv"
    if not p.exists():
        return None
    rep = pd.read_csv(p)
    rep = rep[rep["split"] == split_name]
    if rep.empty:
        return None
    return pd.to_datetime(rep["cutoff"].iloc[0], errors="coerce")


def _client_group_frame(interactions: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in ["ClientID", "ClientGender", "Age", "ClientCountry"] if c in interactions.columns]
    if "ClientID" not in cols:
        raise ValueError("interactions missing ClientID")
    out = interactions[cols].drop_duplicates("ClientID").copy()
    if "ClientGender" in out.columns:
        # Normalize compact code used in source data.
        out["ClientGender"] = out["ClientGender"].replace({"U": "Unisex"})
    return out


def _history_for_split(split_name: str) -> pd.DataFrame:
    history = load_interactions()
    cutoff = _load_split_cutoff(split_name)
    if cutoff is not None and "txn_date" in history.columns:
        history = history[history["txn_date"] <= cutoff].copy()
    return history


def _collaborative_signal(candidates: pd.DataFrame, history: pd.DataFrame) -> pd.Series:
    """
    Lightweight collaborative proxy:
      weighted product popularity among all interactions.
    """
    pop = history.groupby("ProductID")["ClientID"].count().rename("hist_txn")
    c = candidates.merge(pop, on="ProductID", how="left")
    c["hist_txn"] = c["hist_txn"].fillna(0.0)
    return _safe_normalize(np.log1p(c["hist_txn"]))


def _content_signal(candidates: pd.DataFrame, history: pd.DataFrame) -> pd.Series:
    """
    Content proxy:
      client-level preference probability for candidate Category from history.
    """
    if "Category" not in candidates.columns or "Category" not in history.columns:
        return pd.Series(np.zeros(len(candidates)), index=candidates.index, dtype=float)

    c_hist = (
        history.dropna(subset=["Category"])
        .groupby(["ClientID", "Category"])
        .size()
        .rename("n")
        .reset_index()
    )
    totals = c_hist.groupby("ClientID")["n"].sum().rename("n_total")
    c_hist = c_hist.merge(totals, on="ClientID", how="left", validate="m:1")
    c_hist["pref_p"] = c_hist["n"] / c_hist["n_total"].replace(0, np.nan)
    c_hist["pref_p"] = c_hist["pref_p"].fillna(0.0)

    out = candidates.merge(
        c_hist[["ClientID", "Category", "pref_p"]],
        on=["ClientID", "Category"],
        how="left",
    )
    return out["pref_p"].fillna(0.0)


def _popularity_signal(candidates: pd.DataFrame, history: pd.DataFrame) -> pd.Series:
    buyers = history.groupby("ProductID")["ClientID"].nunique().rename("hist_buyers")
    c = candidates.merge(buyers, on="ProductID", how="left")
    c["hist_buyers"] = c["hist_buyers"].fillna(0.0)
    return _safe_normalize(np.log1p(c["hist_buyers"]))


def _exclude_seen_pairs(candidates: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    seen = history[["ClientID", "ProductID"]].drop_duplicates()
    seen["seen"] = 1
    out = candidates.merge(seen, on=["ClientID", "ProductID"], how="left")
    out = out[out["seen"].fillna(0) == 0].drop(columns=["seen"])
    return out


def _fallback_affinity_score(split: pd.DataFrame) -> pd.Series:
    """
    Dependency-free affinity proxy used when model loading is unavailable.
    """
    cp_txn = split["cp_cp_txns"] if "cp_cp_txns" in split.columns else pd.Series(0.0, index=split.index)
    pop_txn = split["p_txns"] if "p_txns" in split.columns else pd.Series(0.0, index=split.index)
    recency = split["cp_days_since_last_cp"] if "cp_days_since_last_cp" in split.columns else pd.Series(9999.0, index=split.index)
    affinity = (
        (0.55 * _safe_normalize(np.log1p(cp_txn)))
        + (0.35 * _safe_normalize(np.log1p(pop_txn)))
        + (0.10 * (1.0 - _safe_normalize(recency)))
    )
    return affinity.fillna(0.0)


def score_hybrid_candidates(
    split_name: str,
    model_path: Path,
    weights: HybridWeights = HybridWeights(),
    exclude_seen: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Score split candidates with:
      hybrid = w1*affinity + w2*collaborative + w3*content + w4*popularity.
    Returns scored candidates and client demographic frame.
    """
    split = _load_split(split_name).copy()
    try:
        model = load(model_path)
        features = _select_features(split)
        split["affinity_score"] = model.predict_proba(split[features])[:, 1]
    except ModuleNotFoundError:
        # Keep inference usable in lightweight envs where training dependencies are absent.
        split["affinity_score"] = _fallback_affinity_score(split)

    history = _history_for_split(split_name)
    if exclude_seen:
        split = _exclude_seen_pairs(split, history)

    split["collab_score"] = _collaborative_signal(split, history)
    split["content_score"] = _content_signal(split, history)
    split["popularity_score"] = _popularity_signal(split, history)
    split["hybrid_score"] = (
        (weights.affinity * split["affinity_score"])
        + (weights.collaborative * split["collab_score"])
        + (weights.content * split["content_score"])
        + (weights.popularity * split["popularity_score"])
    )

    client_groups = _client_group_frame(history)
    return split, client_groups


def top_k_recommendations(scored: pd.DataFrame, top_k: int = 10) -> pd.DataFrame:
    out = (
        scored.sort_values(["ClientID", "hybrid_score"], ascending=[True, False])
        .groupby("ClientID", as_index=False)
        .head(top_k)
        .copy()
    )
    out["rank"] = out.groupby("ClientID").cumcount() + 1
    return out


def run_recommendation_pipeline(
    split_name: str,
    model_path: Path,
    top_k: int = 10,
    group_col: str = "ClientGender",
    lambda_fairness: float = 0.25,
    fairness_groups: list[str] | None = None,
    weights: HybridWeights = HybridWeights(),
    output_prefix: str | None = None,
) -> Dict[str, object]:
    scored, client_groups = score_hybrid_candidates(
        split_name=split_name,
        model_path=model_path,
        weights=weights,
        exclude_seen=True,
    )
    base = top_k_recommendations(scored, top_k=top_k)
    reranked = rerank_for_exposure_balance(
        recommendations=scored,
        client_groups=client_groups,
        group_col=group_col,
        top_k=top_k,
        lambda_fairness=lambda_fairness,
        score_col="hybrid_score",
        category_col="Category",
        eligible_groups=fairness_groups,
    )

    exp_pre, cat_pre, sum_pre = fairness_report(
        recommendations=base,
        client_groups=client_groups,
        group_col=group_col,
        item_col="ProductID",
        category_col="Category",
        score_col="hybrid_score",
    )
    exp_post, cat_post, sum_post = fairness_report(
        recommendations=reranked,
        client_groups=client_groups,
        group_col=group_col,
        item_col="ProductID",
        category_col="Category",
        score_col="fair_score",
    )
    # Focused fairness view (used for decisioning) while preserving full reporting above.
    if fairness_groups is not None:
        base_focus = base.merge(
            client_groups[["ClientID", group_col]],
            on="ClientID",
            how="left",
            validate="m:1",
        )
        base_focus = base_focus[base_focus[group_col].isin(set(fairness_groups))].copy()
        rerank_focus = reranked[reranked[group_col].isin(set(fairness_groups))].copy()

        _, _, sum_pre_focus = fairness_report(
            recommendations=base_focus,
            client_groups=client_groups,
            group_col=group_col,
            item_col="ProductID",
            category_col="Category",
            score_col="hybrid_score",
        )
        _, _, sum_post_focus = fairness_report(
            recommendations=rerank_focus,
            client_groups=client_groups,
            group_col=group_col,
            item_col="ProductID",
            category_col="Category",
            score_col="fair_score",
        )
    else:
        sum_pre_focus = None
        sum_post_focus = None

    stem = output_prefix if output_prefix else f"reco_{split_name}"
    out_dir = REPORTS / "recommendations"
    out_dir.mkdir(parents=True, exist_ok=True)

    base_path = out_dir / f"{stem}_base.parquet"
    fair_path = out_dir / f"{stem}_fair.parquet"
    exp_pre_path = out_dir / f"{stem}_fairness_pre.csv"
    exp_post_path = out_dir / f"{stem}_fairness_post.csv"
    cat_pre_path = out_dir / f"{stem}_category_parity_pre.csv"
    cat_post_path = out_dir / f"{stem}_category_parity_post.csv"
    summary_path = out_dir / f"{stem}_fairness_summary.json"

    base.to_parquet(base_path, index=False)
    reranked.to_parquet(fair_path, index=False)
    exp_pre.to_csv(exp_pre_path, index=False)
    exp_post.to_csv(exp_post_path, index=False)
    cat_pre.to_csv(cat_pre_path, index=False)
    cat_post.to_csv(cat_post_path, index=False)
    summary_payload: Dict[str, object] = {
        "pre": sum_pre,
        "post": sum_post,
        "group_col": group_col,
        "top_k": top_k,
    }
    if fairness_groups is not None:
        summary_payload["fairness_groups"] = fairness_groups
        summary_payload["focus_pre"] = sum_pre_focus
        summary_payload["focus_post"] = sum_post_focus
    summary_path.write_text(json.dumps(summary_payload, indent=2))

    return {
        "base_path": str(base_path),
        "fair_path": str(fair_path),
        "exp_pre_path": str(exp_pre_path),
        "exp_post_path": str(exp_post_path),
        "cat_pre_path": str(cat_pre_path),
        "cat_post_path": str(cat_post_path),
        "summary_path": str(summary_path),
        "summary_pre": sum_pre,
        "summary_post": sum_post,
    }

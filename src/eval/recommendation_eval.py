from __future__ import annotations

from pathlib import Path
from typing import Dict

import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
REPORTS = PROC / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)


def _load_labels(split_path: Path) -> pd.DataFrame:
    if not split_path.exists():
        raise FileNotFoundError(f"Missing split file: {split_path}")
    split = pd.read_parquet(split_path, columns=["ClientID", "ProductID", "label"])
    split["label"] = split["label"].astype(int)
    return split


def _prepare_recommendations(
    recommendations: pd.DataFrame,
    labels: pd.DataFrame,
    k: int,
    score_col: str,
) -> pd.DataFrame:
    req = {"ClientID", "ProductID"}
    missing = req - set(recommendations.columns)
    if missing:
        raise ValueError(f"recommendations missing columns: {sorted(missing)}")

    df = recommendations.copy()
    if "label" not in df.columns:
        df = df.merge(labels, on=["ClientID", "ProductID"], how="left", validate="m:1")
    df["label"] = df["label"].fillna(0).astype(int)

    if "rank" in df.columns:
        df = df.sort_values(["ClientID", "rank"], ascending=[True, True])
    elif score_col in df.columns:
        df = df.sort_values(["ClientID", score_col], ascending=[True, False])
    else:
        raise ValueError(f"recommendations must contain either 'rank' or '{score_col}'")

    return df.groupby("ClientID", as_index=False).head(k).copy()


def _ap_at_k(labels: np.ndarray, denom_pos: int, k: int) -> float:
    if denom_pos <= 0:
        return np.nan
    hits = 0
    acc = 0.0
    for i, y in enumerate(labels[:k], start=1):
        if y == 1:
            hits += 1
            acc += hits / i
    denom = min(k, denom_pos)
    if denom == 0:
        return np.nan
    return float(acc / denom)


def _ndcg_at_k(labels: np.ndarray, denom_pos: int, k: int) -> float:
    if denom_pos <= 0:
        return np.nan
    gains = labels[:k].astype(float)
    discounts = 1.0 / np.log2(np.arange(2, len(gains) + 2))
    dcg = float((gains * discounts).sum())
    ideal_len = min(k, denom_pos)
    ideal = np.ones(ideal_len, dtype=float)
    idcg = float((ideal * (1.0 / np.log2(np.arange(2, ideal_len + 2)))).sum())
    if idcg == 0:
        return np.nan
    return dcg / idcg


def evaluate_recommendations(
    recommendations: pd.DataFrame,
    split_labels: pd.DataFrame,
    k: int = 10,
    score_col: str = "hybrid_score",
) -> Dict[str, float]:
    top = _prepare_recommendations(recommendations, split_labels, k=k, score_col=score_col)
    pos_counts = (
        split_labels[split_labels["label"] == 1]
        .groupby("ClientID")
        .size()
        .rename("pos_total")
    )

    rows = []
    for cid, gdf in top.groupby("ClientID", sort=False):
        labels = gdf["label"].to_numpy(dtype=int)
        hits = int(labels.sum())
        pos_total = int(pos_counts.get(cid, 0))
        rows.append(
            {
                "ClientID": cid,
                "hits": hits,
                "precision": hits / float(k),
                "recall": (hits / float(pos_total)) if pos_total > 0 else np.nan,
                "map": _ap_at_k(labels, denom_pos=pos_total, k=k),
                "ndcg": _ndcg_at_k(labels, denom_pos=pos_total, k=k),
                "hitrate": 1.0 if hits > 0 else 0.0,
                "has_positive": 1 if pos_total > 0 else 0,
            }
        )

    if not rows:
        return {
            "clients": 0.0,
            f"precision@{k}": 0.0,
            f"recall@{k}": 0.0,
            f"map@{k}": 0.0,
            f"ndcg@{k}": 0.0,
            f"hitrate@{k}": 0.0,
        }

    per_client = pd.DataFrame(rows)
    mask = per_client["has_positive"] == 1

    return {
        "clients": float(len(per_client)),
        f"precision@{k}": float(per_client["precision"].mean()),
        f"recall@{k}": float(per_client.loc[mask, "recall"].mean()) if mask.any() else 0.0,
        f"map@{k}": float(per_client.loc[mask, "map"].mean()) if mask.any() else 0.0,
        f"ndcg@{k}": float(per_client.loc[mask, "ndcg"].mean()) if mask.any() else 0.0,
        f"hitrate@{k}": float(per_client["hitrate"].mean()),
    }


def compare_base_vs_fair(
    base_path: Path,
    fair_path: Path,
    split_path: Path,
    k: int = 10,
) -> Dict[str, Dict[str, float]]:
    base = pd.read_parquet(base_path)
    fair = pd.read_parquet(fair_path)
    labels = _load_labels(split_path)

    base_metrics = evaluate_recommendations(base, labels, k=k, score_col="hybrid_score")
    fair_metrics = evaluate_recommendations(fair, labels, k=k, score_col="fair_score")

    delta: Dict[str, float] = {}
    for metric, v_base in base_metrics.items():
        if metric == "clients":
            continue
        delta[metric] = float(fair_metrics[metric] - v_base)

    return {"base": base_metrics, "fair": fair_metrics, "delta": delta}


def save_comparison_report(
    comparison: Dict[str, Dict[str, float]],
    output_dir: Path,
    stem: str,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{stem}_offline_eval.json"
    csv_path = output_dir / f"{stem}_offline_eval.csv"

    json_path.write_text(json.dumps(comparison, indent=2))

    keys = [k for k in comparison["base"].keys() if k != "clients"]
    rows = []
    for k in keys:
        rows.append(
            {
                "metric": k,
                "base": comparison["base"][k],
                "fair": comparison["fair"][k],
                "delta_fair_minus_base": comparison["delta"][k],
            }
        )
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return json_path, csv_path


from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from src.utils.io_helpers import load_interactions
from src.utils.dates import normalize_txn_date

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
MODEL_DIR = PROC / "model"
REPORTS = PROC / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)

@dataclass
class SplitSpec:
    name : str
    cutoff: pd.Timestamp
    label_days : int
    pos_target: str # 'Binary' | 'count_txn' | 'sum_qty'

def _history_and_window(
        full: pd.DataFrame, cutoff: pd.Timestamp, label_days: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split full data into history and window based on cutoff and label days."""
    df = full.copy()
    df['txn_date'] = normalize_txn_date(df['txn_date'])
    history = df[df['txn_date'] < cutoff]
    hi = cutoff + pd.Timedelta(days=label_days)
    window = df[(df['txn_date'] >= cutoff) & (df['txn_date'] <= hi)]
    return history, window

def _presence_and_counts(split_path: Path) -> Dict[str, float]:
    if not split_path.exists():
        return {"exists": False}

    ds = pd.read_parquet(split_path)
    out = {
        "exists": True,
        "rows": float(len(ds)),
        "clients": float(ds["ClientID"].nunique()),
        "products": float(ds["ProductID"].nunique()),
        "positives": float((ds["label"] == 1).sum()),
        "negatives": float((ds["label"] == 0).sum()),
    }
    denom = out["rows"] if out["rows"] else 1.0
    out["pos_ratio"] = out["positives"] / denom
    return out

def _check_positive_membership(window: pd.DataFrame, ds: pd.DataFrame) -> Dict[str, float]:
    """All ds positives must be subset of window (ClientID, ProductID)."""
    pos_pairs = set(map(tuple, ds.loc[ds["label"] == 1, ["ClientID", "ProductID"]].to_numpy()))
    win_pairs = set(map(tuple, window[["ClientID", "ProductID"]].drop_duplicates().to_numpy()))
    extra = pos_pairs - win_pairs
    return {
        "pos_in_window_ok": len(extra) == 0,
        "pos_not_in_window": float(len(extra)),
    }
def _check_negative_not_seen(history: pd.DataFrame, ds: pd.DataFrame, sample_clients: int = 2000) -> Dict[str, float]:
    """
    For a sample of clients (or all if small), ensure negatives are not in history for that client.
    """
    neg = ds.loc[ds["label"] == 0, ["ClientID", "ProductID"]]
    if neg.empty:
        return {"neg_not_seen_ok": True, "neg_seen_violations": 0.0, "clients_checked": 0.0}

    clients = neg["ClientID"].unique()
    if sample_clients and sample_clients < len(clients):
        rng = np.random.default_rng(42)
        clients = rng.choice(clients, size=sample_clients, replace=False)

    # build seen map
    seen_map = history.groupby("ClientID")["ProductID"].unique().to_dict()

    violations = 0
    checked = 0
    for cid in clients:
        seen = set(seen_map.get(cid, []))
        cand = neg.loc[neg["ClientID"] == cid, "ProductID"].unique().tolist()
        if not cand:
            continue
        checked += 1
        overlap = seen.intersection(cand)
        violations += len(overlap)

    return {
        "neg_not_seen_ok": violations == 0,
        "neg_seen_violations": float(violations),
        "clients_checked": float(checked),
    }

def _check_target_consistency(window: pd.DataFrame, ds: pd.DataFrame, pos_target: str) -> Dict[str, float]:
    """
    For positives:
      - binary   : target should be 1
      - count_txn: target should equal #txns in window for that pair
      - sum_qty  : target should equal sum(Quantity) in window for that pair
    Negatives: target should be 0 (already enforced in builder).
    """
    out = {}
    pos = ds.loc[ds["label"] == 1, ["ClientID", "ProductID", "target"]].copy()

    if pos.empty:
        return {"target_check_ok": True, "target_mismatch_pairs": 0.0}

    if pos_target == "binary":
        mism = (pos["target"] != 1).sum()
        out["target_check_ok"] = mism == 0
        out["target_mismatch_pairs"] = float(mism)
        return out

    if pos_target == "count_txn":
        agg = window.groupby(["ClientID", "ProductID"])["txn_date"].count().rename("true_target").reset_index()
    else:  # "sum_qty"
        agg = window.groupby(["ClientID", "ProductID"])["Quantity"].sum().rename("true_target").reset_index()

    chk = pos.merge(agg, on=["ClientID", "ProductID"], how="left")
    chk["true_target"] = chk["true_target"].fillna(0)
    mism = (chk["target"] != chk["true_target"]).sum()
    out["target_check_ok"] = mism == 0
    out["target_mismatch_pairs"] = float(mism)
    return out

def _feature_null_coverage(ds: pd.DataFrame) -> Dict[str, float]:
    """
    Quick feature coverage by block:
      - u_*  user features
      - p_*  product features
      - cp_* client-product recency features
      - cat_* category features (if present)
    Returns share of nulls across each block (averaged).
    """
    out: Dict[str, float] = {}
    blocks = {
        "u": [c for c in ds.columns if c.startswith("u_")],
        "p": [c for c in ds.columns if c.startswith("p_")],
        "cp": [c for c in ds.columns if c.startswith("cp_")],
        "cat": [c for c in ds.columns if c.startswith("cat_")],
    }
    for k, cols in blocks.items():
        if not cols:
            out[f"{k}_null_share"] = 0.0
            continue
        null_share = ds[cols].isna().mean().mean()
        out[f"{k}_null_share"] = float(null_share)
    return out

def validate_split(spec: SplitSpec) -> Dict[str, float]:
    """
    Validate one split and return a flat metrics dict.
    """
    path = MODEL_DIR / f"{spec.name}.parquet"
    basic = _presence_and_counts(path)
    result = {"split": spec.name, "cutoff": str(spec.cutoff.date()), "label_days": float(spec.label_days),
              "pos_target": spec.pos_target}

    # If file missing, return early with existence only
    result.update(basic)
    if not basic.get("exists", False):
        return result

    # Load ds + source windows
    ds = pd.read_parquet(path)
    full = load_interactions()
    history, window = _history_and_window(full, spec.cutoff, spec.label_days)

    # core checks
    result.update(_check_positive_membership(window, ds))
    result.update(_check_negative_not_seen(history, ds, sample_clients=2000))
    result.update(_check_target_consistency(window, ds, spec.pos_target))
    result.update(_feature_null_coverage(ds))

    # optional: uniqueness check on pairs (should not be duped)
    dup_pairs = ds.duplicated(subset=["ClientID", "ProductID"], keep=False).sum()
    result["pair_unique_ok"] = dup_pairs == 0
    result["pair_duplicates"] = float(dup_pairs)

    return result

def main() -> None:
    ap = argparse.ArgumentParser(description="Validate sampled train/val/test splits.")
    ap.add_argument("--train-end", required=True)
    ap.add_argument("--val-end", required=True)
    ap.add_argument("--test-end", required=True)
    ap.add_argument("--label-days", type=int, default=30)
    ap.add_argument("--pos-target", type=str, choices=["binary", "count_txn", "sum_qty"], default="binary")
    args = ap.parse_args()

    specs = [
        SplitSpec("train", pd.to_datetime(args.train_end), args.label_days, args.pos_target),
        SplitSpec("val", pd.to_datetime(args.val_end), args.label_days, args.pos_target),
        SplitSpec("test", pd.to_datetime(args.test_end), args.label_days, args.pos_target),
    ]

    rows = []
    for spec in specs:
        print(f"[validate] split={spec.name} cutoff={spec.cutoff.date()} Δ={spec.label_days} target={spec.pos_target}")
        rows.append(validate_split(spec))

    rep = pd.DataFrame(rows)
    out = REPORTS / "label_sampling_validation.csv"
    rep.to_csv(out, index=False)
    print(f"[OK] wrote {out}\n{rep}")

if __name__ == "__main__":
    main()
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from src.utils.io_helpers import load_interactions
from src.utils.dates import normalize_txn_date

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
MODEL_DIR = PROC / "model"
OUTDIR = PROC / "reports"
OUTDIR.mkdir(parents=True, exist_ok=True)

def history_and_window(full: pd.DataFrame, cutoff: pd.Timestamp, label_days: int):
    df = full.copy()
    df["txn_date"] = normalize_txn_date(df["txn_date"])
    history = df[df["txn_date"] <= cutoff]
    hi = cutoff + pd.Timedelta(days=label_days)
    window = df[(df["txn_date"] > cutoff) & (df["txn_date"] <= hi)]
    return history, window

def recompute_true_targets(window: pd.DataFrame, pos_target: str) -> pd.DataFrame:
    if pos_target == "binary":
        g = window[["ClientID","ProductID"]].drop_duplicates()
        g["true_target"] = 1
        return g
    if pos_target == "count_txn":
        return (window
                .groupby(["ClientID","ProductID"])["txn_date"]
                .count().rename("true_target").reset_index())
    # sum_qty
    return (window
            .groupby(["ClientID","ProductID"])["Quantity"]
            .sum().rename("true_target").reset_index())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["train","val","test"])
    ap.add_argument("--cutoff", required=True)
    ap.add_argument("--label-days", type=int, default=30)
    ap.add_argument("--pos-target", choices=["binary","count_txn","sum_qty"], default="count_txn")
    args = ap.parse_args()

    split_path = MODEL_DIR / f"{args.split}.parquet"
    if not split_path.exists():
        raise SystemExit(f"Missing file: {split_path}")

    ds = pd.read_parquet(split_path)
    full = load_interactions()
    _, window = history_and_window(full, pd.to_datetime(args.cutoff), args.label_days)

    # only positives need checking
    pos = ds.loc[ds["label"] == 1, ["ClientID","ProductID","target"]].copy()
    truth = recompute_true_targets(window, args.pos_target)

    chk = pos.merge(truth, on=["ClientID","ProductID"], how="left")
    chk["true_target"] = chk["true_target"].fillna(0)

    mism = chk[chk["target"] != chk["true_target"]].copy()
    print(f"[{args.split}] positives={len(pos):,} | mismatches={len(mism):,}")

    out = OUTDIR / f"target_mismatches_{args.split}.csv"
    mism.to_csv(out, index=False)
    print(f"[OK] wrote {out}")

if __name__ == "__main__":
    main()
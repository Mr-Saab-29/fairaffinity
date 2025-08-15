from __future__ import annotations
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
from src.utils.io_helpers import load_interactions
from src.utils.dates import apply_cutoff

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"

def build_user_rfm(cutoff : str | None = None) -> pd.DataFrame:
    """Build user-level RFM features from interactions data.

    Args:
        cutoff (str | None, optional): Cutoff date. Defaults to None.

    Returns:
        pd.DataFrame: User RFM features DataFrame.
    """
    df = load_interactions()
    df = apply_cutoff(df, cutoff)
    ref_date = pd.to_datetime(cutoff) if cutoff else df["txn_date"].max()

    g = df.groupby("ClientID", as_index=False).agg(
        freq_txns = ("txn_date", "count"),
        qty_sum = ("Quantity", "sum"),
        eur_sum = ("SalesNetAmountEuro", "sum"),
        first_dt = ("txn_date", "min"),
        last_dt = ("txn_date", "max"),
    )
    g["recency"] = (ref_date - g["last_dt"]).dt.days.clip(lower=0)
    g['tenure_days'] = (g['last_dt'] - g['first_dt']).dt.days.clip(lower=0)
    g['avg_basket_eur'] = (g['eur_sum'] / g['freq_txns']).replace([np.inf, -np.inf], np.nan)

    out = g.drop(columns=["first_dt", "last_dt"]).sort_values("ClientID")
    fname = "user_rfm.parquet" if not cutoff else f"user_rfm_{ref_date.date()}.parquet"
    path = PROC / fname
    out.to_parquet(path, index=False)
    print(f"[OK] user_rfm -> {path} | rows={len(out):,}")
    return out

def build_client_product_recency(cutoff: str | None = None) -> pd.DataFrame:
    df = load_interactions()
    df = apply_cutoff(df, cutoff)
    ref_date = pd.to_datetime(cutoff) if cutoff else df["txn_date"].max()

    grp = df.groupby(["ClientID", "ProductID"], as_index=False).agg(
        cp_txns = ("txn_date", "count"),
        cp_qty_sum = ("Quantity", "sum"),
        cp_eur_sum = ("SalesNetAmountEuro", "sum"),
        cp_first_dt = ("txn_date", "min"),
        cp_last_dt = ("txn_date", "max"),
    )
    grp['days_since_last_cp'] = (ref_date - grp['cp_last_dt']).dt.days.clip(lower=0)
    grp['cp_tenure_days'] = (grp['cp_last_dt'] - grp['cp_first_dt']).dt.days.clip(lower=0)

    out = grp.drop(columns=["cp_first_dt", "cp_last_dt"]).sort_values(["ClientID", "ProductID"])
    fname = "client_product_recency.parquet" if not cutoff else f"client_product_recency_{ref_date.date()}.parquet"
    path = PROC / fname
    out.to_parquet(path, index=False)
    print(f"[OK] client_product_recency -> {path} | rows={len(out):,}")
    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--cutoff",
        type=str,
        default=None,
        help="Cutoff date in YYYY-MM-DD format, or None for no cutoff.",
    )
    args = ap.parse_args()
    build_user_rfm(cutoff=args.cutoff)
    build_client_product_recency(cutoff=args.cutoff)
from __future__ import annotations
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from src.utils.io_helpers import load_interactions
from src.utils.dates import apply_cutoff

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"

def build_category_features(cutoff: str | None = None) -> pd.DataFrame:
    """Build category level features from interactions data.

    Args:
        cutoff (str | None, optional): Cutoff date. Defaults to None.

    Returns:
        pd.DataFrame: DataFrame with category features.
    """
    df = load_interactions()
    df = apply_cutoff(df, cutoff)
    ref_date = pd.to_datetime(cutoff) if cutoff else df["txn_date"].max()

    # Popularity by Category
    total_txns = max(len(df), 1)
    cat = df.groupby("Category", as_index=False).agg(
        cat_txns=('txn_date', 'count'),
        cat_buyers=('ClientID', 'nunique'),
        cat_qty=('Quantity', 'sum'),
        cat_eur=('SalesNetAmountEuro', 'sum'),
    )
    cat['cat_txn_share'] = cat['cat_txns'] / total_txns
    cat['cat_eur_per_txn'] = (cat['cat_eur'] / cat['cat_txns']).replace([np.inf, -np.inf], np.nan)

    # Attach to product granularity (one row per Product)
    prod = (
        df.groupby("ProductID", as_index=False).agg(
            Category =('Category', 'last'),
            FamilyLevel1 = ('FamilyLevel1', 'last'),
            FamilyLevel2 = ('FamilyLevel2', 'last'),
            Universe = ('Universe', 'last'),
        )
    ).merge(cat, on ="Category", how="left", validate="m:1").sort_values("ProductID")
    
    fname = "category_features.parquet" if not cutoff else f"category_features_{ref_date.date()}.parquet"
    path = PROC / fname
    prod.to_parquet(path, index=False)
    
    print(f"[OK] category_features -> {path} | rows={len(prod):,}")
    return prod

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff", type=str, default=None)
    args = ap.parse_args()
    build_category_features(args.cutoff)
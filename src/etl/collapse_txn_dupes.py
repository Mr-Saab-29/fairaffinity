from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import numpy as np
pd.set_option('future.no_silent_downcasting', True)

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
OUT = PROC
REPORTS = PROC / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)

def normalize_txn_date(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure txn_date is in naive datetime format."""
    df["txn_date"] = pd.to_datetime(df["txn_date"], errors="coerce", utc=True).dt.tz_localize(None)
    return df

def key_cols_for(level:str) -> List[str]:
    """Return key columns based on the level of duplication."""
    if level == "c_p_d_s":
        return ["ClientID", "ProductID", "txn_date", "StoreID"]
    elif level == "c_p_d":
        return ["ClientID", "ProductID", "txn_date"]
    else:
        raise ValueError(f"Level must be one of : 'c_p_d_s', 'c_p_d'.")
    
def find_collapsible_grpoups( df: pd.DataFrame, level : str) -> pd.DataFrame:
    """Return per group stats and a booelan 'collapsible' flag
    """

    keys = key_cols_for(level)
    g = df.groupby(keys, as_index = False).agg(
        rows = ("ClientID", "size"),
        qty_nunique = ("Quantity", "nunique"),
        amt_nunique = ("SalesNetAmountEuro", "nunique"),
        stores_nunique = ("StoreID", "nunique"),
    )
    g['same_qty_amt'] = (g['qty_nunique'] == 1) & (g['amt_nunique'] == 1)

    if level == "c_p_d":
        g['collapsible'] = g['same_qty_amt'] & (g['stores_nunique'] == 1)
    else:
        g['collapsible'] = g['same_qty_amt'] 
    return g

def collapse_groups(df:pd.DataFrame, groups: pd.DataFrame, level : str) -> Tuple[pd.DataFrame, dict]:
    """Keep exactly one representative row for collapsible groups.
    Non- collapsible groups are left unchanged."""

    keys = key_cols_for(level)
    df = df.copy()

    ckeys = groups.loc[groups['collapsible'], keys]
    ckeys["__flag__"] = True

    df = df.merge(ckeys, on=keys, how="left")
    df["__flag__"] = df["__flag__"].fillna(False).astype(bool)

    # For collapsible keys: compute dup_count and keep first row
    dup_counts = (
        df[df["__flag__"]]
        .groupby(keys, as_index=False)['ClientID']
        .size()
        .rename(columns={'size': 'dup_count'}) 
    )

    # default dup_count = 1 for non-collapsible groups
    df['dup_count'] = 1

    if not dup_counts.empty:
        df = df.merge(dup_counts, on=keys, how="left", suffixes=("", "_coll"))
        df['dup_count'] = df["dup_count_coll"].fillna(df['dup_count']).astype(int)
        df.drop(columns=["dup_count_coll"], inplace=True)

    # Build final DataFrame
    kept_collapsible = (
        df[df["__flag__"]]
        .sort_values(keys)
        .drop_duplicates(subset=keys, keep="first")
    )     
    kept_non_collapsible = df[~df["__flag__"]]

    result = pd.concat([kept_collapsible, kept_non_collapsible], ignore_index=True)

    result.drop(columns=["__flag__"], inplace=True)
    before = len(df)
    after = len(result)
    removed = before - after
    metrics = {
        "rows_before" : int(before),
        "rows_after" : int(after),
        "rows_removed" : int(removed),
        "collapsible_groups" : int(groups['collapsible'].sum()),
        "groups_total" : int(len(groups)),
        "level" : level,
    }
    return result, metrics

def collapse_interactions(level : str = "c_p_d_s", sample_frac : float | None = None) -> dict:
    """Collapse duplicate transactions based on the specified level of duplication.
    """
    path = PROC / "interactions.parquet"
    if not path.exists:
        raise FileNotFoundError(f"Interactions data not found at {path}")
    
    df = pd.read_parquet(path)
    df = normalize_txn_date(df)

    if sample_frac and 0 < sample_frac < 1:
        df = df.sample(frac=sample_frac, random_state=42).reset_index(drop=True)
        print(f"[info] sampled interactions at frac={sample_frac} -> rows={len(df):,}")
    
    # Compute collapsible groups
    groups = find_collapsible_grpoups(df, level)

    #actually collapse
    result, metrics = collapse_groups(df, groups, level)

    #Write output
    out_path = PROC / f"interactions_collapsed_{level}.parquet"
    result.to_parquet(out_path, index=False)
    print(f"Collapsed interactions saved to {out_path}")

    rep = REPORTS/ f"collapse_txn_dupes_{level}.csv"
    pd.DataFrame([metrics]).to_csv(rep, index=False)
    print(f"Metrics saved to {rep}")

    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--level",
        type=str,
        choices=["c_p_d_s", "c_p_d"],
        default="c_p_d_s",
        help="Level of duplication to collapse: 'c_p_d_s' for " \
        "Client-Product-Date-Store, 'c_p_d' for Client-Product-Date.",
    )
    parser.add_argument("--sample-frac", type = float, default=None, 
                        help="Optional sampling fraction for validation (0 < frac < 1).")
    args = parser.parse_args()

    collapse_interactions(level=args.level, sample_frac=args.sample_frac)



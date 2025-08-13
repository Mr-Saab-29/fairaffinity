from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
from src.features.user_features import _apply_cutoff

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
OUT = PROC

def build_product_features(cutoff: str| None = None, velocity_days: tuple[int, ...]= (30,60))-> pd.DataFrame:
    """Build product level features from interactions data

    Args:
        cutoff (str | None, optional): cutoff date in general. Defaults to None.
        velocity_days (tuple[int, ...], optional): Days to consider for velocity calculations. Defaults to (30, 60).

    Returns:
        pd.DataFrame: Dataframe with product features
    """
    df = pd.read_parquet(PROC / "interactions.parquet")
    df["txn_date"] = (
    pd.to_datetime(df["txn_date"], errors="coerce", utc=True)
      .dt.tz_localize(None)
    )

    df = _apply_cutoff(df, cutoff)
    if df.empty:
        raise ValueError(" No data before cutoff; choose a later cutoff date")
    
    ref_date = pd.to_datetime(cutoff) if cutoff else df["txn_date"].max()

    # Base product stats
    base = df.groupby("ProductID", as_index=False).agg(
    txns = ('txn_date', 'count'),
    buyers = ('ClientID', 'nunique'),
    qty_sum = ('Quantity', 'sum'),
    eur_sum = ('SalesNetAmountEuro', 'sum'),
    first_dt = ('txn_date', 'min'),
    last_dt = ('txn_date', 'max'),
    category = ('Category', 'last'),
    family_lvl1 = ('FamilyLevel1', 'last'),
    family_lvl2 = ('FamilyLevel2', 'last'),
    universe = ('Universe', 'last'),
    )

    #Velocity metrics
    result = base.copy()
    for days in velocity_days:
        window_df = df[df['txn_date'] >= (ref_date - pd.Timedelta(days=days))]
        velocity_stats = (
            window_df.groupby("ProductID")
            .agg(
                **{f'qty_{days}d' : ('Quantity', 'sum')},
                **{f'eur_{days}d' : ('SalesNetAmountEuro', 'sum')},
                **{f'txns_{days}d' : ('txn_date', 'count')},
                **{f'buyers_{days}d' : ('ClientID', 'nunique')},
                )
                .reset_index()
            )
        result = result.merge(velocity_stats, on="ProductID", how="left", validate="1:1")
    
    # Price Intensity
    result['eur_per_qty'] = (
        result['eur_sum'] / result['qty_sum']
    ).replace([np.inf, -np.inf], np.nan)

    # Product age and recency
    result['product_age_days'] = (
        result['last_dt'] - result['first_dt']
    ).dt.days.clip(lower=0)

    result['days_since_sold'] = (
        ref_date - result['last_dt']
    ).dt.days.clip(lower=0)

    # Clean up
    result = result.drop(columns=['first_dt', 'last_dt'])
    result = result.sort_values(by='ProductID').reset_index(drop=True)

    # Save the result
    filename = (
        "product_features.parquet"
        if not cutoff
        else f'product_features_ {pd.to_datetime(cutoff).date()}.parquet'
    )

    out_path = OUT / filename
    result.to_parquet(out_path, index=False)
    print(f"Product features saved to {out_path}")
    return result

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cutoff",
        type=str,
        default=None,
        help="Cutoff date in YYYY-MM-DD format, or None for no cutoff.",
    )
    parser.add_argument(
        "--velocity-days",
        type=int,
        nargs="+",
        default=[30, 60],
        help="Days to consider for velocity calculations (default: 30, 60).",
    )

    args = parser.parse_args()
    build_product_features(cutoff=args.cutoff, velocity_days=tuple(args.velocity_days))


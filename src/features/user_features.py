from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
import time

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
OUT  = PROC

def _apply_cutoff(df: pd.DataFrame, cutoff: float | None) -> pd.DataFrame:
    """ 
    Filter transactions before the cutoff date. (if given)
    """
    if cutoff:
        cutoff_ts = pd.to_datetime(cutoff, errors='coerce')
        if pd.isna(cutoff_ts):
            raise ValueError(f"Invalid cutoff date: {cutoff}")
        return df[df['txn_date'] < cutoff_ts].copy()
    return df

def mode_safe(series: pd.Series) -> pd.Series:
    """
    Return the most frequent value or NaN if empty.
    """
    if series.dropna().empty:
        return np.nan
    return series.value_counts().idxmax()

def row_entropy(arr: np.ndarray) -> np.ndarray:
    """Row-wise entropy for a 2D non-negative array."""
    row_sums = arr.sum(axis=1, keepdims=True)
    # avoid divide-by-zero
    probs = np.divide(arr, row_sums, out=np.zeros_like(arr, dtype=float), where=row_sums != 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        logp = np.log(probs, where=probs > 0)
    ent = -(probs * logp)
    ent[np.isnan(ent)] = 0.0
    return ent.sum(axis=1)

def build_user_features(cutoff: str | None = None, sample_frac: float | None = None) -> pd.DataFrame:
    t0 = time.perf_counter()
    print("[user] loading interactions…")
    df = pd.read_parquet(PROC / "interactions.parquet")

    # ensure datetime
    df["txn_date"] = (
    pd.to_datetime(df["txn_date"], errors="coerce", utc=True)
      .dt.tz_localize(None)
    )

    if sample_frac and 0 < sample_frac < 1:
        df = df.sample(frac=sample_frac, random_state=42).reset_index(drop=True)
        print(f"[user] sampled {sample_frac:.2%} -> rows={len(df):,}")

    df = _apply_cutoff(df, cutoff)
    if df.empty:
        raise ValueError("No data before cutoff; choose a later cutoff.")

    print(f"[user] rows={len(df):,} | clients={df['ClientID'].nunique():,} | elapsed={time.perf_counter()-t0:.1f}s")

    # ---------- basic spend/volume ----------
    t1 = time.perf_counter()
    grouped = df.groupby("ClientID", as_index=False).agg(
        txns=("txn_date", "count"),
        qty_sum=("Quantity", "sum"),
        eur_sum=("SalesNetAmountEuro", "sum"),
        first_dt=("txn_date", "min"),
        last_dt=("txn_date", "max"),
    )
    ref_date = pd.to_datetime(cutoff) if cutoff else df["txn_date"].max()
    grouped["recency_days"] = (ref_date - grouped["last_dt"]).dt.days.clip(lower=0)
    print(f"[user] base agg done in {time.perf_counter()-t1:.1f}s")

    # ---------- basket stats ----------
    t2 = time.perf_counter()
    basket_stats = (
        df.assign(basket_val=df["SalesNetAmountEuro"])
          .groupby("ClientID")
          .agg(basket_mean=("basket_val", "mean"),
               basket_median=("basket_val", "median"))
          .reset_index()
    )
    print(f"[user] basket stats in {time.perf_counter()-t2:.1f}s")

    # ---------- diversity ----------
    t3 = time.perf_counter()
    diversity = (
        df.groupby("ClientID")
          .agg(n_products=("ProductID", "nunique"),
               n_categories=("Category", "nunique"),
               n_stores=("StoreID", "nunique"))
          .reset_index()
    )
    print(f"[user] diversity in {time.perf_counter()-t3:.1f}s")

    # ---------- preferences (modes) ----------
    t4 = time.perf_counter()
    def top_by_count(frame: pd.DataFrame, key: str, col: str, out_col: str) -> pd.DataFrame:
        tmp = (
            frame.dropna(subset=[col])
                .groupby([key, col], observed=True)
                .size()
                .reset_index(name="cnt")
                .sort_values(["ClientID", "cnt"], ascending=[True, False])
        )
    # keep top row per ClientID
        top = tmp.drop_duplicates(subset=[key], keep="first")[[key, col]]
        return top.rename(columns={col: out_col})

    preferences = top_by_count(df, "ClientID", "StoreCountry", "pref_store_country")
    preferences = preferences.merge(
        top_by_count(df, "ClientID", "ClientCountry", "pref_client_country"),
        on="ClientID", how="outer"
    )
    preferences = preferences.merge(
        top_by_count(df, "ClientID", "Category", "top_category"),
        on="ClientID", how="outer"
    )
    preferences = preferences.merge(
        top_by_count(df, "ClientID", "FamilyLevel1", "top_family_lvl1"),
        on="ClientID", how="outer"
    )

    print(f"[user] preferences in {time.perf_counter()-t4:.1f}s")

    # ---------- FAST entropy via crosstab ----------
    t5 = time.perf_counter()
    # Month entropy
    month_ct = pd.crosstab(df["ClientID"], df["txn_date"].dt.month)
    month_ent = row_entropy(month_ct.to_numpy())
    month_ent_df = pd.DataFrame({"ClientID": month_ct.index, "month_entropy": month_ent})

    # DOW entropy
    dow_ct = pd.crosstab(df["ClientID"], df["txn_date"].dt.dayofweek)
    dow_ent = row_entropy(dow_ct.to_numpy())
    dow_ent_df = pd.DataFrame({"ClientID": dow_ct.index, "dow_entropy": dow_ent})

    temporal = month_ent_df.merge(dow_ent_df, on="ClientID", how="outer")
    print(f"[user] temporal entropy in {time.perf_counter()-t5:.1f}s")

    # ---------- merge all ----------
    t6 = time.perf_counter()
    user_features = (
        grouped.merge(basket_stats, on="ClientID", how="left")
               .merge(diversity, on="ClientID", how="left")
               .merge(preferences, on="ClientID", how="left")
               .merge(temporal, on="ClientID", how="left")
    )
    user_features["customer_age_days"] = (
        grouped["last_dt"] - grouped["first_dt"]
    ).dt.days.clip(lower=0)
    user_features = user_features.drop(columns=["first_dt", "last_dt"]).sort_values("ClientID").reset_index(drop=True)
    print(f"[user] merged in {time.perf_counter()-t6:.1f}s")

    # ---------- save ----------
    filename = "user_features.parquet" if not cutoff else f"user_features_{pd.to_datetime(cutoff).date()}.parquet"
    out_path = OUT / filename
    user_features.to_parquet(out_path, index=False)
    print(f"[OK] user_features -> {out_path} | rows={len(user_features):,} | total={time.perf_counter()-t0:.1f}s")
    return user_features


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--cutoff", type=str, default=None)
    parser.add_argument("--sample-frac", type=float, default=None, help="0<frac<1 to test quickly")
    args = parser.parse_args()
    build_user_features(cutoff=args.cutoff, sample_frac=args.sample_frac)

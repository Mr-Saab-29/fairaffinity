from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
OUT = PROC / "reports"
OUT.mkdir(parents=True, exist_ok=True)


def load_interactions(sample_frac: float | None = None) -> pd.DataFrame:
    path = PROC / "interactions.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    df = pd.read_parquet(path)
    # normalize datetime to be naive (no tz)
    df["txn_date"] = pd.to_datetime(df["txn_date"], errors="coerce", utc=True).dt.tz_localize(None)

    if sample_frac and 0 < sample_frac < 1:
        df = df.sample(frac=sample_frac, random_state=42).reset_index(drop=True)
        print(f"[info] sampled interactions at frac={sample_frac} -> rows={len(df):,}")
    return df


def describe_dup_clusters(df: pd.DataFrame, key_cols: list[str]) -> pd.DataFrame:
    """Return per-key cluster stats: size and whether qty/amount vary."""
    # count rows per composite key
    grp = df.groupby(key_cols, as_index=False).agg(
        rows=("ClientID", "size"),
        qty_nunique=("Quantity", "nunique"),
        amt_nunique=("SalesNetAmountEuro", "nunique"),
        stores_nunique=("StoreID", "nunique"),
    )
    grp["has_var_qty"] = grp["qty_nunique"] > 1
    grp["has_var_amt"] = grp["amt_nunique"] > 1
    grp["multi_store"] = grp["stores_nunique"] > 1 if "StoreID" not in key_cols else False
    return grp


def summary_from_clusters(cl: pd.DataFrame, label: str) -> pd.Series:
    s = pd.Series(dtype="object", name=label)
    s["clusters"] = len(cl)
    s["rows_in_clusters"] = int(cl["rows"].sum())
    s["median_rows_per_cluster"] = float(cl["rows"].median())
    s["p95_rows_per_cluster"] = float(cl["rows"].quantile(0.95))
    s["share_var_qty"] = float((cl["has_var_qty"]).mean())
    s["share_var_amt"] = float((cl["has_var_amt"]).mean())
    if "multi_store" in cl.columns:
        s["share_multi_store"] = float(cl["multi_store"].mean())
    return s


def export_examples(df: pd.DataFrame, key_cols: list[str], out_prefix: str, n: int = 10) -> None:
    """Write a few example clusters for inspection."""
    cl = (
        df.groupby(key_cols, as_index=False)
          .size()
          .sort_values("size", ascending=False)
          .head(n)
    )
    merged = df.merge(cl.drop(columns="size"), on=key_cols, how="inner")
    merged.sort_values(key_cols + ["Quantity", "SalesNetAmountEuro"]).to_csv(
        OUT / f"{out_prefix}_examples.csv", index=False
    )


def analyze(sample_frac: float | None = None) -> None:
    df = load_interactions(sample_frac=sample_frac)

    # 1) Exact row duplicates (already removed earlier, but report anyway)
    exact_dupes = df.duplicated().sum()
    print(f"[report] exact duplicate rows: {exact_dupes:,}")

    # 2) Composite keys
    k1 = ["ClientID", "ProductID", "txn_date", "StoreID"]
    k2 = ["ClientID", "ProductID", "txn_date"]

    cl1 = describe_dup_clusters(df, k1)
    cl2 = describe_dup_clusters(df, k2)

    # Summaries
    s1 = summary_from_clusters(cl1, label="by_client_product_date_store")
    s2 = summary_from_clusters(cl2, label="by_client_product_date")

    # Save summaries
    summary = pd.concat([s1, s2], axis=1)
    summary.to_csv(OUT / "dupes_summary.csv")
    print(f"[OK] wrote {OUT / 'dupes_summary.csv'}")

    # Save full cluster distributions (optional but useful)
    cl1.to_csv(OUT / "dupe_clusters_client_product_date_store.csv", index=False)
    cl2.to_csv(OUT / "dupe_clusters_client_product_date.csv", index=False)
    print(f"[OK] wrote {OUT / 'dupe_clusters_client_product_date_store.csv'}")
    print(f"[OK] wrote {OUT / 'dupe_clusters_client_product_date.csv'}")

    # Export a few heaviest clusters for eyeballing
    export_examples(df, k1, out_prefix="heaviest_by_c_p_d_store", n=10)
    export_examples(df, k2, out_prefix="heaviest_by_c_p_d", n=10)
    print(f"[OK] wrote example clusters to CSV in {OUT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-frac", type=float, default=None)
    args = parser.parse_args()
    analyze(sample_frac=args.sample_frac)
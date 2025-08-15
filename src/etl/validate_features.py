from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
REP  = PROC / "reports"
REP.mkdir(parents=True, exist_ok=True)

def validate_user_features(path: Path) -> dict:
    df = pd.read_parquet(path)
    rep = {"artifact": "user_features", "rows": len(df)}
    rep["key_unique"] = bool(df["ClientID"].is_unique)
    rep["null_key"] = int(df["ClientID"].isna().sum())
    for c in ["recency_days", "customer_age_days"]:
        if c in df.columns:
            rep[f"{c}_nonneg"] = bool((df[c] >= 0).all())
    return rep

def validate_product_features(path: Path) -> dict:
    df = pd.read_parquet(path)
    rep = {"artifact": "product_features", "rows": len(df)}
    rep["key_unique"] = bool(df["ProductID"].is_unique)
    rep["null_key"] = int(df["ProductID"].isna().sum())
    if "eur_per_qty" in df.columns:
        rep["eur_per_qty_nonneg_or_nan"] = bool(((df["eur_per_qty"].isna()) | (df["eur_per_qty"] >= 0)).all())
    for c in ["product_age_days", "days_since_sold"]:
        if c in df.columns:
            rep[f"{c}_nonneg"] = bool((df[c] >= 0).all())
    return rep

def validate_user_rfm(path: Path) -> dict:
    df = pd.read_parquet(path)
    rep = {"artifact": "user_rfm", "rows": len(df)}
    rep["key_unique"] = bool(df["ClientID"].is_unique)
    rep["null_key"] = int(df["ClientID"].isna().sum())
    for c in ["recency_days", "tenure_days", "freq_txns", "qty_sum", "eur_sum"]:
        if c in df.columns:
            rep[f"{c}_nonneg"] = bool((df[c] >= 0).all())
    if "avg_basket_eur" in df.columns:
        rep["avg_basket_eur_nonneg_or_nan"] = bool(((df["avg_basket_eur"].isna()) | (df["avg_basket_eur"] >= 0)).all())
    return rep

def validate_client_product_recency(path: Path) -> dict:
    df = pd.read_parquet(path)
    rep = {"artifact": "client_product_recency", "rows": len(df)}
    rep["key_unique"] = bool(df[["ClientID","ProductID"]].drop_duplicates().shape[0] == len(df))
    rep["null_key"] = int(df["ClientID"].isna().sum() + df["ProductID"].isna().sum())
    for c in ["days_since_last_cp", "cp_tenure_days", "cp_txns", "cp_qty_sum", "cp_eur_sum"]:
        if c in df.columns:
            rep[f"{c}_nonneg"] = bool((df[c] >= 0).all())
    return rep

def validate_category_features(path: Path) -> dict:
    df = pd.read_parquet(path)
    rep = {"artifact": "category_features", "rows": len(df)}
    rep["key_unique"] = bool(df["ProductID"].is_unique)
    rep["null_key"] = int(df["ProductID"].isna().sum())
    for c in ["cat_txns", "cat_qty", "cat_eur", "cat_buyers", "cat_txn_share"]:
        if c in df.columns:
            rep[f"{c}_nonneg"] = bool((df[c] >= 0).all())
    if "cat_eur_per_txn" in df.columns:
        rep["cat_eur_per_txn_nonneg_or_nan"] = bool(((df["cat_eur_per_txn"].isna()) | (df["cat_eur_per_txn"] >= 0)).all())
    return rep

def main() -> None:
    checks = []
    paths = {
        "user_features": PROC / "user_features.parquet",
        "product_features": PROC / "product_features.parquet",
        "user_rfm": PROC / "user_rfm.parquet",
        "client_product_recency": PROC / "client_product_recency.parquet",
        "category_features": PROC / "category_features.parquet",
    }
    if paths["user_features"].exists():
        checks.append(validate_user_features(paths["user_features"]))
    if paths["product_features"].exists():
        checks.append(validate_product_features(paths["product_features"]))
    if paths["user_rfm"].exists():
        checks.append(validate_user_rfm(paths["user_rfm"]))
    if paths["client_product_recency"].exists():
        checks.append(validate_client_product_recency(paths["client_product_recency"]))
    if paths["category_features"].exists():
        checks.append(validate_category_features(paths["category_features"]))

    out = REP / "features_validation.csv"
    pd.DataFrame(checks).to_csv(out, index=False)
    print(f"[OK] wrote {out}")
    for row in checks:
        print(row)

if __name__ == "__main__":
    main()
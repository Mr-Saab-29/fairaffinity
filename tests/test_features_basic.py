from __future__ import annotations

from pathlib import Path
import pandas as pd
import pytest

# Paths
ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"

# ---------- helpers ----------

def assert_nonneg(series, name: str):
    assert (series >= 0).all(), f"{name} contains negative values"

def assert_unique_key(df: pd.DataFrame, cols: list[str], name: str):
    n = len(df)
    n_unique = df[cols].drop_duplicates().shape[0]
    assert n == n_unique, f"{name}: key {cols} not unique (rows={n}, unique={n_unique})"

# ---------- tests ----------

def test_user_features_basic():
    p = PROC / "user_features.parquet"
    if not p.exists():
        pytest.skip(f"Missing artifact: {p}")
    df = pd.read_parquet(p)

    # key
    assert "ClientID" in df.columns, "ClientID missing in user_features"
    assert df["ClientID"].is_unique, "ClientID must be unique"

    # non-negatives (if present)
    for col in ["recency_days", "customer_age_days", "txns", "qty_sum", "eur_sum"]:
        if col in df.columns:
            assert_nonneg(df[col], f"user_features.{col}")

def test_product_features_basic():
    p = PROC / "product_features.parquet"
    if not p.exists():
        pytest.skip(f"Missing artifact: {p}")
    df = pd.read_parquet(p)

    # key
    assert "ProductID" in df.columns, "ProductID missing in product_features"
    assert df["ProductID"].is_unique, "ProductID must be unique"

    # non-negatives (if present)
    for col in ["product_age_days", "days_since_sold", "txns", "qty_sum", "eur_sum"]:
        if col in df.columns:
            assert_nonneg(df[col], f"product_features.{col}")

    # ratios can be NaN but not negative
    if "eur_per_qty" in df.columns:
        ok = (df["eur_per_qty"].isna()) | (df["eur_per_qty"] >= 0)
        assert ok.all(), "product_features.eur_per_qty has negative values"

def test_user_rfm_basic():
    p = PROC / "user_rfm.parquet"
    if not p.exists():
        pytest.skip(f"Missing artifact: {p}")
    df = pd.read_parquet(p)

    assert "ClientID" in df.columns and df["ClientID"].is_unique, "ClientID not unique in user_rfm"
    for col in ["recency_days", "tenure_days", "freq_txns", "qty_sum", "eur_sum"]:
        if col in df.columns:
            assert_nonneg(df[col], f"user_rfm.{col}")
    if "avg_basket_eur" in df.columns:
        ok = (df["avg_basket_eur"].isna()) | (df["avg_basket_eur"] >= 0)
        assert ok.all(), "user_rfm.avg_basket_eur has negative values"

def test_client_product_recency_basic():
    p = PROC / "client_product_recency.parquet"
    if not p.exists():
        pytest.skip(f"Missing artifact: {p}")
    df = pd.read_parquet(p)

    # composite key uniqueness
    assert_unique_key(df, ["ClientID", "ProductID"], "client_product_recency")

    for col in ["days_since_last_cp", "cp_tenure_days", "cp_txns", "cp_qty_sum", "cp_eur_sum"]:
        if col in df.columns:
            assert_nonneg(df[col], f"client_product_recency.{col}")

def test_category_features_basic():
    p = PROC / "category_features.parquet"
    if not p.exists():
        pytest.skip(f"Missing artifact: {p}")
    df = pd.read_parquet(p)

    assert "ProductID" in df.columns and df["ProductID"].is_unique, "ProductID not unique in category_features"

    for col in ["cat_txns", "cat_qty", "cat_eur", "cat_buyers", "cat_txn_share"]:
        if col in df.columns:
            assert_nonneg(df[col], f"category_features.{col}")
    if "cat_eur_per_txn" in df.columns:
        ok = (df["cat_eur_per_txn"].isna()) | (df["cat_eur_per_txn"] >= 0)
        assert ok.all(), "category_features.cat_eur_per_txn has negative values"

def test_loader_prefers_collapsed_interactions():
    """Ensure io_helpers.load_interactions() prefers the collapsed file when present."""
    from src.utils.io_helpers import load_interactions

    collapsed = PROC / "interactions_collapsed_c_p_d_s.parquet"
    raw = PROC / "interactions.parquet"

    # At least one must exist for this test to be meaningful
    if not (collapsed.exists() or raw.exists()):
        pytest.skip("No interactions parquet found in data/processed/")

    df_loaded = load_interactions()
    n_loaded = len(df_loaded)

    if collapsed.exists():
        n_collapsed = len(pd.read_parquet(collapsed))
        assert n_loaded == n_collapsed, (
            "Loader should use collapsed interactions when present"
        )
    else:
        # fallback to raw
        n_raw = len(pd.read_parquet(raw))
        assert n_loaded == n_raw, "Loader should fall back to raw interactions when collapsed is absent"

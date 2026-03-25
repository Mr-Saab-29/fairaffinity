from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
INTERACTIONS_PATH = ROOT / "data" / "processed" / "interactions_collapsed_c_p_d_s.parquet"
INTERACTIONS_FALLBACK = ROOT / "data" / "processed" / "interactions.parquet"


@st.cache_data(show_spinner=False)
def _load_interactions() -> pd.DataFrame:
    src = INTERACTIONS_PATH if INTERACTIONS_PATH.exists() else INTERACTIONS_FALLBACK
    if not src.exists():
        return pd.DataFrame(
            columns=["ClientID", "ProductID", "Category", "FamilyLevel1", "FamilyLevel2", "Universe"]
        )
    cols = ["ClientID", "ProductID", "Category", "FamilyLevel1", "FamilyLevel2", "Universe"]
    try:
        from pyarrow.parquet import read_schema

        schema_cols = set(read_schema(src).names)
    except Exception:
        schema_cols = set(pd.read_parquet(src).columns)
    df = pd.read_parquet(src, columns=[c for c in cols if c in schema_cols])
    for c in cols:
        if c not in df.columns:
            df[c] = "UNKNOWN"
    return df.dropna(subset=["ClientID", "ProductID"]).copy()


@st.cache_data(show_spinner=False)
def _build_catalog(df: pd.DataFrame) -> pd.DataFrame:
    cat = (
        df.groupby("ProductID", as_index=False)
        .agg(
            Category=("Category", "last"),
            FamilyLevel1=("FamilyLevel1", "last"),
            FamilyLevel2=("FamilyLevel2", "last"),
            Universe=("Universe", "last"),
        )
        .fillna("UNKNOWN")
    )
    cat["ProductID"] = cat["ProductID"].astype(int)
    cat["ProductName"] = (
        cat["Category"].astype(str)
        + " · "
        + cat["FamilyLevel1"].astype(str)
        + " · "
        + cat["FamilyLevel2"].astype(str)
    )
    cat["ProductName"] = cat["ProductName"].str.replace(r"\s+", " ", regex=True).str.strip(" ·")
    # Keep labels human-friendly while making duplicate names selectable.
    dup_rank = cat.groupby("ProductName").cumcount() + 1
    dup_total = cat.groupby("ProductName")["ProductName"].transform("size")
    cat["ProductLabel"] = np.where(dup_total > 1, cat["ProductName"] + " (option " + dup_rank.astype(str) + ")", cat["ProductName"])
    return cat.sort_values("ProductName").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def _build_maps(df: pd.DataFrame) -> tuple[dict[int, set[int]], dict[int, set[int]]]:
    by_user = df.groupby("ClientID")["ProductID"].apply(lambda s: set(s.astype(int))).to_dict()
    by_product = df.groupby("ProductID")["ClientID"].apply(lambda s: set(s.astype(int))).to_dict()
    return by_user, by_product


def _recommend_from_product(
    seed_product: int,
    by_user: dict[int, set[int]],
    by_product: dict[int, set[int]],
    top_n: int = 5,
) -> pd.DataFrame:
    seed_users = by_product.get(seed_product, set())
    if not seed_users:
        return pd.DataFrame(columns=["ProductID", "_score", "_co_users"])

    candidate_counts: dict[int, int] = {}
    for uid in seed_users:
        for pid in by_user.get(uid, set()):
            if pid == seed_product:
                continue
            candidate_counts[pid] = candidate_counts.get(pid, 0) + 1

    seed_user_count = max(len(seed_users), 1)
    rows = []
    for pid, co in candidate_counts.items():
        cand_users = by_product.get(pid, set())
        cand_user_count = max(len(cand_users), 1)
        score = co / np.sqrt(seed_user_count * cand_user_count)
        rows.append({"ProductID": int(pid), "_score": float(score), "_co_users": int(co)})
    return pd.DataFrame(rows).sort_values(["_score", "_co_users"], ascending=[False, False]).head(top_n)


def _recommend_from_category(
    seed_category: str,
    interactions: pd.DataFrame,
    by_user: dict[int, set[int]],
    by_product: dict[int, set[int]],
    top_n: int = 5,
) -> pd.DataFrame:
    _ = by_user, by_product
    cat_df = interactions.loc[interactions["Category"] == seed_category, ["ClientID", "ProductID"]].copy()
    if cat_df.empty:
        return pd.DataFrame(columns=["ProductID", "_score", "_co_users"])
    agg = (
        cat_df.groupby("ProductID", as_index=False)
        .agg(_co_users=("ClientID", "nunique"), _score=("ClientID", "count"))
        .sort_values(["_score", "_co_users"], ascending=[False, False])
    )
    agg["ProductID"] = agg["ProductID"].astype(int)
    agg["_score"] = agg["_score"].astype(float)
    return agg.head(top_n)


def _enrich_with_catalog(df: pd.DataFrame, catalog: pd.DataFrame) -> pd.DataFrame:
    if "ProductID" in df.columns:
        out = df.copy()
        out["ProductID"] = pd.to_numeric(out["ProductID"], errors="coerce").astype("Int64")
        out = out.dropna(subset=["ProductID"]).copy()
        out["ProductID"] = out["ProductID"].astype(int)
    else:
        out = df.copy()
    out = out.merge(catalog, on="ProductID", how="left")
    for c in ["Category", "FamilyLevel1", "FamilyLevel2", "Universe"]:
        if c not in out.columns:
            out[c] = "UNKNOWN"
        out[c] = out[c].fillna("UNKNOWN")
    if "ProductName" not in out.columns:
        out["ProductName"] = "UNKNOWN"
    out["ProductName"] = out["ProductName"].fillna(out["ProductID"].map(lambda x: f"Product {x}")).fillna("UNKNOWN")
    return out


def _render_product_cards(df: pd.DataFrame, key_prefix: str, add_to_cart: bool = True) -> None:
    if df.empty:
        st.info("No products found.")
        return
    cols_per_row = 3
    for i in range(0, len(df), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, (_, row) in enumerate(df.iloc[i : i + cols_per_row].iterrows()):
            with cols[j]:
                with st.container(border=True):
                    st.markdown(f"**{row['ProductName']}**")
                    st.caption(f"Category: {row['Category']} | Family: {row['FamilyLevel1']}")
                    if "Why this is recommended" in row:
                        st.write(row["Why this is recommended"])
                    if add_to_cart:
                        bkey = f"{key_prefix}_add_{int(row['ProductID'])}_{i}_{j}"
                        if st.button("Add to cart", key=bkey, use_container_width=True):
                            cart = st.session_state.setdefault("cart", [])
                            pid = int(row["ProductID"])
                            if pid not in cart:
                                cart.append(pid)
                                st.toast("Added to cart")


def _cart_related_items(
    cart_ids: list[int],
    by_user: dict[int, set[int]],
    by_product: dict[int, set[int]],
    top_n: int = 5,
) -> pd.DataFrame:
    if not cart_ids:
        return pd.DataFrame(columns=["ProductID", "_score", "_co_users"])
    score: dict[int, float] = {}
    support: dict[int, int] = {}
    for pid in cart_ids:
        recs = _recommend_from_product(pid, by_user, by_product, top_n=50)
        for _, r in recs.iterrows():
            rid = int(r["ProductID"])
            if rid in cart_ids:
                continue
            score[rid] = score.get(rid, 0.0) + float(r["_score"])
            support[rid] = support.get(rid, 0) + int(r["_co_users"])
    rows = [{"ProductID": k, "_score": v, "_co_users": support[k]} for k, v in score.items()]
    return pd.DataFrame(rows).sort_values(["_score", "_co_users"], ascending=[False, False]).head(top_n)


def _filter_similar_to_seed(recs: pd.DataFrame, seed_pid: int, catalog: pd.DataFrame) -> pd.DataFrame:
    if recs.empty:
        return recs
    seed = catalog[catalog["ProductID"] == int(seed_pid)]
    if seed.empty:
        return recs
    seed_family = str(seed["FamilyLevel1"].iloc[0])
    seed_category = str(seed["Category"].iloc[0])
    same_family = recs[recs["FamilyLevel1"] == seed_family]
    if not same_family.empty:
        return same_family
    same_category = recs[recs["Category"] == seed_category]
    if not same_category.empty:
        return same_category
    return recs


st.set_page_config(page_title="SportsLand", layout="wide")
st.title("SportsLand")
st.caption("Pick products you like and discover related items.")

interactions = _load_interactions()
if interactions.empty:
    st.error("Interactions data not found.")
    st.stop()

catalog = _build_catalog(interactions)
by_user, by_product = _build_maps(interactions)

if "cart" not in st.session_state:
    st.session_state["cart"] = []

with st.sidebar:
    st.header("Your Cart")
    cart_ids = st.session_state["cart"]
    if not cart_ids:
        st.caption("Cart is empty.")
    else:
        cart_df = catalog[catalog["ProductID"].isin(cart_ids)][["ProductID", "ProductName", "Category"]]
        for _, r in cart_df.iterrows():
            c1, c2 = st.columns([5, 2])
            with c1:
                st.caption(f"{r['ProductName']} ({r['Category']})")
            with c2:
                if st.button("Remove", key=f"rm_{int(r['ProductID'])}", use_container_width=True):
                    st.session_state["cart"] = [pid for pid in st.session_state["cart"] if pid != int(r["ProductID"])]
                    st.rerun()
        if st.button("Clear cart", use_container_width=True):
            st.session_state["cart"] = []
            st.rerun()

tabs = st.tabs(["Shop by Product", "Shop by Category"])

with tabs[0]:
    st.subheader("Shop by Product Name")
    search = st.text_input("Search product", placeholder="Type product/category name (e.g., Football)")
    filtered = catalog
    if search.strip():
        q = search.strip().lower()
        filtered = catalog[
            catalog["ProductName"].str.lower().str.contains(q, na=False)
            | catalog["Category"].str.lower().str.contains(q, na=False)
            | catalog["FamilyLevel1"].str.lower().str.contains(q, na=False)
            | catalog["FamilyLevel2"].str.lower().str.contains(q, na=False)
        ]
    if filtered.empty:
        st.info("No products match your search.")
    else:
        option_pids = filtered["ProductID"].astype(int).tolist()
        pid_to_label = dict(zip(filtered["ProductID"].astype(int), filtered["ProductLabel"].astype(str)))
        selected_pid = st.selectbox(
            "Select a product (autocomplete by name)",
            options=option_pids,
            index=0,
            format_func=lambda pid: pid_to_label.get(int(pid), str(pid)),
        )
        top_n = st.slider("Number of recommendations", min_value=6, max_value=24, value=12, step=1, key="p_topn")
        recs = _recommend_from_product(selected_pid, by_user, by_product, top_n=200)
        recs = _enrich_with_catalog(recs, catalog)
        recs = _filter_similar_to_seed(recs, selected_pid, catalog).head(top_n)
        recs["Why this is recommended"] = "Frequently bought together with your selected product."
        _render_product_cards(recs, key_prefix="prod")

with tabs[1]:
    st.subheader("Shop by Category")
    categories = sorted(catalog["Category"].dropna().astype(str).unique().tolist())
    selected_category = st.selectbox("Choose a category", options=categories, index=0)
    top_n_cat = st.slider("Number of recommendations", min_value=6, max_value=24, value=12, step=1, key="c_topn")
    recs_cat = _recommend_from_category(selected_category, interactions, by_user, by_product, top_n=top_n_cat)
    recs_cat = _enrich_with_catalog(recs_cat, catalog)
    recs_cat["Why this is recommended"] = f"Popular among shoppers browsing {selected_category}."
    _render_product_cards(recs_cat, key_prefix="cat")

st.subheader("Related Items for Your Cart")
related = _cart_related_items(st.session_state["cart"], by_user, by_product, top_n=12)
related = _enrich_with_catalog(related, catalog)
if st.session_state["cart"]:
    cart_categories = set(catalog.loc[catalog["ProductID"].isin(st.session_state["cart"]), "Category"].astype(str))
    same_cart_categories = related[related["Category"].astype(str).isin(cart_categories)]
    if not same_cart_categories.empty:
        related = same_cart_categories.head(12)
related["Why this is recommended"] = "Matches the items currently in your cart."
_render_product_cards(related, key_prefix="cart")

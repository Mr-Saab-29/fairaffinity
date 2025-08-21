from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Literal

import numpy as np
import pandas as pd

from src.utils.io_helpers import load_interactions
from src.utils.dates import normalize_txn_date

from src.features.user_features import build_user_features
from src.features.category_features import build_category_features
from src.features.rfm_features import build_client_product_recency
from src.features.product_features import build_product_features

# ---------------------------------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
MODEL_DIR = PROC / 'model'
REPORTS = PROC / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------------------------------
# Configure dataclass
# ---------------------------------------------------------------------------------------------------
HardNegMode = Literal['none', 'pop', 'pop_in_artefact']
PosTargetMode = Literal['binary', 'count_txn', 'sum_qty']

@dataclass
class SplitCfg:
    """ Configuration for one split (train, validation, test) """
    name : str
    cutoff : pd.Timestamp
    label_days : int
    negatives_per_pos : int
    hard_negatives : HardNegMode
    pos_target: PosTargetMode
    random_state : int = 42
    sample_clients : int | None = None  # Optional: for debugging, sample a subset of clients
    cap_negatives : int | None = None  # Optional: cap negatives per client, if set to None, no cap is applied

# ---------------------------------------------------------------------------------------------------
# Helper Functions : Window, Positives, Negatives and Feature Artifact IO
# ---------------------------------------------------------------------------------------------------

def _history_window(
        full: pd.DataFrame, cutoff: pd.Timestamp, label_days: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split full DataFrame into history and label windows.
    - history : txn_date <= cutoff
    - window : (cutoff, cutoff + delta], where labels/ targets are measured

    Args:
        full (pd.DataFrame): Full DataFrame with transactions.
        cutoff (pd.Timestamp): Cutoff date for splitting.
        label_days (int): Number of days after cutoff to consider for labels.
    
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Tuple containing history DataFrame and window DataFrame.
    """
    df = full.copy()
    df['txn_date'] = normalize_txn_date(df['txn_date'])
    history = df[df['txn_date'] <= cutoff]
    hi = cutoff + pd.Timedelta(days=label_days)
    window = df[(df['txn_date'] > cutoff) & (df['txn_date'] <= hi)]
    return history, window

def _positives(window: pd.DataFrame, mode: PosTargetMode) -> pd.DataFrame:
    """Build positives (one row per CLientID, ProductID) from the label window.
    - binary: label = 1
    - count_txn: targetr = # of transactions in the window
    - sum_qty: target = sum of Quantity in the window

    Args:
        window (pd.DataFrame): DataFrame containing transactions in the label window.
        mode (PosTargetMode): Mode for positive targets, can be 'binary', 'count_txn', or 'sum_qty'.

    Returns:
        pd.DataFrame: Dataframe with columns [(ClientID, ProductID), label, target]
    """
    if window.empty:
        return pd.DataFrame(columns=['ClientID', 'ProductID', 'label', 'target'])
    
    if mode == 'binary':
        pos = window[['ClientID', 'ProductID']].drop_duplicates()
        pos['label'] = 1
        pos['target'] = 1
        return pos
    
    agg = {
        'count_txn': ('txn_date', 'count'),
        'sum_qty': ('Quantity', 'sum')
    }
    pick = 'count_txn' if mode == 'count_txn' else 'sum_qty'

    g = (
        window.groupby(['ClientID', 'ProductID'], as_index=False)
        .agg(**agg)
    )
    g = g[g[pick] > 0].copy()
    g['label'] = 1
    g['target'] = g[pick]
    return g[['ClientID', 'ProductID', 'label', 'target']]

def _client_pref_category(history:pd.DataFrame) -> pd.Series:
    """For each client pick their most frequent category in history (ties break arbitarily).

    Args:
        history (pd.DataFrame): DataFrame containing transactions in the history window.

    Returns:
        pd.DataFrame: Series with ClientID as index and most frequent category as values.
    """
    if "Category" not in history.columns:
        raise ValueError("History DataFrame must contain 'Category' column.")
    
    tmp = (
        history.dropna(subset=['Category'])
        .groupby(['ClientID', 'Category'], observed=True)
        .size()
        .reset_index(name='cnt')
        .sort_values(['ClientID', 'cnt'], ascending=[True, False])
    )

    top = tmp.drop_duplicates(subset= 'ClientID', keep='first')
    return top.set_index('ClientID')['Category']

# --------------------------------------------------------------------------------------
# Precomputations for fast negative sampling
# --------------------------------------------------------------------------------------
def _precompute_maps(history: pd.DataFrame) -> dict:
    """
    Precompute structures used inside the client loop, once:
      - all_products: np.array of unique ProductID
      - seen_map: dict[int -> np.array(ProductID)] products seen by client in history
      - prod_pop_global: np.array(ProductID) sorted by global popularity desc
      - prod_to_cat: dict[int -> str]
      - cat_to_pop: dict[str -> np.array(ProductID)] popular-by-category in global order
      - pref_cat_map: dict[int -> str] client's preferred category
    """
    # Popularity (txn counts)
    pop_counts = history.groupby("ProductID")["txn_date"].count()
    prod_pop_global = pop_counts.sort_values(ascending=False).index.to_numpy()

    # Product -> Category (take last seen category per product)
    prod_to_cat_ser = (
        history.dropna(subset=["Category"])
        .drop_duplicates(["ProductID"])
        .set_index("ProductID")["Category"]
    )
    prod_to_cat = prod_to_cat_ser.to_dict()

    # Category -> product set
    cat_groups_df = (
        history.dropna(subset=["Category"])[["ProductID", "Category"]]
        .drop_duplicates()
    )
    cat_groups = cat_groups_df.groupby("Category")["ProductID"].apply(set).to_dict()

    # Category -> popular products (respect global popularity order)
    cat_to_pop: dict[str, np.ndarray] = {}
    prod_pop_set = set(prod_pop_global.tolist())
    for cat, pid_set in cat_groups.items():
        # keep only pids that are in global popularity & that belong to this category
        ordered = [pid for pid in prod_pop_global if pid in pid_set]
        cat_to_pop[cat] = np.array(ordered, dtype=prod_pop_global.dtype)

    # Client preferred category (mode by txn count)
    tmp = (
        history.dropna(subset=["Category"])
        .groupby(["ClientID", "Category"], observed=True)
        .size()
        .reset_index(name="cnt")
        .sort_values(["ClientID", "cnt"], ascending=[True, False])
    )
    pref_cat_map = dict(
        tmp.drop_duplicates("ClientID", keep="first")[["ClientID", "Category"]]
        .to_records(index=False)
    )

    # Seen products per client
    seen_map = {
        cid: arr for cid, arr in history.groupby("ClientID")["ProductID"].unique().items()
    }

    all_products = history["ProductID"].drop_duplicates().to_numpy()

    return {
        "all_products": all_products,
        "seen_map": seen_map,
        "prod_pop_global": prod_pop_global,
        "prod_to_cat": prod_to_cat,
        "cat_to_pop": cat_to_pop,
        "pref_cat_map": pref_cat_map,
    }

# --------------------------------------------------------------------------------------
# Fast hard-negative sampler
# --------------------------------------------------------------------------------------
def _choose_negatives_for_client_fast(
    pos_products: np.ndarray,
    seen_products: np.ndarray,
    all_products: np.ndarray,
    n_negs: int,
    rng: np.random.Generator,
    hard_mode: HardNegMode,
    prod_pop_global: np.ndarray,
    pref_cat: str | None,
    cat_to_pop: dict[str, np.ndarray],
) -> np.ndarray:
    """
    Vectorized negative sampling:
      - 'none'            : random from universe (unseen & not-positive)
      - 'pop'             : most popular from universe
      - 'pop_in_prefcat'  : popular within client's preferred category; then backfill global popular
    """
    if n_negs <= 0:
        return np.empty(0, dtype=all_products.dtype)

    # Universe = all_products \ (positives ∪ seen)
    if seen_products.size:
        blocked = np.unique(np.concatenate([pos_products, seen_products]))
    else:
        blocked = pos_products
    universe = np.setdiff1d(all_products, blocked, assume_unique=False)
    if universe.size == 0:
        return np.empty(0, dtype=all_products.dtype)

    if hard_mode == "none":
        k = min(n_negs, universe.size)
        return rng.choice(universe, size=k, replace=False)

    # Intersect universe with global popularity order (preserve order)
    universe_set = set(universe.tolist())
    pop_universe = np.array([pid for pid in prod_pop_global if pid in universe_set], dtype=all_products.dtype)
    if pop_universe.size == 0:
        return np.empty(0, dtype=all_products.dtype)

    if hard_mode == "pop":
        return pop_universe[: min(n_negs, pop_universe.size)]

    # hard_mode == "pop_in_prefcat"
    chosen: list[int] = []
    if pref_cat and pref_cat in cat_to_pop:
        in_cat = cat_to_pop[pref_cat]
        # keep those also in universe
        in_cat_univ = np.array([pid for pid in in_cat if pid in universe_set], dtype=all_products.dtype)
        if in_cat_univ.size:
            take = min(n_negs, in_cat_univ.size)
            chosen.extend(in_cat_univ[:take].tolist())

    if len(chosen) < n_negs:
        remaining = [pid for pid in pop_universe if pid not in chosen]
        need = n_negs - len(chosen)
        chosen.extend(remaining[:need])

    return np.array(chosen[:n_negs], dtype=all_products.dtype)

def _build_candidates(
    history: pd.DataFrame,
    window: pd.DataFrame,
    negatives_per_pos: int,
    hard_mode: HardNegMode,
    seed: int,
    sample_clients: int | None = None,
    cap_negatives: int | None = None,
) -> pd.DataFrame:
    """
    Return candidate pairs with labels:
      - positives: unique (ClientID, ProductID) in window (label=1)
      - negatives: K per positive using chosen hard-negative policy (label=0)
    """
    rng = np.random.default_rng(seed)

    pos_pairs = window[["ClientID", "ProductID"]].drop_duplicates()
    if pos_pairs.empty:
        return pos_pairs.assign(label=pd.Series(dtype=int))

    # Precompute maps/arrays once
    maps = _precompute_maps(history)
    all_products = maps["all_products"]
    seen_map = maps["seen_map"]
    prod_pop_global = maps["prod_pop_global"]
    pref_cat_map = maps["pref_cat_map"]
    cat_to_pop = maps["cat_to_pop"]

    # positives per client
    pos_map = pos_pairs.groupby("ClientID")["ProductID"].unique().to_dict()
    client_ids = np.array(list(pos_map.keys()))

    # dev-mode: limit number of clients
    if sample_clients is not None and sample_clients < client_ids.size:
        rng.shuffle(client_ids)
        client_ids = client_ids[:sample_clients]

    rows = [pos_pairs.assign(label=1)]
    for cid in client_ids:
        pos_prods = np.array(pos_map[cid], dtype=all_products.dtype)
        seen_prods = np.array(seen_map.get(cid, np.empty(0, dtype=all_products.dtype)), dtype=all_products.dtype)
        n_negs = negatives_per_pos * len(pos_prods)
        if cap_negatives is not None:
            n_negs = min(n_negs, cap_negatives)

        client_pref = pref_cat_map.get(cid, None)
        neg = _choose_negatives_for_client_fast(
            pos_products=pos_prods,
            seen_products=seen_prods,
            all_products=all_products,
            n_negs=n_negs,
            rng=rng,
            hard_mode=hard_mode,
            prod_pop_global=prod_pop_global,
            pref_cat=client_pref,
            cat_to_pop=cat_to_pop,
        )
        if neg.size:
            rows.append(pd.DataFrame({"ClientID": cid, "ProductID": neg, "label": 0}))

    return pd.concat(rows, ignore_index=True)

# ---------------------------------------------------------------------------------------------------
# Feature Artifact IO
# ---------------------------------------------------------------------------------------------------
def _dated_artifact(name: str, cutoff: str) -> Path:
    """Your builders save dated files like '<name>_YYYY-MM-DD.parquet' when cutoff is provided."""
    return PROC / f"{name}_{pd.to_datetime(cutoff).date()}.parquet"


def _ensure_and_load_features(cutoff: pd.Timestamp) -> Dict[str, pd.DataFrame]:
    """
    Ensure features for the cutoff exist (build if missing), then load them.
    Returns dict: 'user', 'product', 'cp', 'category'.
    """
    cutoff_str = str(cutoff.date())

    p_user = _dated_artifact("user_features", cutoff_str)
    p_prod = _dated_artifact("product_features", cutoff_str)
    p_cp = _dated_artifact("client_product_recency", cutoff_str)
    p_cat = _dated_artifact("category_features", cutoff_str)

    if not p_user.exists():
        build_user_features(cutoff=cutoff_str)
    if not p_prod.exists():
        build_product_features(cutoff=cutoff_str)
    if not p_cp.exists():
        build_client_product_recency(cutoff=cutoff_str)
    if not p_cat.exists():
        build_category_features(cutoff=cutoff_str)

    return {
        "user": pd.read_parquet(p_user),
        "product": pd.read_parquet(p_prod),
        "cp": pd.read_parquet(p_cp),
        "category": pd.read_parquet(p_cat),
    }


def _assemble_dataset_from_artifacts(
    candidates: pd.DataFrame,
    window: pd.DataFrame,
    pos_mode: PosTargetMode,
    cutoff: pd.Timestamp,
) -> pd.DataFrame:
    """
    Merge leakage-safe features (precomputed up to cutoff) into candidate pairs.
    Adds:
      - 'label' always (binary positive/negative)
      - 'target' depending on pos_mode:
           binary   -> 1 for positives, 0 for negatives
           count_txn-> #txns in window (positives), 0 for negatives
           sum_qty  -> sum qty in window (positives), 0 for negatives
    """
    feats = _ensure_and_load_features(cutoff)
    uf = feats["user"].copy()
    pf = feats["product"].copy()
    cpf = feats["cp"].copy()
    catf = feats["category"].copy()

    # Calibrated positives from window (has target for positives)
    pos_cal = _positives(window, mode=pos_mode)[["ClientID", "ProductID", "target"]]

    # Prefix columns (except keys & product identity columns)
    u_ren = {c: f"u_{c}" for c in uf.columns if c not in ["ClientID"]}
    uf = uf.rename(columns=u_ren)

    p_exclude = {"ProductID", "Category", "FamilyLevel1", "FamilyLevel2", "Universe"}
    p_ren = {c: f"p_{c}" for c in pf.columns if c not in p_exclude}
    pf = pf.rename(columns=p_ren)

    cp_ren = {c: f"cp_{c}" for c in cpf.columns if c not in ["ClientID", "ProductID"]}
    cpf = cpf.rename(columns=cp_ren)

    df = (
        candidates
        .merge(uf, on="ClientID", how="left")
        .merge(pf, on="ProductID", how="left")
        .merge(catf, on="ProductID", how="left")
        .merge(cpf, on=["ClientID", "ProductID"], how="left")
        .merge(pos_cal, on=["ClientID", "ProductID"], how="left")
    )

    # Fill calibrated target for negatives
    if pos_mode == "binary":
        df["target"] = df["label"].astype(int)
    else:
        df["target"] = df["target"].fillna(0)

    # Fill pair-features for unseen pairs
    fill_defaults = {
        "cp_txns": 0,
        "cp_qty_sum": 0,
        "cp_eur_sum": 0,
        "cp_tenure_days": 0,
        "cp_days_since_last_cp": 9999,
    }
    for col, val in fill_defaults.items():
        if col in df.columns:
            df[col] = df[col].fillna(val)

    # Clip obvious non-negatives across known prefixes
    for col in df.columns:
        if any(col.startswith(prefix) for prefix in ("u_", "p_", "cp_", "cat_")):
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].clip(lower=0)

    return df


# --------------------------------------------------------------------------------------
# Build one split
# --------------------------------------------------------------------------------------
def build_split_dataset(cfg: SplitCfg) -> Tuple[pd.DataFrame, dict]:
    """
    Build one split (train/val/test):
      1) Carve history and label window based on cutoff & label_days.
      2) Build positives and sampled negatives (with chosen hard-negative policy).
      3) Merge in features computed by your existing feature builders (up to cutoff).
      4) Save parquet + return a small summary dict.
    """
    full = load_interactions()
    history, window = _history_window(full, cfg.cutoff, cfg.label_days)

    # Candidates (positives + negatives)
    candidates = _build_candidates(
        history=history,
        window=window,
        negatives_per_pos=cfg.negatives_per_pos,
        hard_mode=cfg.hard_negatives,
        seed=cfg.random_state,
        sample_clients=cfg.sample_clients,
        cap_negatives=cfg.cap_negatives,
    )
    if candidates.empty:
        raise ValueError(
            f"No positives found in label window for split '{cfg.name}'. "
            f"Try increasing label_days or adjusting cutoff."
        )

    # Merge with precomputed features and calibrated target
    ds = _assemble_dataset_from_artifacts(
        candidates=candidates,
        window=window,
        pos_mode=cfg.pos_target,
        cutoff=cfg.cutoff,
    )

    # Persist
    out_path = MODEL_DIR / f"{cfg.name}.parquet"
    ds.to_parquet(out_path, index=False)

    # Summary
    summary = {
        "split": cfg.name,
        "cutoff": str(cfg.cutoff),
        "label_days": cfg.label_days,
        "rows": int(len(ds)),
        "clients": int(ds["ClientID"].nunique()),
        "products": int(ds["ProductID"].nunique()),
        "positives": int((ds["label"] == 1).sum()),
        "negatives": int((ds["label"] == 0).sum()),
        "neg_per_pos": float(cfg.negatives_per_pos),
        "hard_negatives": cfg.hard_negatives,
        "pos_target": cfg.pos_target,
    }
    return ds, summary


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Build labeled datasets (reuse existing feature builders).")
    ap.add_argument("--train-end", required=True, help="YYYY-MM-DD cutoff for train history; labels after this date")
    ap.add_argument("--val-end",   required=True, help="YYYY-MM-DD cutoff for val history; labels after this date")
    ap.add_argument("--test-end",  required=True, help="YYYY-MM-DD cutoff for test history; labels after this date")
    ap.add_argument("--label-days", type=int, default=30, help="Label window size Δ in days (default: 30)")
    ap.add_argument("--neg-per-pos", type=int, default=5, help="Negatives per positive (default: 5)")
    ap.add_argument("--hard-negatives", type=str, default="pop_in_prefcat",
                    choices=["none", "pop", "pop_in_prefcat"],
                    help="Negative sampling policy")
    ap.add_argument("--pos-target", type=str, default="binary",
                    choices=["binary", "count_txn", "sum_qty"],
                    help="Positive target definition inside the window")
    ap.add_argument("--seed", type=int, default=42)
    # Dev-mode speed knobs
    ap.add_argument("--sample-clients", type=int, default=None, help="Limit number of clients processed (dev)")
    ap.add_argument("--cap-negatives", type=int, default=None, help="Cap negatives per client (dev)")
    args = ap.parse_args()

    cfgs = [
        SplitCfg(
            "train", pd.to_datetime(args.train_end), args.label_days, args.neg_per_pos,
            hard_negatives=args.hard_negatives, pos_target=args.pos_target, random_state=args.seed,
            sample_clients=args.sample_clients, cap_negatives=args.cap_negatives
        ),
        SplitCfg(
            "val", pd.to_datetime(args.val_end), args.label_days, args.neg_per_pos,
            hard_negatives=args.hard_negatives, pos_target=args.pos_target, random_state=args.seed,
            sample_clients=args.sample_clients, cap_negatives=args.cap_negatives
        ),
        SplitCfg(
            "test", pd.to_datetime(args.test_end), args.label_days, args.neg_per_pos,
            hard_negatives=args.hard_negatives, pos_target=args.pos_target, random_state=args.seed,
            sample_clients=args.sample_clients, cap_negatives=args.cap_negatives
        ),
    ]

    rows = []
    for cfg in cfgs:
        print(
            f"[build] split={cfg.name} cutoff={cfg.cutoff.date()} Δ={cfg.label_days}d "
            f"neg/pos={cfg.negatives_per_pos} hard={cfg.hard_negatives} target={cfg.pos_target} "
            f"sample_clients={cfg.sample_clients} cap_negs={cfg.cap_negatives}"
        )
        _, summary = build_split_dataset(cfg)
        rows.append(summary)

    rep = pd.DataFrame(rows).sort_values("split")
    out = REPORTS / "label_sampling_summary.csv"
    rep.to_csv(out, index=False)
    print(f"[OK] wrote {out}\n{rep}")


if __name__ == "__main__":
    main()
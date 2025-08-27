from __future__ import annotations

import argparse
from pathlib import Path
import json
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from joblib import dump

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import (
    average_precision_score,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
SPLITS = PROC / "model"
REPORTS = PROC / "reports"
MODELS_DIR = PROC / "models"
REPORTS.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------------------
# Utility Functions
# -------------------------------------------------------------------------
def load_split(name: str) -> pd.DataFrame:
    path = SPLITS / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Split file not found: {path}")
    return pd.read_parquet(path)

def pick_feature_columns(df: pd.DataFrame) -> List[str]:
    """Return numeric feature columns (exclude IDs/ labels/ target and obvious text columns)"""
    drop = {
        "ClientID", "ProductID", "label", "target",
        "Category", "FamilyLevel1", "FamilyLevel2", "Universe",
    }

    cols = [c for c in df.columns if c not in drop]
    cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    return cols

def split_xy(df: pd.DataFrame, feat_cols: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    " Return X, y and groups (ClientID for ranking metrics)"
    X = df[feat_cols]
    y = df["label"].astype(int).to_numpy()
    groups = df["ClientID"].to_numpy()
    return X, y, groups

def group_recall_at_k(y_true: np.ndarray, y_score: np.ndarray, groups: np.ndarray, k: int = 10) -> float:
    """
    Recall@K computed per-client and averaged:
    recall_k(client) = (# positives ranked in top-K) / (min(K, #positives for client in cand set))
    If a client has 0 positives in the candidate set, they are skipped.
    """
    df = pd.DataFrame({"g" : groups, "y": y_true, "p": y_score})
    out = []
    for gid, gdf in df.groupby("g"):
        gdf = gdf.sort_values("p", ascending=False)
        topk = gdf.head(k)
        pos_total = gdf["y"].sum()
        if pos_total == 0:
            continue
        hit = topk["y"].sum()
        denom = min(k, pos_total)
        out.append(hit / denom)
    return float(np.mean(out)) if out else 0.0

def map_at_k(y_true: np.ndarray, y_score: np.ndarray, groups: np.ndarray, k: int = 10) -> float:
    """
    MAP@K: mean of per-group average precision at K.
    AP@K(client) = sum_{i=1..K} (Precision@i * rel_i) / min(K, #positives in group),
    where rel_i is 1 if i-th item is positive else 0.
    """
    df = pd.DataFrame({"g" : groups, "y": y_true, "p": y_score})
    ap_list = []
    for gid, gdf in df.groupby("g"):
        gdf = gdf.sort_values("p", ascending=False).head(k)
        y = gdf["y"].to_numpy()
        if y.sum() == 0:
            continue
        precision = []
        hits = 0
        for i in range(len(y)):
            if y[i] == 1:
                hits += 1
                precision.append(hits / (i + 1))
        denom = min(k, int(y.sum()))
        ap_k = (np.sum(precision) / denom) if denom > 0 else 0.0
        ap_list.append(ap_k)
    return float(np.mean(ap_list)) if ap_list else 0.0

def scale_pos_weight(y: np.ndarray) -> float:
    """ Compute the scale_pos_weight parameter for imbalanced datasets."""
    pos = (y == 1).sum()
    neg = (y == 0).sum()
    if pos == 0:
        return 1.0
    return float(neg / pos)

def _load_ranked_features() -> List[str] | None:
    " Return ranked feature names from feature selection script"
    path = REPORTS / "feature_importance_lgbm.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "feature" not in df.columns or df.empty:
        return None
    return df["feature"].to_list()

def _load_best_k(default: int | None = None) -> int | None:
    "Return the best_k featrures if present, else default"
    path = REPORTS / "feature_selection_best_k.json"
    if not path.exists():
        return default
    try:
        meta = json.loads(path.read_text())
        k = int(meta.get("best_k", 0))
        return k if k > 0 else default
    except Exception:
        return default

def _numeric_feature_columns(df: pd.DataFrame) -> list[str]:
    """Fallback: choose numeric columns except IDs/targets/date."""
    drop = {
        "ClientID", "ProductID", "StoreID",
        "label", "target",
        "SaleTransactionDate", "txn_date",
    }
    return [c for c in df.columns if c not in drop and pd.api.types.is_numeric_dtype(df[c])]

def _select_features(df: pd.DataFrame) -> list[str]:
    """
    Main entry: prefer ranked top-k if available; else all numeric.
    """
    ranked = _load_ranked_features()
    best_k = _load_best_k()
    if ranked and best_k:
        # only keep columns that actually exist in df
        return [c for c in ranked[:best_k] if c in df.columns]
    # Fallbacks
    if ranked:
        return [c for c in ranked if c in df.columns]
    return _numeric_feature_columns(df)

# -------------------------------------------------------------------------
# Model Factory
# -------------------------------------------------------------------------
def make_log_reg_pipeline(num_cols : List[str])-> Pipeline:
    """ Logistic Regression with basic preprocessing."""
    pre = ColumnTransformer(
        transformers = [("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                           ("scale", StandardScaler())]), num_cols)],
        remainder="drop",
        n_jobs= None,
    )
    clf = LogisticRegression(
        solver = "lbfgs",
        max_iter = 100,
        class_weight = "balanced",
        n_jobs = None,)
    return Pipeline([("prep", pre), ("clf", clf)])

def make_rf_pipeline(num_cols : List[str]) -> Pipeline:
    """ Random Forest with basic preprocessing."""
    pre = ColumnTransformer(
        transformers = [("num",SimpleImputer(strategy="median"), num_cols)],
        remainder="drop",
    )
    clf = RandomForestClassifier(
        n_estimators = 400,
        max_depth = None,
        min_samples_leaf= 2,
        class_weight = "balanced_subsample",
        n_jobs = -1,
        random_state = 42,
        )
    return Pipeline([("prep", pre), ("clf", clf)])

def make_xgb_pipeline(num_cols : List[str], y_train : np.ndarray) -> Pipeline:
    """ XGBoost with basic preprocessing."""
    pre = ColumnTransformer(
        transformers = [("num", SimpleImputer(strategy="median"), num_cols)],
        remainder="drop",
    )
    clf = xgb.XGBClassifier(
        n_estimators = 600,
        max_depth = 8,
        learning_rate = 0.05,
        subsample = 0.8,
        colsample_bytree = 0.8,
        reg_lambda = 1.0,
        tree_method = "hist",
        use_label_encoder = False,
        eval_metric = "logloss",
        scale_pos_weight = scale_pos_weight(y_train),
        n_jobs = -1,
        random_state = 42,)
    return Pipeline([("prep", pre), ("clf", clf)])

def make_lgbm_pipeline(num_cols : List[str], y_train : np.ndarray) -> Pipeline:
    """ LightGBM with basic preprocessing."""
    pre = ColumnTransformer(
        transformers = [("num", SimpleImputer(strategy="median"), num_cols)],
        remainder="drop",
    )
    clf = lgb.LGBMClassifier(
        n_estimators = 800,
        num_leaves= 63,
        learning_rate = 0.05,
        subsample = 0.8,
        colsample_bytree = 0.8,
        reg_lambda = 1.0,
        class_weight = None,
        n_jobs = -1,
        random_state = 42,)
    return Pipeline([("prep", pre), ("clf", clf)])

# -------------------------------------------------------------------------
# Baseline Registry and Helper Functions
# -------------------------------------------------------------------------

def get_feature_selector():
    "Expose the same column selector used for training"
    return _select_features

MODEL_REGISTRY = {
    "logreg" : make_log_reg_pipeline,
    "random_forest" : make_rf_pipeline,
    "xgboost" : make_xgb_pipeline,
    "lightgbm" : make_lgbm_pipeline,
}

def get_model_names() -> List[str]:
    "Return available model names."
    return list(MODEL_REGISTRY.keys())

def make_pipeline_by_name(name: str, num_cols: list[str], y_train=None):
    """
    Build a pipeline by name using the same constructors as baselines.
    xgboost/lightgbm accept y_train for class-weighting; others ignore it.
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}")
    builder = MODEL_REGISTRY[name]
    try:
        # prefer signature (num_cols, y_train) when provided
        return builder(num_cols, y_train)  # xgb/lgbm path
    except TypeError:
        # fall back to (num_cols) for models that don't take y_train
        return builder(num_cols)

def read_baseline_leaderboard(report_path: Path) -> pd.DataFrame:
    """Read the CSV leaderboard written by this script."""
    if not report_path.exists():
        raise FileNotFoundError(f"Missing leaderboard CSV: {report_path}")
    return pd.read_csv(report_path)

def top_n_models_from_leaderboard(report_path: Path, metric: str = "pr_auc", top_n: int = 2) -> list[str]:
    """Pick the top-N model names by a metric from the leaderboard CSV."""
    df = read_baseline_leaderboard(report_path)
    if "model" not in df.columns or metric not in df.columns:
        raise ValueError(f"{report_path} must have columns ['model', '{metric}']")
    return df.sort_values(metric, ascending=False)["model"].head(top_n).tolist()

# -------------------------------------------------------------------------
# Train / Evaluation Loop
# -------------------------------------------------------------------------

def eval_metrics(y_true: np.ndarray, proba: np.ndarray, groups: np.ndarray) -> Dict[str, float]:
    """Return ROC-AUC, PR-AUC, LogLoss, Recall@K and MAP@K."""
    # Guard against pathological cases
    try:
        roc = roc_auc_score(y_true, proba)
    except Exception:
        roc = float("nan")
    try:
        ap = average_precision_score(y_true, proba)
    except Exception:
        ap = float("nan")
    try:
        ll = log_loss(y_true, proba, labels=[0, 1])
    except Exception:
        ll = float("nan")

    r10 = group_recall_at_k(y_true, proba, groups, k=10)
    map10 = map_at_k(y_true, proba, groups, k=10)

    # Find the operating point of PR curve that gives best F1
    try:
        prec, rec, thr = precision_recall_curve(y_true, proba)
        f1 = (2 * prec * rec / np.maximum(prec + rec, 1e-12)).max()
    except Exception:
        f1 = float("nan")

    return {
        "roc_auc": roc,
        "pr_auc": ap,
        "logloss": ll,
        "recall@10": r10,
        "map@10": map10,
        "best_f1_on_pr": f1,
    }

def fit_and_score(
        name: str,
        pipe: Pipeline,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val : np.ndarray,
        y_val : np.ndarray,
        groups_val: np.ndarray,
) -> Dict[str, float]:
    """ Fit the pipeline and return evaluation metrics on validation set."""
    pipe.fit(X_train, y_train)
    proba = pipe.predict_proba(X_val)[:, 1]
    m = eval_metrics(y_val, proba, groups_val)
    m["model"] = name
    return m, pipe

# -------------------------------------------------------------------------
# Main Functions
# -------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Train 4 baselines and pick best on validation.")
    ap.add_argument("--metric", default="pr_auc", choices=["pr_auc", "roc_auc", "map@10", "recall@10"],
                    help="Model selection metric on validation.")
    ap.add_argument("--save-name", default="best_baseline.pkl", help="Filename for the best model.")
    args = ap.parse_args()

    # Load splits
    train = load_split("train")
    val = load_split("val")
    test = load_split("test")

    # Features/labels
    feat_cols = _select_features(train)
    X_tr, y_tr, g_tr = split_xy(train, feat_cols)
    X_va, y_va, g_va = split_xy(val, feat_cols)
    X_te, y_te, g_te = split_xy(test, feat_cols)

    # Build models
    models: List[Tuple[str, Pipeline | None]] = [
        ("logreg", make_log_reg_pipeline(feat_cols)),
        ("random_forest", make_rf_pipeline(feat_cols)),
        ("xgboost", make_xgb_pipeline(feat_cols, y_tr)),
        ("lightgbm", make_lgbm_pipeline(feat_cols, y_tr)),
    ]

    results: List[Dict[str, float]] = []
    fitted: Dict[str, Pipeline] = {}

    for name, pipe in models:
        if pipe is None:
            print(f"[skip] {name}: library not installed.")
            continue
        print(f"[fit] {name} …")
        m, fitted_pipe = fit_and_score(name, pipe, X_tr, y_tr, X_va, y_va, g_va)
        results.append(m)
        fitted[name] = fitted_pipe
        print(f"       val metrics: {json.dumps(m, indent=2)}")

    if not results:
        raise RuntimeError("No models were trained. Check optional dependencies.")

    # Pick best by chosen metric
    metric_key = args.metric
    def key_fn(d: Dict[str, float]) -> float:
        return d["pr_auc"] if metric_key == "pr_auc" else (
            d["roc_auc"] if metric_key == "roc_auc" else (
                d["map@10"] if metric_key == "map@10" else d["recall@10"]
            )
        )

    best = max(results, key=key_fn)
    best_name = best["model"]
    best_pipe = fitted[best_name]
    print(f"[best] {best_name} by {metric_key}: {key_fn(best):.6f}")

    # Evaluate on test
    proba_te = best_pipe.predict_proba(X_te)[:, 1]
    test_metrics = eval_metrics(y_te, proba_te, g_te)
    print(f"[test] {best_name}: {json.dumps(test_metrics, indent=2)}")

    # Save artifacts
    metrics_df = pd.DataFrame(results).sort_values(metric_key, ascending=False)
    metrics_df.to_csv(REPORTS / "baseline_val_metrics.csv", index=False)
    pd.Series(test_metrics).to_json(REPORTS / "baseline_test_metrics.json", indent=2)
    dump(best_pipe, MODELS_DIR / args.save_name)
    print(f"[OK] saved best model → {MODELS_DIR / args.save_name}")
    print(f"[OK] wrote reports → {REPORTS / 'baseline_val_metrics.csv'}, {REPORTS / 'baseline_test_metrics.json'}")


if __name__ == "__main__":
    main()
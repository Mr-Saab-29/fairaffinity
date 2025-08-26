from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from src.models.train_baselines import load_split
import matplotlib
matplotlib.use("Agg") # Use a non-interactive backend
import matplotlib.pyplot as plt

# Paths
ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
SPLIT = PROC / "model"
REPORTS = PROC / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------
# Helper Functions
# ----------------------------------------------------------------------
def _feature_columns(df: pd.DataFrame) -> List[str]:
    """ Numeric features only (Drop High cardinality features)"""
    drop = {
        "ClientID", "ProductID", "StoreID", "label", "target", "txn_date"
    }
    num_cols = [
        c for c in df.columns if c not in drop and pd.api.types.is_numeric_dtype(df[c])
    ]
    return num_cols

def _fit_lgbm_rank_features(
        X: pd.DataFrame, y: np.ndarray, groups: np.ndarray | None = None, random_state: int = 42
) -> pd.DataFrame:
    "Train a LightGBM model and return feature importances."
    clf = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=64,
        random_state=random_state,
        subsample=0.8,
        colsample_bytree=0.9,
        n_jobs=-1,
    )
    clf.fit(X, y)
    imp = pd.DataFrame(
        {"feature": X.columns, "importance": clf.booster_.feature_importance(importance_type="gain")}
    ).sort_values(by="importance", ascending=False).reset_index(drop=True)
    return imp

def _evaluate_with_top_k(
        X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, ranked_features: List[str], ks: List[int],
        cv_folds: int = 3, random_state: int = 42,
) -> pd.DataFrame:
    """ For each k, train the LGBM model on top k features and CV scores (PR_AUC, ROC_AUC)."""
    rows = []
    gkf = GroupKFold(n_splits=cv_folds)

    for k in ks:
        feats = ranked_features[:k]
        pr_scores, roc_scores = [], []
        for train_idx, val_idx in gkf.split(X[feats], y, groups = groups):
            X_train, X_val = X.iloc[train_idx][feats], X.iloc[val_idx][feats]
            y_train, y_val = y[train_idx], y[val_idx]

            clf = LGBMClassifier(
                n_estimators=400,
                learning_rate=0.05,
                num_leaves=96,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=random_state,
                n_jobs=-1,
            )
            clf.fit(X_train, y_train)
            p = clf.predict_proba(X_val)[:, 1]
            pr_scores.append(average_precision_score(y_val, p))
            roc_scores.append(roc_auc_score(y_val, p))
        
        rows.append({
            "k" : k,
            "features" : k,
            "pr_auc_mean" : float(np.mean(pr_scores)),
            "pr_auc_std" : float(np.std(pr_scores)),
            "roc_auc_mean" : float(np.mean(roc_scores)),
            "roc_auc_std" : float(np.std(roc_scores)),
        })
    
    return pd.DataFrame(rows).sort_values(by="k")

def run_feature_selection() -> None:
    # Ensure reports directory
    REPORTS.mkdir(parents=True, exist_ok=True)

    # 1) load split
    train = load_split("train")
    assert "label" in train.columns, "'label' column missing in train split"

    # 2) features
    feat_cols = _feature_columns(train)
    if not feat_cols:
        raise RuntimeError("No numeric feature columns found (after drops).")

    X_tr = train[feat_cols]
    y_tr = train["label"].astype(int).to_numpy()
    g_tr = train["ClientID"].to_numpy()

    # 3) rank features
    ranked = _fit_lgbm_rank_features(X_tr, y_tr)
    ranked_path = REPORTS / "feature_importance_lgbm.csv"
    ranked.to_csv(ranked_path, index=False)
    if ranked.empty:
        return  # nothing to do

    # 4) evaluate curve
    ks = list(range(5, min(51, len(ranked) + 1), 5)) or [min(5, len(ranked))]
    curve = _evaluate_with_top_k(
        X_tr, y_tr, g_tr, ranked["feature"].tolist(), ks, cv_folds=3
    )
    curve_path = REPORTS / "feature_selection_curve.csv"
    curve.to_csv(curve_path, index=False)
    if curve.empty:
        return  # nothing to plot

    # 5) best k (saved for downstream use if needed)
    best_k = int(curve.loc[curve['pr_auc_mean'].idxmax(), 'k'])
    print(f"[info] best k by PR-AUC: {best_k}")

    plt.figure(figsize=(8, 6))
    plt.plot(curve["k"], curve["pr_auc_mean"], marker="o", label="PR-AUC")
    plt.plot(curve["k"], curve["roc_auc_mean"], marker="s", label="ROC-AUC")
    plt.fill_between(curve["k"],
                     curve["pr_auc_mean"] - curve["pr_auc_std"],
                     curve["pr_auc_mean"] + curve["pr_auc_std"],
                     alpha=0.2)
    plt.fill_between(curve["k"],
                     curve["roc_auc_mean"] - curve["roc_auc_std"],
                     curve["roc_auc_mean"] + curve["roc_auc_std"],
                     alpha=0.2)
    plt.xlabel("# Features (top-k)")
    plt.ylabel("Score")
    plt.title("Feature selection curve (LightGBM)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)

    fig_path = REPORTS / "feature_curve_lgbm.png"
    plt.savefig(str(fig_path), dpi=200, bbox_inches="tight")
    plt.close()

def main() -> None:
    run_feature_selection()

if __name__ == "__main__":
    main()
from __future__ import annotations

import argparse
from pathlib import Path
import json
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np
import optuna as optuna
from joblib import dump
from sklearn.model_selection import GroupKFold
try:
    import mlflow
except Exception:
    mlflow = None

from src.models.train_baselines import (
    load_split,
    get_feature_selector,
    make_pipeline_by_name,
    top_n_models_from_leaderboard,
    eval_metrics,
    split_xy,
)
from src.fairness.training import (
    attach_client_groups,
    compute_sample_weights,
    compute_sample_weights_from_groups,
    load_client_groups,
)

# -----------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
SPLIT = PROC / "model"
REPORTS = PROC / "reports"
MODELS = PROC / "models"
REPORTS.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)

LEADERBOARD = REPORTS / "baseline_val_metrics.csv"


def _mlflow_safe_metric_name(name: str) -> str:
    return name.replace("@", "_at_")


def _setup_mlflow(args) -> bool:
    if not args.mlflow:
        return False
    if mlflow is None:
        print("[mlflow] disabled: mlflow package not installed.")
        return False
    if args.mlflow_tracking_uri:
        mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    mlflow.set_experiment(args.mlflow_experiment)
    return True

# ----------------------------------------------------------------------
# Utility Functions
# ----------------------------------------------------------------------

def _scale_pos_weight(y: np.ndarray) -> float:
    "Calculate the imbalance ratio for the dataset."
    pos = (y == 1).sum()
    neg = (y == 0).sum()
    return float(neg / max(pos, 1))

def _suggest_params(trial:optuna.Trial, model: str, y_train : np.ndarray) -> Dict[str, object]:
    """
    Return a dict of Pipeline.set_params(...) keys/values for the model.
    Use 'clf__param' keys to set underlying classifier hyperparams.
    """
    if model == "logreg":
        return {
            "clf__C": trial.suggest_float("logreg_C", 1e-3, 10.0, log=True),
            "clf__max_iter": trial.suggest_int("logreg_max_iter", 200, 800),
            # class_weight balanced already inside pipeline; leave as-is
        }

    if model == "random_forest":
        return {
            "clf__n_estimators": trial.suggest_int("rf_n_estimators", 200, 900),
            "clf__max_depth": trial.suggest_int("rf_max_depth", 6, 40),
            "clf__min_samples_leaf": trial.suggest_int("rf_min_samples_leaf", 1, 10),
            "clf__min_samples_split": trial.suggest_int("rf_min_samples_split", 2, 20),
            "clf__max_features": trial.suggest_float("rf_max_features", 0.2, 1.0),
        }

    if model == "xgboost":
        spw = _scale_pos_weight(y_train)
        return {
            "clf__n_estimators": trial.suggest_int("xgb_n_estimators", 300, 1200),
            "clf__max_depth": trial.suggest_int("xgb_max_depth", 4, 12),
            "clf__learning_rate": trial.suggest_float("xgb_eta", 0.01, 0.2, log=True),
            "clf__subsample": trial.suggest_float("xgb_subsample", 0.5, 1.0),
            "clf__colsample_bytree": trial.suggest_float("xgb_colsample", 0.5, 1.0),
            "clf__reg_lambda": trial.suggest_float("xgb_l2", 0.0, 5.0),
            "clf__gamma": trial.suggest_float("xgb_gamma", 0.0, 5.0),
            "clf__scale_pos_weight": spw,
        }

    if model == "lightgbm":
        return {
            "clf__n_estimators": trial.suggest_int("lgbm_n_estimators", 300, 1200),
            "clf__num_leaves": trial.suggest_int("lgbm_num_leaves", 31, 255),
            "clf__learning_rate": trial.suggest_float("lgbm_eta", 0.01, 0.2, log=True),
            "clf__subsample": trial.suggest_float("lgbm_subsample", 0.6, 1.0),
            "clf__colsample_bytree": trial.suggest_float("lgbm_colsample", 0.6, 1.0),
            "clf__reg_lambda": trial.suggest_float("lgbm_l2", 0.0, 5.0),
            "clf__min_child_samples": trial.suggest_int("lgbm_min_child_samples", 10, 200),
        }

    raise ValueError(f"Unknown model for HPO: {model}")

def _cv_objective(
        trail : optuna.Trial,
        model_name : str,
        X : pd.DataFrame,
        y : np.ndarray,
        groups : np.ndarray,
        metric: str = "pr_auc",
        random_state: int = 42,
        fairness_group_values: np.ndarray | None = None,
        fairness_groups: list[str] | None = None,
        fairness_others_weight: float = 1.0,
) -> float:
    """ GroupKFold CV objective for Optuna.
    metric is one of {'pr_auc', 'roc_auc', 'map@10', 'recall@10'}"""
    # Build pipeline with the current featrures set
    pipe = make_pipeline_by_name(model_name, list(X.columns), y_train=y)

    #Suggest Hyperparameters
    params = _suggest_params(trail, model_name, y)
    pipe.set_params(**params)

    gkf = GroupKFold(n_splits=3)
    scores: List[float] = []

    for train_idx, val_idx in gkf.split(X, y, groups=groups):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        g_va = groups[val_idx]
        if fairness_group_values is not None:
            sw = compute_sample_weights_from_groups(
                groups=fairness_group_values[train_idx],
                eligible_groups=fairness_groups,
                others_weight=fairness_others_weight,
            )
            pipe.fit(X_train, y_train, clf__sample_weight=sw)
        else:
            pipe.fit(X_train, y_train)
        proba = pipe.predict_proba(X_val)[:, 1]
        m = eval_metrics(y_val, proba, groups = g_va)

        if metric == "pr_auc":
            scores.append(m["pr_auc"])
        elif metric == "roc_auc":
            scores.append(m["roc_auc"])
        elif metric == "map@10":
            scores.append(m["map@10"])
        elif metric == "recall@10":
            scores.append(m["recall@10"])
        else:
            raise ValueError(f"Unknown metric for HPO: {metric}")
    
    return float(np.mean(scores))

def _pick_models(top_n: int, metric :str) -> List[str]:
    if LEADERBOARD.exists():
        return top_n_models_from_leaderboard(LEADERBOARD, top_n=top_n, metric = metric)
    return ['lightgbm', 'xgboost'][:top_n]

# ----------------------------------------------------------------------
# Main HPO Function
# ----------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Optuna HPO for top-N baseline models.")
    ap.add_argument("--metric", default="pr_auc",
                    choices=["pr_auc", "roc_auc", "map@10", "recall@10"],
                    help="Optimization objective.")
    ap.add_argument("--top-n", type=int, default=2,
                    help="How many top models (from baseline leaderboard) to tune.")
    ap.add_argument("--n-trials", type=int, default=30,
                    help="Optuna trials per model.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fair-train", action="store_true", help="Enable fairness-aware sample weighting in HPO.")
    ap.add_argument("--fair-group-col", default="ClientGender", help="Demographic group column used for fairness weights.")
    ap.add_argument(
        "--fair-groups",
        default="Male,Female,Unisex",
        help="Comma-separated groups to balance with inverse-frequency weighting.",
    )
    ap.add_argument("--fair-others-weight", type=float, default=1.0, help="Weight assigned to non-target groups.")
    ap.add_argument("--mlflow", action="store_true", help="Enable MLflow tracking.")
    ap.add_argument("--mlflow-experiment", default="FairAffinity-HPO", help="MLflow experiment name.")
    ap.add_argument("--mlflow-tracking-uri", default=None, help="Optional MLflow tracking URI.")
    ap.add_argument("--mlflow-run-name", default=None, help="Optional parent run name.")
    args = ap.parse_args()
    mlflow_enabled = _setup_mlflow(args)

    # Load data once
    train = load_split("train")
    val = load_split("val")
    test = load_split("test")

    fairness_group_values: np.ndarray | None = None
    fairness_groups: list[str] | None = None
    sample_weight_tr: np.ndarray | None = None
    if args.fair_train:
        fairness_groups = [s.strip() for s in args.fair_groups.split(",") if s.strip()]
        cg = load_client_groups(group_col=args.fair_group_col)
        train = attach_client_groups(train, cg, args.fair_group_col)
        fairness_group_values = train[args.fair_group_col].astype(str).to_numpy()
        sample_weight_tr = compute_sample_weights(
            train,
            group_col=args.fair_group_col,
            eligible_groups=fairness_groups if fairness_groups else None,
            others_weight=args.fair_others_weight,
        )
        print(
            f"[fair-train] enabled | group_col={args.fair_group_col} | "
            f"groups={fairness_groups} | mean_w={float(np.mean(sample_weight_tr)):.4f}"
        )

    # Features: same selector used by baselines / feature_selection
    select_features = get_feature_selector()
    feat_cols = select_features(train)

    X_tr, y_tr, g_tr = split_xy(train, feat_cols)
    X_va, y_va, g_va = split_xy(val, feat_cols)
    X_te, y_te, g_te = split_xy(test, feat_cols)

    model_names = _pick_models(args.top_n, metric=args.metric)
    all_results = []

    parent_ctx = mlflow.start_run(run_name=args.mlflow_run_name) if mlflow_enabled else None
    try:
        if mlflow_enabled:
            mlflow.log_params(
                {
                    "metric": args.metric,
                    "top_n": args.top_n,
                    "n_trials": args.n_trials,
                    "seed": args.seed,
                    "fair_train": args.fair_train,
                    "fair_group_col": args.fair_group_col,
                    "fair_groups": args.fair_groups,
                    "fair_others_weight": args.fair_others_weight,
                }
            )

        for model_name in model_names:
            print(f"[hpo] model={model_name} | metric={args.metric} | trials={args.n_trials}")

            try:
                _ = make_pipeline_by_name(model_name, feat_cols, y_train=y_tr)
            except Exception as e:
                print(f"[hpo][skip] {model_name}: {e}")
                continue

            child_ctx = (
                mlflow.start_run(run_name=f"hpo_{model_name}", nested=True) if mlflow_enabled else None
            )
            try:
                # Create & run study
                study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=args.seed))
                study.optimize(
                    lambda t: _cv_objective(
                        t,
                        model_name,
                        X_tr,
                        y_tr,
                        g_tr,
                        metric=args.metric,
                        random_state=args.seed,
                        fairness_group_values=fairness_group_values,
                        fairness_groups=fairness_groups,
                        fairness_others_weight=args.fair_others_weight,
                    ),
                    n_trials=args.n_trials,
                    show_progress_bar=False,
                )

                best_params = study.best_trial.params
                # Translate best trial params to pipeline set_params dict
                best_set_params = _suggest_params(study.best_trial, model_name, y_tr)

                # Refit on full train, evaluate on val & test
                pipe = make_pipeline_by_name(model_name, feat_cols, y_train=y_tr)
                pipe.set_params(**best_set_params)
                if sample_weight_tr is not None:
                    pipe.fit(X_tr, y_tr, clf__sample_weight=sample_weight_tr)
                else:
                    pipe.fit(X_tr, y_tr)

                proba_va = pipe.predict_proba(X_va)[:, 1]
                m_val = eval_metrics(y_va, proba_va, g_va)
                m_val["model"] = model_name
                m_val["split"] = "val"

                proba_te = pipe.predict_proba(X_te)[:, 1]
                m_test = eval_metrics(y_te, proba_te, g_te)
                m_test["model"] = model_name
                m_test["split"] = "test"

                # Persist artifacts
                # 1) study summary
                study_df = study.trials_dataframe()
                study_csv = REPORTS / f"hpo_{model_name}_trials.csv"
                study_df.to_csv(study_csv, index=False)

                # 2) best params
                params_json = REPORTS / f"hpo_{model_name}_best_params.json"
                params_json.write_text(json.dumps(best_params, indent=2))

                # 3) fitted model
                model_pkl = MODELS / f"hpo_{model_name}.pkl"
                dump(pipe, model_pkl)

                # 4) metrics
                out_json = REPORTS / f"hpo_{model_name}_metrics.json"
                out_json.write_text(json.dumps({"val": m_val, "test": m_test, "best_params": best_params}, indent=2))

                print(f"[hpo][{model_name}] best trial value={study.best_value:.6f} | saved → {model_pkl.name}")
                print(f"[hpo][{model_name}] val={m_val[args.metric]:.6f} | test={m_test[args.metric]:.6f}")

                all_results.append(
                    {
                        "model": model_name,
                        **{f"val_{k}": v for k, v in m_val.items() if k not in ("model", "split")},
                        **{f"test_{k}": v for k, v in m_test.items() if k not in ("model", "split")},
                    }
                )
                if mlflow_enabled:
                    try:
                        mlflow.log_metric("best_trial_value", float(study.best_value))
                        mlflow.log_metrics(
                            {
                                _mlflow_safe_metric_name(f"val_{k}"): float(v)
                                for k, v in m_val.items()
                                if k not in ("model", "split")
                            }
                        )
                        mlflow.log_metrics(
                            {
                                _mlflow_safe_metric_name(f"test_{k}"): float(v)
                                for k, v in m_test.items()
                                if k not in ("model", "split")
                            }
                        )
                        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
                        mlflow.log_artifact(str(study_csv))
                        mlflow.log_artifact(str(params_json))
                        mlflow.log_artifact(str(out_json))
                        mlflow.log_artifact(str(model_pkl))
                    except Exception as e:
                        print(f"[mlflow][warn] model={model_name}: {e}")
            finally:
                if child_ctx is not None:
                    mlflow.end_run()

        # Leader table for HPO results
        if all_results:
            hpo_board = pd.DataFrame(all_results)
            out_board = REPORTS / "hpo_summary.csv"
            hpo_board.to_csv(out_board, index=False)
            print(f"[hpo] wrote leaderboard → {out_board}")
            if mlflow_enabled:
                mlflow.log_artifact(str(out_board))
    finally:
        if parent_ctx is not None:
            mlflow.end_run()


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from pathlib import Path
try:
    import mlflow
except Exception:
    mlflow = None

from src.recommender.pipeline import HybridWeights, run_recommendation_pipeline


def _mlflow_safe_metric_name(name: str) -> str:
    return name.replace("@", "_at_")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run hybrid recommendation + fairness re-ranking pipeline.")
    ap.add_argument("--split", required=True, choices=["train", "val", "test"])
    ap.add_argument(
        "--model-path",
        default="data/processed/models/hpo_lightgbm.pkl",
        help="Path to trained affinity model .pkl",
    )
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--group-col", default="ClientGender")
    ap.add_argument("--lambda-fairness", type=float, default=0.25)
    ap.add_argument(
        "--fairness-groups",
        default="Male,Female,Unisex",
        help="Comma-separated groups used for fairness optimization; reporting still includes all groups.",
    )
    ap.add_argument("--w-affinity", type=float, default=0.60)
    ap.add_argument("--w-collab", type=float, default=0.20)
    ap.add_argument("--w-content", type=float, default=0.15)
    ap.add_argument("--w-pop", type=float, default=0.05)
    ap.add_argument("--output-prefix", default=None)
    ap.add_argument("--mlflow", action="store_true", help="Enable MLflow tracking.")
    ap.add_argument("--mlflow-experiment", default="FairAffinity-Recommend", help="MLflow experiment name.")
    ap.add_argument("--mlflow-tracking-uri", default=None, help="Optional MLflow tracking URI.")
    ap.add_argument("--mlflow-run-name", default=None, help="Optional MLflow run name.")
    args = ap.parse_args()

    weights = HybridWeights(
        affinity=args.w_affinity,
        collaborative=args.w_collab,
        content=args.w_content,
        popularity=args.w_pop,
    )
    fairness_groups = [s.strip() for s in args.fairness_groups.split(",") if s.strip()]
    run_ctx = None
    if args.mlflow and mlflow is not None:
        if args.mlflow_tracking_uri:
            mlflow.set_tracking_uri(args.mlflow_tracking_uri)
        mlflow.set_experiment(args.mlflow_experiment)
        run_ctx = mlflow.start_run(run_name=args.mlflow_run_name)
        mlflow.log_params(
            {
                "split": args.split,
                "top_k": args.top_k,
                "group_col": args.group_col,
                "lambda_fairness": args.lambda_fairness,
                "fairness_groups": args.fairness_groups,
                "w_affinity": args.w_affinity,
                "w_collab": args.w_collab,
                "w_content": args.w_content,
                "w_pop": args.w_pop,
                "model_path": args.model_path,
            }
        )
    elif args.mlflow and mlflow is None:
        print("[mlflow] disabled: mlflow package not installed.")

    try:
        out = run_recommendation_pipeline(
            split_name=args.split,
            model_path=Path(args.model_path),
            top_k=args.top_k,
            group_col=args.group_col,
            lambda_fairness=args.lambda_fairness,
            fairness_groups=fairness_groups if fairness_groups else None,
            weights=weights,
            output_prefix=args.output_prefix,
        )
        if run_ctx is not None:
            for metric_name, metric_value in out["summary_pre"].items():
                mlflow.log_metric(_mlflow_safe_metric_name(f"pre_{metric_name}"), float(metric_value))
            for metric_name, metric_value in out["summary_post"].items():
                mlflow.log_metric(_mlflow_safe_metric_name(f"post_{metric_name}"), float(metric_value))
            mlflow.log_artifact(out["base_path"])
            mlflow.log_artifact(out["fair_path"])
            mlflow.log_artifact(out["exp_pre_path"])
            mlflow.log_artifact(out["exp_post_path"])
            mlflow.log_artifact(out["cat_pre_path"])
            mlflow.log_artifact(out["cat_post_path"])
            mlflow.log_artifact(out["summary_path"])
    finally:
        if run_ctx is not None:
            mlflow.end_run()
    print("[OK] recommendation artifacts written")
    print(f" - base recs: {out['base_path']}")
    print(f" - fair recs: {out['fair_path']}")
    print(f" - fairness summary: {out['summary_path']}")
    print(f" - fairness(pre): {out['summary_pre']}")
    print(f" - fairness(post): {out['summary_post']}")


if __name__ == "__main__":
    main()

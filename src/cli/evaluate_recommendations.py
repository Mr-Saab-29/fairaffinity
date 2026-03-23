from __future__ import annotations

import argparse
from pathlib import Path
try:
    import mlflow
except Exception:
    mlflow = None

from src.eval.recommendation_eval import compare_base_vs_fair, save_comparison_report


def _mlflow_safe_metric_name(name: str) -> str:
    return name.replace("@", "_at_")


def main() -> None:
    ap = argparse.ArgumentParser(description="Offline evaluator for base vs fairness-aware recommendations.")
    ap.add_argument("--base-path", required=True, help="Path to base recommendations parquet")
    ap.add_argument("--fair-path", required=True, help="Path to fairness-reranked recommendations parquet")
    ap.add_argument("--split-path", default=None, help="Path to split parquet with labels (defaults from --split)")
    ap.add_argument("--split", choices=["train", "val", "test"], default="val")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--output-dir", default="data/processed/reports/recommendations")
    ap.add_argument("--output-stem", default="comparison")
    ap.add_argument("--mlflow", action="store_true", help="Enable MLflow tracking.")
    ap.add_argument("--mlflow-experiment", default="FairAffinity-Eval", help="MLflow experiment name.")
    ap.add_argument("--mlflow-tracking-uri", default=None, help="Optional MLflow tracking URI.")
    ap.add_argument("--mlflow-run-name", default=None, help="Optional MLflow run name.")
    args = ap.parse_args()

    split_path = (
        Path(args.split_path)
        if args.split_path
        else Path(f"data/processed/model/{args.split}.parquet")
    )

    run_ctx = None
    if args.mlflow and mlflow is not None:
        if args.mlflow_tracking_uri:
            mlflow.set_tracking_uri(args.mlflow_tracking_uri)
        mlflow.set_experiment(args.mlflow_experiment)
        run_ctx = mlflow.start_run(run_name=args.mlflow_run_name)
        mlflow.log_params(
            {
                "base_path": args.base_path,
                "fair_path": args.fair_path,
                "split_path": str(split_path),
                "k": args.k,
                "output_stem": args.output_stem,
            }
        )
    elif args.mlflow and mlflow is None:
        print("[mlflow] disabled: mlflow package not installed.")

    try:
        comparison = compare_base_vs_fair(
            base_path=Path(args.base_path),
            fair_path=Path(args.fair_path),
            split_path=split_path,
            k=args.k,
        )
        json_path, csv_path = save_comparison_report(
            comparison=comparison,
            output_dir=Path(args.output_dir),
            stem=args.output_stem,
        )
        if run_ctx is not None:
            for k, v in comparison["base"].items():
                if k != "clients":
                    mlflow.log_metric(_mlflow_safe_metric_name(f"base_{k}"), float(v))
            for k, v in comparison["fair"].items():
                if k != "clients":
                    mlflow.log_metric(_mlflow_safe_metric_name(f"fair_{k}"), float(v))
            for k, v in comparison["delta"].items():
                mlflow.log_metric(_mlflow_safe_metric_name(f"delta_{k}"), float(v))
            mlflow.log_artifact(str(json_path))
            mlflow.log_artifact(str(csv_path))
    finally:
        if run_ctx is not None:
            mlflow.end_run()

    print("[OK] offline evaluation completed")
    print(f" - json: {json_path}")
    print(f" - csv: {csv_path}")
    print(" - base metrics:", comparison["base"])
    print(" - fair metrics:", comparison["fair"])
    print(" - delta:", comparison["delta"])


if __name__ == "__main__":
    main()

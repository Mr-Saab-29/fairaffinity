from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
REPORTS = PROC / "reports"
RECS = REPORTS / "recommendations"
REGISTRY = PROC / "model_registry"
REPORTS.mkdir(parents=True, exist_ok=True)
RECS.mkdir(parents=True, exist_ok=True)


def _infer_cutoffs(label_days: int) -> tuple[str, str, str]:
    p_collapsed = PROC / "interactions_collapsed_c_p_d_s.parquet"
    p_raw = PROC / "interactions.parquet"
    path = p_collapsed if p_collapsed.exists() else p_raw
    if not path.exists():
        raise FileNotFoundError("No interactions parquet found for cutoff inference.")
    df = pd.read_parquet(path, columns=["txn_date"])
    mx = pd.to_datetime(df["txn_date"], errors="coerce").max()
    if pd.isna(mx):
        raise ValueError("Unable to infer cutoffs: txn_date has no valid values.")
    test_end = mx - pd.Timedelta(days=label_days)
    val_end = test_end - pd.Timedelta(days=label_days)
    train_end = val_end - pd.Timedelta(days=label_days)
    return str(train_end.date()), str(val_end.date()), str(test_end.date())


def _run(cmd: list[str], steps: list[dict]) -> None:
    started = time.time()
    print(f"[orchestrator] RUN: {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=str(ROOT))
    duration = time.time() - started
    steps.append(
        {
            "command": cmd,
            "duration_sec": round(duration, 3),
            "returncode": proc.returncode,
        }
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run end-to-end FairAffinity MLOps pipeline.")
    ap.add_argument("--train-end", default=None)
    ap.add_argument("--val-end", default=None)
    ap.add_argument("--test-end", default=None)
    ap.add_argument("--label-days", type=int, default=30)
    ap.add_argument("--neg-per-pos", type=int, default=5)
    ap.add_argument("--hard-negatives", default="pop_in_prefcat", choices=["none", "pop", "pop_in_prefcat"])
    ap.add_argument("--hpo-trials", type=int, default=20)
    ap.add_argument("--hpo-top-n", type=int, default=2)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--lambda-fairness", type=float, default=0.35)
    ap.add_argument("--fairness-groups", default="Male,Female,Unisex")
    ap.add_argument("--mlflow", action="store_true")
    ap.add_argument("--run-tag", default=None, help="Prefix for recommendation/eval artifacts.")
    ap.add_argument("--promote", action="store_true", help="Promote best model after HPO.")
    ap.add_argument("--min-val-pr-auc", type=float, default=0.0)
    ap.add_argument("--min-val-map10", type=float, default=0.0)
    ap.add_argument("--min-test-pr-auc", type=float, default=0.0)
    args = ap.parse_args()

    run_tag = args.run_tag or datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S")
    steps: list[dict] = []
    started_at = datetime.now(timezone.utc).isoformat()

    train_end, val_end, test_end = args.train_end, args.val_end, args.test_end
    if not (train_end and val_end and test_end):
        train_end, val_end, test_end = _infer_cutoffs(label_days=args.label_days)
        print(f"[orchestrator] inferred cutoffs train={train_end} val={val_end} test={test_end}")

    py = sys.executable
    _run([py, "-m", "src.cli.ingest"], steps)
    _run([py, "-m", "src.etl.validate_canonical"], steps)
    _run([py, "-m", "src.etl.build_interactions"], steps)
    _run([py, "-m", "src.etl.collapse_txn_dupes", "--level", "c_p_d_s"], steps)
    _run([py, "-m", "src.etl.validate_interactions"], steps)
    _run([py, "-m", "src.cli.build_features"], steps)
    _run([py, "-m", "src.cli.data_quality_gates", "--stage", "features", "--strict"], steps)
    _run(
        [
            py,
            "-m",
            "src.etl.label_sampling",
            "--train-end",
            train_end,
            "--val-end",
            val_end,
            "--test-end",
            test_end,
            "--label-days",
            str(args.label_days),
            "--neg-per-pos",
            str(args.neg_per_pos),
            "--hard-negatives",
            args.hard_negatives,
        ],
        steps,
    )
    _run([py, "-m", "src.models.feature_selection"], steps)

    hpo_cmd = [
        py,
        "-m",
        "src.models.hpo",
        "--metric",
        "pr_auc",
        "--top-n",
        str(args.hpo_top_n),
        "--n-trials",
        str(args.hpo_trials),
        "--fair-train",
        "--fair-group-col",
        "ClientGender",
        "--fair-groups",
        args.fairness_groups,
    ]
    if args.mlflow:
        hpo_cmd.append("--mlflow")
    _run(hpo_cmd, steps)

    reg_cmd = [
        py,
        "-m",
        "src.cli.model_registry",
        "--metric",
        "pr_auc",
        "--source",
        "orchestrate_pipeline",
        "--min-val-pr-auc",
        str(args.min_val_pr_auc),
        "--min-val-map10",
        str(args.min_val_map10),
        "--min-test-pr-auc",
        str(args.min_test_pr_auc),
    ]
    if args.promote:
        reg_cmd.append("--promote")
    _run(reg_cmd, steps)

    production_pointer = REGISTRY / "production_model.json"
    if production_pointer.exists():
        production_meta = json.loads(production_pointer.read_text())
        model_path = production_meta["model_path"]
    else:
        model_path = str(PROC / "models" / "hpo_random_forest.pkl")

    for split in ["val", "test"]:
        prefix = f"{run_tag}_{split}"
        rec_cmd = [
            py,
            "-m",
            "src.cli.recommend",
            "--split",
            split,
            "--top-k",
            str(args.top_k),
            "--group-col",
            "ClientGender",
            "--lambda-fairness",
            str(args.lambda_fairness),
            "--fairness-groups",
            args.fairness_groups,
            "--model-path",
            model_path,
            "--output-prefix",
            prefix,
        ]
        if args.mlflow:
            rec_cmd.append("--mlflow")
        _run(rec_cmd, steps)

        base_path = RECS / f"{prefix}_base.parquet"
        fair_path = RECS / f"{prefix}_fair.parquet"
        eval_cmd = [
            py,
            "-m",
            "src.cli.evaluate_recommendations",
            "--base-path",
            str(base_path),
            "--fair-path",
            str(fair_path),
            "--split",
            split,
            "--k",
            str(args.top_k),
            "--output-stem",
            f"{prefix}_compare",
        ]
        if args.mlflow:
            eval_cmd.append("--mlflow")
        _run(eval_cmd, steps)

    _run([py, "-m", "src.cli.experiment_governance", "--name", run_tag], steps)

    finished_at = datetime.now(timezone.utc).isoformat()
    out = REPORTS / f"{run_tag}_orchestration_report.json"
    out.write_text(
        json.dumps(
            {
                "run_tag": run_tag,
                "started_at_utc": started_at,
                "finished_at_utc": finished_at,
                "cutoffs": {"train_end": train_end, "val_end": val_end, "test_end": test_end},
                "steps": steps,
            },
            indent=2,
        )
    )
    print(f"[orchestrator] done -> {out}")


if __name__ == "__main__":
    main()

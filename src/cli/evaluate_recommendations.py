from __future__ import annotations

import argparse
from pathlib import Path

from src.eval.recommendation_eval import compare_base_vs_fair, save_comparison_report


def main() -> None:
    ap = argparse.ArgumentParser(description="Offline evaluator for base vs fairness-aware recommendations.")
    ap.add_argument("--base-path", required=True, help="Path to base recommendations parquet")
    ap.add_argument("--fair-path", required=True, help="Path to fairness-reranked recommendations parquet")
    ap.add_argument("--split-path", default=None, help="Path to split parquet with labels (defaults from --split)")
    ap.add_argument("--split", choices=["train", "val", "test"], default="val")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--output-dir", default="data/processed/reports/recommendations")
    ap.add_argument("--output-stem", default="comparison")
    args = ap.parse_args()

    split_path = (
        Path(args.split_path)
        if args.split_path
        else Path(f"data/processed/model/{args.split}.parquet")
    )

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

    print("[OK] offline evaluation completed")
    print(f" - json: {json_path}")
    print(f" - csv: {csv_path}")
    print(" - base metrics:", comparison["base"])
    print(" - fair metrics:", comparison["fair"])
    print(" - delta:", comparison["delta"])


if __name__ == "__main__":
    main()


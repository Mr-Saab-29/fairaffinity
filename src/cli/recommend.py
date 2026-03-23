from __future__ import annotations

import argparse
from pathlib import Path

from src.recommender.pipeline import HybridWeights, run_recommendation_pipeline


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
    args = ap.parse_args()

    weights = HybridWeights(
        affinity=args.w_affinity,
        collaborative=args.w_collab,
        content=args.w_content,
        popularity=args.w_pop,
    )
    fairness_groups = [s.strip() for s in args.fairness_groups.split(",") if s.strip()]
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
    print("[OK] recommendation artifacts written")
    print(f" - base recs: {out['base_path']}")
    print(f" - fair recs: {out['fair_path']}")
    print(f" - fairness summary: {out['summary_path']}")
    print(f" - fairness(pre): {out['summary_pre']}")
    print(f" - fairness(post): {out['summary_post']}")


if __name__ == "__main__":
    main()

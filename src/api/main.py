from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.eval.recommendation_eval import compare_base_vs_fair, save_comparison_report
from src.recommender.pipeline import HybridWeights, run_recommendation_pipeline

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
RECS_DIR = PROC / "reports" / "recommendations"
MODELS_DIR = PROC / "models"

app = FastAPI(
    title="FairAffinity API",
    description="Fairness-aware recommendation and offline evaluation service.",
    version="1.0.0",
)


class RecommendRequest(BaseModel):
    split: Literal["train", "val", "test"] = "val"
    model_path: str = "data/processed/models/hpo_random_forest.pkl"
    top_k: int = Field(default=20, ge=1, le=200)
    group_col: str = "ClientGender"
    lambda_fairness: float = 0.5
    fairness_groups: list[str] = ["Male", "Female", "Unisex"]
    w_affinity: float = 0.6
    w_collab: float = 0.2
    w_content: float = 0.15
    w_pop: float = 0.05
    output_prefix: str = "api_run"


class EvaluateRequest(BaseModel):
    base_path: str
    fair_path: str
    split: Literal["train", "val", "test"] = "val"
    k: int = Field(default=20, ge=1, le=200)
    output_stem: str = "api_eval"


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "time_utc": datetime.now(timezone.utc).isoformat(),
        "recommendations_dir": str(RECS_DIR),
    }


@app.get("/artifacts/recommendations")
def list_recommendation_artifacts(limit: int = 100) -> dict:
    if not RECS_DIR.exists():
        return {"files": []}
    files = sorted(RECS_DIR.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)[:limit]
    return {
        "files": [
            {"name": p.name, "path": str(p), "size_bytes": p.stat().st_size}
            for p in files
            if p.is_file()
        ]
    }


@app.post("/recommend/run")
def run_recommend(request: RecommendRequest) -> dict:
    model_path = Path(request.model_path)
    if not model_path.is_absolute():
        model_path = ROOT / model_path
    if not model_path.exists():
        raise HTTPException(status_code=404, detail=f"Model not found: {model_path}")

    weights = HybridWeights(
        affinity=request.w_affinity,
        collaborative=request.w_collab,
        content=request.w_content,
        popularity=request.w_pop,
    )
    out = run_recommendation_pipeline(
        split_name=request.split,
        model_path=model_path,
        top_k=request.top_k,
        group_col=request.group_col,
        lambda_fairness=request.lambda_fairness,
        fairness_groups=request.fairness_groups,
        weights=weights,
        output_prefix=request.output_prefix,
    )
    return {
        "status": "ok",
        "split": request.split,
        "output": out,
    }


@app.post("/evaluate/run")
def run_evaluate(request: EvaluateRequest) -> dict:
    base_path = Path(request.base_path)
    fair_path = Path(request.fair_path)
    if not base_path.is_absolute():
        base_path = ROOT / base_path
    if not fair_path.is_absolute():
        fair_path = ROOT / fair_path
    if not base_path.exists():
        raise HTTPException(status_code=404, detail=f"Base recommendations not found: {base_path}")
    if not fair_path.exists():
        raise HTTPException(status_code=404, detail=f"Fair recommendations not found: {fair_path}")

    split_path = PROC / "model" / f"{request.split}.parquet"
    if not split_path.exists():
        raise HTTPException(status_code=404, detail=f"Split not found: {split_path}")

    comparison = compare_base_vs_fair(
        base_path=base_path,
        fair_path=fair_path,
        split_path=split_path,
        k=request.k,
    )
    json_path, csv_path = save_comparison_report(
        comparison=comparison,
        output_dir=RECS_DIR,
        stem=request.output_stem,
    )
    return {
        "status": "ok",
        "comparison": comparison,
        "json_path": str(json_path),
        "csv_path": str(csv_path),
    }


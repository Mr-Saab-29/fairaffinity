from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
REPORTS = PROC / "reports"
MODELS = PROC / "models"
REGISTRY_DIR = PROC / "model_registry"
REGISTRY_MODELS = REGISTRY_DIR / "models"
REGISTRY_JSON = REGISTRY_DIR / "registry.json"
PRODUCTION_POINTER = REGISTRY_DIR / "production_model.json"


@dataclass
class Thresholds:
    min_val_pr_auc: float = 0.0
    min_val_map10: float = 0.0
    min_test_pr_auc: float = 0.0


def _load_registry() -> dict:
    if not REGISTRY_JSON.exists():
        return {"versions": []}
    return json.loads(REGISTRY_JSON.read_text())


def _save_registry(payload: dict) -> None:
    REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    REGISTRY_MODELS.mkdir(parents=True, exist_ok=True)
    REGISTRY_JSON.write_text(json.dumps(payload, indent=2))


def _pick_best_from_hpo(metric: str) -> tuple[str, dict]:
    summary_path = REPORTS / "hpo_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing {summary_path}")
    df = pd.read_csv(summary_path)
    val_col = f"val_{metric}"
    if val_col not in df.columns:
        raise ValueError(f"{summary_path} missing column {val_col}")
    best = df.sort_values(val_col, ascending=False).iloc[0].to_dict()
    model_name = str(best["model"])
    model_path = MODELS / f"hpo_{model_name}.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model artifact: {model_path}")
    return str(model_path), best


def _passes_thresholds(metrics: dict, thresholds: Thresholds) -> tuple[bool, list[str]]:
    issues: list[str] = []
    if float(metrics.get("val_pr_auc", 0.0)) < thresholds.min_val_pr_auc:
        issues.append(f"val_pr_auc < {thresholds.min_val_pr_auc}")
    if float(metrics.get("val_map@10", 0.0)) < thresholds.min_val_map10:
        issues.append(f"val_map@10 < {thresholds.min_val_map10}")
    if float(metrics.get("test_pr_auc", 0.0)) < thresholds.min_test_pr_auc:
        issues.append(f"test_pr_auc < {thresholds.min_test_pr_auc}")
    return len(issues) == 0, issues


def _next_version(registry: dict) -> int:
    versions = registry.get("versions", [])
    if not versions:
        return 1
    return max(int(v["version"]) for v in versions) + 1


def main() -> None:
    ap = argparse.ArgumentParser(description="Register and promote trained models.")
    ap.add_argument("--metric", default="pr_auc", choices=["pr_auc", "roc_auc", "map@10", "recall@10"])
    ap.add_argument("--model-path", default=None, help="Optional manual model path. If omitted, picks best from hpo_summary.csv.")
    ap.add_argument("--source", default="hpo", help="Registry source label.")
    ap.add_argument("--promote", action="store_true", help="Promote this model to production if thresholds pass.")
    ap.add_argument("--min-val-pr-auc", type=float, default=0.0)
    ap.add_argument("--min-val-map10", type=float, default=0.0)
    ap.add_argument("--min-test-pr-auc", type=float, default=0.0)
    args = ap.parse_args()

    if args.model_path:
        selected_model_path = Path(args.model_path)
        if not selected_model_path.is_absolute():
            selected_model_path = ROOT / selected_model_path
        if not selected_model_path.exists():
            raise FileNotFoundError(f"Missing model path: {selected_model_path}")
        best_row = {"model": selected_model_path.stem}
    else:
        selected_model_path_str, best_row = _pick_best_from_hpo(metric=args.metric)
        selected_model_path = Path(selected_model_path_str)

    registry = _load_registry()
    version = _next_version(registry)
    REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    REGISTRY_MODELS.mkdir(parents=True, exist_ok=True)

    model_dst = REGISTRY_MODELS / f"model_v{version}.pkl"
    shutil.copy2(selected_model_path, model_dst)

    entry = {
        "version": version,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": args.source,
        "model_name": best_row.get("model"),
        "origin_model_path": str(selected_model_path),
        "registry_model_path": str(model_dst),
        "metrics": best_row,
        "status": "staging",
    }
    registry.setdefault("versions", []).append(entry)
    _save_registry(registry)

    thresholds = Thresholds(
        min_val_pr_auc=args.min_val_pr_auc,
        min_val_map10=args.min_val_map10,
        min_test_pr_auc=args.min_test_pr_auc,
    )
    pass_thresholds, issues = _passes_thresholds(best_row, thresholds)
    decision = {
        "version": version,
        "promote_requested": args.promote,
        "thresholds": thresholds.__dict__,
        "pass_thresholds": pass_thresholds,
        "issues": issues,
    }

    if args.promote and pass_thresholds:
        entry["status"] = "production"
        for other in registry.get("versions", []):
            if other["version"] != version and other.get("status") == "production":
                other["status"] = "archived"
        _save_registry(registry)
        production_model = MODELS / "production.pkl"
        shutil.copy2(model_dst, production_model)
        PRODUCTION_POINTER.write_text(
            json.dumps(
                {
                    "version": version,
                    "model_path": str(production_model),
                    "registry_model_path": str(model_dst),
                    "promoted_at_utc": datetime.now(timezone.utc).isoformat(),
                    "metrics": best_row,
                },
                indent=2,
            )
        )
        decision["promoted"] = True
    else:
        decision["promoted"] = False

    decision_path = REGISTRY_DIR / "last_promotion_decision.json"
    decision_path.write_text(json.dumps(decision, indent=2))

    print(f"[registry] registered version={version} -> {model_dst}")
    print(f"[registry] decision written -> {decision_path}")
    if decision["promoted"]:
        print(f"[registry] promoted to production -> {MODELS / 'production.pkl'}")
    else:
        print(f"[registry] not promoted | issues={issues}")


if __name__ == "__main__":
    main()

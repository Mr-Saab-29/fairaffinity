from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
REPORTS = PROC / "reports"
RECS = REPORTS / "recommendations"
REGISTRY = PROC / "model_registry"
OUTDIR = REPORTS / "governance"
OUTDIR.mkdir(parents=True, exist_ok=True)


def _safe_read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _latest_file(pattern: str) -> Path | None:
    files = sorted(RECS.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def main() -> None:
    ap = argparse.ArgumentParser(description="Build experiment governance artifacts (JSON + markdown).")
    ap.add_argument("--name", default="latest")
    args = ap.parse_args()

    hpo_summary = REPORTS / "hpo_summary.csv"
    label_summary = REPORTS / "label_sampling_summary.csv"
    production = REGISTRY / "production_model.json"
    decision = REGISTRY / "last_promotion_decision.json"
    fairness_val = _latest_file("*val*fairness_summary.json")
    fairness_test = _latest_file("*test*fairness_summary.json")
    eval_val = _latest_file("*val*offline_eval.json")
    eval_test = _latest_file("*test*offline_eval.json")

    best_hpo = None
    if hpo_summary.exists():
        hpo_df = pd.read_csv(hpo_summary)
        if "val_pr_auc" in hpo_df.columns and not hpo_df.empty:
            best_hpo = hpo_df.sort_values("val_pr_auc", ascending=False).iloc[0].to_dict()

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifacts": {
            "hpo_summary": str(hpo_summary) if hpo_summary.exists() else None,
            "label_sampling_summary": str(label_summary) if label_summary.exists() else None,
            "production_model": str(production) if production.exists() else None,
            "promotion_decision": str(decision) if decision.exists() else None,
            "fairness_val": str(fairness_val) if fairness_val else None,
            "fairness_test": str(fairness_test) if fairness_test else None,
            "offline_eval_val": str(eval_val) if eval_val else None,
            "offline_eval_test": str(eval_test) if eval_test else None,
        },
        "best_hpo_model": best_hpo,
        "production_model": _safe_read_json(production),
        "promotion_decision": _safe_read_json(decision),
        "fairness_val": _safe_read_json(fairness_val) if fairness_val else None,
        "fairness_test": _safe_read_json(fairness_test) if fairness_test else None,
        "offline_eval_val": _safe_read_json(eval_val) if eval_val else None,
        "offline_eval_test": _safe_read_json(eval_test) if eval_test else None,
    }

    out_json = OUTDIR / f"{args.name}_governance.json"
    out_md = OUTDIR / f"{args.name}_model_card.md"
    out_json.write_text(json.dumps(payload, indent=2))

    lines = [
        "# FairAffinity Experiment Governance",
        "",
        f"- Generated (UTC): {payload['generated_at_utc']}",
        f"- Best HPO model: {best_hpo.get('model') if best_hpo else 'N/A'}",
        f"- Production version: {(payload['production_model'] or {}).get('version', 'N/A')}",
        "",
        "## Validation Quality",
        f"- PR-AUC (val): {best_hpo.get('val_pr_auc') if best_hpo else 'N/A'}",
        f"- MAP@10 (val): {best_hpo.get('val_map@10') if best_hpo else 'N/A'}",
        "",
        "## Fairness Summary",
        f"- Validation fairness file: {payload['artifacts']['fairness_val']}",
        f"- Test fairness file: {payload['artifacts']['fairness_test']}",
        "",
        "## Offline Evaluation",
        f"- Validation offline eval: {payload['artifacts']['offline_eval_val']}",
        f"- Test offline eval: {payload['artifacts']['offline_eval_test']}",
        "",
    ]
    out_md.write_text("\n".join(lines))

    print(f"[governance] wrote {out_json}")
    print(f"[governance] wrote {out_md}")


if __name__ == "__main__":
    main()

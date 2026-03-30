from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
REGISTRY = PROC / "model_registry"
REPORTS = PROC / "reports"
MONITORING = PROC / "monitoring"


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _latest_eval() -> dict | None:
    recs = REPORTS / "recommendations"
    files = sorted(recs.glob("*test*offline_eval.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        return None
    return _load_json(files[0])


def main() -> None:
    ap = argparse.ArgumentParser(description="Retraining policy decision + optional execution.")
    ap.add_argument("--max-model-age-days", type=int, default=14)
    ap.add_argument("--min-test-precision-at-k", type=float, default=0.10)
    ap.add_argument("--max-api-p95-latency-ms", type=float, default=1500.0)
    ap.add_argument("--execute", action="store_true", help="Run retraining command if policy triggers.")
    ap.add_argument(
        "--run-command",
        default="python -m src.cli.orchestrate_pipeline --promote",
        help="Command executed when retraining is required and --execute is enabled.",
    )
    args = ap.parse_args()

    production = _load_json(REGISTRY / "production_model.json")
    monitoring = _load_json(MONITORING / "monitoring_summary.json")
    eval_test = _latest_eval()

    reasons: list[str] = []
    now = datetime.now(timezone.utc)

    if production:
        promoted_at = datetime.fromisoformat(production["promoted_at_utc"].replace("Z", "+00:00"))
        if now - promoted_at > timedelta(days=args.max_model_age_days):
            reasons.append("model_age_exceeded")
    else:
        reasons.append("no_production_model")

    if eval_test:
        fair = eval_test.get("fair", {})
        pr_key = next((k for k in fair.keys() if k.startswith("precision@")), None)
        # Precision@k is available in eval output; use it as a health proxy.
        if pr_key is not None and float(fair.get(pr_key, 0.0)) < args.min_test_precision_at_k:
            reasons.append(f"low_{pr_key}")

    if monitoring:
        p95 = ((monitoring.get("api") or {}).get("p95_latency_ms")) or 0.0
        if float(p95) > args.max_api_p95_latency_ms:
            reasons.append("high_api_latency")

    should_retrain = len(reasons) > 0
    decision = {
        "timestamp_utc": now.isoformat(),
        "should_retrain": should_retrain,
        "reasons": reasons,
        "inputs": {
            "production_model_present": production is not None,
            "monitoring_present": monitoring is not None,
            "eval_test_present": eval_test is not None,
        },
    }
    out = REGISTRY / "retrain_policy_decision.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(decision, indent=2))
    print(f"[retrain-policy] wrote {out}")
    print(f"[retrain-policy] should_retrain={should_retrain} reasons={reasons}")

    if args.execute and should_retrain:
        print(f"[retrain-policy] executing: {args.run_command}")
        proc = subprocess.run(args.run_command, shell=True, cwd=str(ROOT))
        if proc.returncode != 0:
            raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()

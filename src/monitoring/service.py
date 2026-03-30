from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
REPORTS = PROC / "reports"
RECS = REPORTS / "recommendations"
MON_DIR = PROC / "monitoring"
MON_DIR.mkdir(parents=True, exist_ok=True)
LOG_CSV = MON_DIR / "inference_requests.csv"
SUMMARY_JSON = MON_DIR / "monitoring_summary.json"


def log_api_request(path: str, method: str, status_code: int, latency_ms: float) -> None:
    row = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "path": path,
        "method": method,
        "status_code": int(status_code),
        "latency_ms": float(latency_ms),
    }
    df = pd.DataFrame([row])
    if LOG_CSV.exists():
        df.to_csv(LOG_CSV, mode="a", header=False, index=False)
    else:
        df.to_csv(LOG_CSV, index=False)


def monitoring_summary(window_hours: int = 24) -> dict:
    out = {
        "window_hours": window_hours,
        "events": 0,
        "p50_latency_ms": None,
        "p95_latency_ms": None,
        "error_rate": None,
        "paths": {},
    }
    if not LOG_CSV.exists():
        return out
    df = pd.read_csv(LOG_CSV)
    if df.empty:
        return out
    ts = pd.to_datetime(df["timestamp_utc"], errors="coerce", utc=True)
    cutoff = datetime.now(timezone.utc) - timedelta(hours=window_hours)
    win = df[ts >= cutoff].copy()
    if win.empty:
        return out

    out["events"] = int(len(win))
    out["p50_latency_ms"] = float(win["latency_ms"].quantile(0.50))
    out["p95_latency_ms"] = float(win["latency_ms"].quantile(0.95))
    out["error_rate"] = float((win["status_code"] >= 400).mean())
    out["paths"] = (
        win.groupby("path", as_index=False)
        .agg(
            requests=("path", "size"),
            p95_latency_ms=("latency_ms", lambda s: float(s.quantile(0.95))),
            error_rate=("status_code", lambda s: float((s >= 400).mean())),
        )
        .sort_values("requests", ascending=False)
        .to_dict(orient="records")
    )
    return out


def recommendation_drift_summary(limit: int = 20) -> dict:
    files = sorted(RECS.glob("*_fairness_summary.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:limit]
    rows = []
    for p in files:
        try:
            data = json.loads(p.read_text())
            pre = data.get("focus_pre") or data.get("pre") or {}
            post = data.get("focus_post") or data.get("post") or {}
            rows.append(
                {
                    "file": p.name,
                    "timestamp_utc": datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc).isoformat(),
                    "pre_max_gap": pre.get("max_gap"),
                    "post_max_gap": post.get("max_gap"),
                    "pre_min_max_ratio": pre.get("min_max_ratio"),
                    "post_min_max_ratio": post.get("min_max_ratio"),
                }
            )
        except Exception:
            continue
    return {"records": rows}


def build_and_save_monitoring_snapshot(window_hours: int = 24) -> Path:
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "api": monitoring_summary(window_hours=window_hours),
        "recommendation_drift": recommendation_drift_summary(),
    }
    SUMMARY_JSON.write_text(json.dumps(payload, indent=2))
    return SUMMARY_JSON

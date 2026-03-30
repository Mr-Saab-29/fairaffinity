from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
REPORTS = PROC / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)


@dataclass
class GateResult:
    name: str
    passed: bool
    message: str

    def to_dict(self) -> dict:
        return {"name": self.name, "passed": self.passed, "message": self.message}


def _exists_any(paths: list[Path], name: str) -> GateResult:
    for p in paths:
        if p.exists():
            return GateResult(name=name, passed=True, message=f"found: {p}")
    return GateResult(name=name, passed=False, message=f"missing all: {[str(p) for p in paths]}")


def _check_interactions() -> list[GateResult]:
    p_collapsed = PROC / "interactions_collapsed_c_p_d_s.parquet"
    p_raw = PROC / "interactions.parquet"
    gate = _exists_any([p_collapsed, p_raw], "interactions_exists")
    if not gate.passed:
        return [gate]
    path = p_collapsed if p_collapsed.exists() else p_raw
    df = pd.read_parquet(path)
    out = [gate]
    out.append(GateResult("interactions_non_empty", len(df) > 0, f"rows={len(df)}"))
    for col in ["ClientID", "ProductID"]:
        if col in df.columns:
            null_ratio = float(df[col].isna().mean())
            out.append(
                GateResult(
                    f"{col}_null_ratio_lt_1pct",
                    null_ratio < 0.01,
                    f"{col}_null_ratio={null_ratio:.4f}",
                )
            )
    if "txn_date" in df.columns:
        valid_ratio = float(pd.to_datetime(df["txn_date"], errors="coerce").notna().mean())
        out.append(
            GateResult(
                "txn_date_valid_ratio_gt_95pct",
                valid_ratio > 0.95,
                f"txn_date_valid_ratio={valid_ratio:.4f}",
            )
        )
    return out


def _check_features() -> list[GateResult]:
    required = [
        PROC / "user_features.parquet",
        PROC / "product_features.parquet",
        PROC / "user_rfm.parquet",
        PROC / "client_product_recency.parquet",
        PROC / "category_features.parquet",
    ]
    out: list[GateResult] = []
    for p in required:
        out.append(GateResult(f"{p.name}_exists", p.exists(), str(p)))
    return out


def _check_model_splits() -> list[GateResult]:
    out: list[GateResult] = []
    for split in ["train", "val", "test"]:
        p = PROC / "model" / f"{split}.parquet"
        if not p.exists():
            out.append(GateResult(f"{split}_split_exists", False, str(p)))
            continue
        df = pd.read_parquet(p, columns=["ClientID", "ProductID", "label"])
        out.append(GateResult(f"{split}_split_exists", True, str(p)))
        out.append(GateResult(f"{split}_rows_gt_0", len(df) > 0, f"rows={len(df)}"))
        label_ok = set(df["label"].dropna().astype(int).unique()).issubset({0, 1})
        out.append(GateResult(f"{split}_label_binary", label_ok, "label must be {0,1}"))
    return out


def run_gates(stage: str) -> tuple[list[GateResult], bool]:
    checks: list[GateResult] = []
    checks.extend(_check_interactions())
    if stage in {"features", "training", "full"}:
        checks.extend(_check_features())
    if stage in {"training", "full"}:
        checks.extend(_check_model_splits())
    passed = all(c.passed for c in checks)
    return checks, passed


def main() -> None:
    ap = argparse.ArgumentParser(description="Fail-fast data quality gates for FairAffinity.")
    ap.add_argument(
        "--stage",
        default="full",
        choices=["ci", "features", "training", "full"],
        help="Gate strictness level.",
    )
    ap.add_argument("--strict", action="store_true", help="Exit non-zero when any gate fails.")
    args = ap.parse_args()

    stage = "full" if args.stage == "ci" else args.stage
    checks, passed = run_gates(stage=stage)
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "stage": stage,
        "passed": passed,
        "checks": [c.to_dict() for c in checks],
    }
    out_json = REPORTS / "data_quality_gates.json"
    out_csv = REPORTS / "data_quality_gates.csv"
    out_json.write_text(json.dumps(payload, indent=2))
    pd.DataFrame([c.to_dict() for c in checks]).to_csv(out_csv, index=False)

    print(f"[quality] stage={stage} passed={passed}")
    print(f"[quality] wrote {out_json}")
    print(f"[quality] wrote {out_csv}")
    for c in checks:
        flag = "PASS" if c.passed else "FAIL"
        print(f" - [{flag}] {c.name}: {c.message}")

    if args.strict and not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

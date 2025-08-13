from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
OUTDIR = PROC / "reports"
OUTDIR.mkdir(parents=True, exist_ok=True)

def summarize_nulls(df: pd.DataFrame) -> Dict[str, float]:
    """Calculate percentage of nulls in each column."""
    return (df.isna().mean() * 100).round(2).to_dict()

def describe_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Get descriptive statistics for numeric columns."""
    present = [c for c in df.columns if c in df.columns]
    return df[present].describe(percentiles = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).T.reset_index(names = "column")

def uniqueness_report(df: pd.DataFrame) -> Dict[str, Any]:
    """Validate the uniqueness of key columns in the DataFrame.
    """
    rep: Dict[str, Any] = {"rows": int(len(df))}
    key1 = ["ClientID", "ProductID", "txn_date", "StoreID"]
    key2 = ["ClientID", "ProductID", "txn_date"]
    for key in (key1, key2):
        existing = [c for c in key if c in df.columns]
        if len(existing) == len(key):
            dup = int(df.duplicated(subset=existing, keep=False).sum())
            rep["|".join(key) + "_dup_rows"] = dup
    return rep

def availability_report(df: pd.DataFrame) -> Dict[str, Any]:
    """Check availability flags in the DataFrame.
    """
    out: Dict[str, Any] = {}
    for c in ["AvailableInStoreCountry", "AvailableInClientCountry"]:
        if c in df.columns:
            out[c + "_share"] = float(df[c].mean())
            vc = df[c].value_counts(dropna=False).to_dict()
            out[c + "_counts"] = {str(k): int(v) for k, v in vc.items()}
    return out

def build_report(df: pd.DataFrame) -> Dict[str, Any]:
    """ Generate a validation report for the given DataFrame."""
    report: Dict[str, Any] = {}

    # shapes and time range
    report["shape"] = {"rows": int(len(df)), "cols": int(df.shape[1])}
    if "txn_date" in df.columns:
        report["date_range"] = {
            "min": str(df["txn_date"].min()),
            "max": str(df["txn_date"].max()),
        }

    # uniqueness
    report["uniqueness"] = uniqueness_report(df)

    # nulls
    report["null_percent"] = summarize_nulls(df)

    # numerics summary
    num_cols = ["Quantity", "SalesNetAmountEuro"]
    report["numeric_summary"] = describe_numeric(df, num_cols).to_dict(orient="records")

    # availability flags
    report["availability"] = availability_report(df)

    # key coverage sanity
    for c in ["ClientID", "ProductID", "StoreID"]:
        if c in df.columns:
            report[f"{c}_nunique"] = int(df[c].nunique())

    # basic validity checks (warning-level)
    report["warnings"] = []
    if "Quantity" in df.columns and (df["Quantity"] < 0).any():
        report["warnings"].append("Negative quantities detected.")
    if "SalesNetAmountEuro" in df.columns and (df["SalesNetAmountEuro"] < 0).any():
        report["warnings"].append("Negative net amounts detected.")
    if "txn_date" in df.columns and df["txn_date"].isna().mean() > 0.05:
        report["warnings"].append("More than 5% missing txn_date.")

    return report

def save_reports(report: Dict[str, Any], tag: str = "interactions") -> None:
    """Save the validation report to a JSON file."""
    out_path = OUTDIR / f"{tag}_validation_report.json"
    csv_rows = []

    shape = report.get("shape", {})
    uniq = report.get("uniqueness", {})
    avail = report.get("availability", {})

    csv_rows.append(
        {
            "rows" : shape.get("rows", 0),
            "cols" : shape.get("cols", 0),
            **{k: v for k, v in uniq.items() if not isinstance(v, dict)},
            **{k: v for k, v in avail.items() if not isinstance(v, dict)},
        }
        )
    pd.DataFrame(csv_rows).to_csv(OUTDIR / f"{tag}_validation_summary.csv", index=False)

    # Full Json
    out_path.write_text(json.dumps(report, indent=2))
    print(f"Validation report saved to {out_path}")

def validate_interactions(sample_frac: float | None = None) -> Dict[str, Any]:
    """ Validate the interactions data by reading the parquet file and generating a report."""
    path = PROC / "interactions.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")

    df = pd.read_parquet(path)

    # ensure datetime
    if "txn_date" in df.columns:
        df["txn_date"] = pd.to_datetime(df["txn_date"], errors="coerce")

    # optional sampling for speed
    if sample_frac is not None and 0 < sample_frac < 1:
        df = df.sample(frac=sample_frac, random_state=42).reset_index(drop=True)
        print(f"[info] sampled interactions at frac={sample_frac} -> rows={len(df):,}")

    report = build_report(df)
    save_reports(report, tag="interactions")

    # console highlights
    uniq = report["uniqueness"]
    print("\n=== Interactions Validation (highlights) ===")
    print(f"rows={report['shape']['rows']:,} | cols={report['shape']['cols']}")
    if "date_range" in report:
        print(f"date_range: {report['date_range']['min']} → {report['date_range']['max']}")
    for k, v in uniq.items():
        if k.endswith("_dup_rows"):
            print(f"{k}: {v:,}")
    for c in ["AvailableInStoreCountry_share", "AvailableInClientCountry_share"]:
        if c in report.get("availability", {}):
            print(f"{c}: {report['availability'][c]:.3f}")
    if report["warnings"]:
        print("warnings:", "; ".join(report["warnings"]))
    else:
        print("warnings: none")

    return report

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sample-frac", type=float, default=None,
        help="Optional sampling fraction for validation (0 < frac < 1)."
    )
    args = parser.parse_args()
    validate_interactions(sample_frac=args.sample_frac)


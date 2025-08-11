#src/cli/ingest.py

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, Any
import pandas as pd
import yaml


#----------- Paths ---------------------

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
CONTRACTS = ROOT/"configs"/"contracts"

#dataset name -> (csv_filename, yaml contract file name)
DATASETS = {
    "clients" : ("clients.csv", "clients.yaml"),
    "products": ("products.csv", "products.yaml"),
    "stores": ("stores.csv", "stores.yaml"),
    "transactions": ("transactions.csv", "transactions.yaml"),
    "stocks"    : ("stocks.csv", "stocks.yaml"),
}

#pandas helper functions
def coerce_column(series: pd.Series, typ: str) -> pd.Series:
    if typ == "datetime64[ns]":
        return pd.to_datetime(series, errors="coerce")
    if typ == "bool":
        # coerce common encodings (1/0, yes/no, true/false)
        s = series.copy()
        if s.dtype == "O":
            m = {
                "yes": True, "no": False,
                "true": True, "false": False,
                "1": True, "0": False,
                1: True, 0: False,
                "Y": True, "N": False,
            }
            s = s.map(lambda x: m.get(x, x))
        return s.astype("boolean")
    if typ == "string":
        return series.astype("string").str.strip()
    if typ == "int64":
        # allow missing -> nullable Int64
        return pd.to_numeric(series, errors="coerce").astype("Int64")
    if typ == "float64":
        return pd.to_numeric(series, errors="coerce")
    # default: return as-is
    return series

def load_contract(contract_path: Path) -> Dict[str, Any]:
    return yaml.safe_load(contract_path.read_text())

def validate_against_contract(df: pd.DataFrame, contract: Dict[str, Any]) -> list[str]:
    errors: list[str] = []

    # required columns present?
    for col, spec in contract.get("columns", {}).items():
        if spec.get("required", False) and col not in df.columns:
            errors.append(f"Missing required column: {col}")

    # primary key uniqueness?
    pk = contract.get("primary_key")
    if pk:
        subset = pk if isinstance(pk, list) else [pk]
        dups = df.duplicated(subset=subset, keep=False).sum()
        if dups:
            errors.append(f"Primary key not unique for {subset}: {dups} duplicate rows")

    return errors

def normalize_values(df: pd.DataFrame, contract: Dict[str, Any]) -> pd.DataFrame:
    # e.g., gender mapping
    for field, cfg in contract.get("value_normalization", {}).items():
        if field in df.columns and "map" in cfg:
            df[field] = df[field].map(cfg["map"]).fillna(df[field])
    return df

def _valid_product_ids():
    """
    Return a set of valid ProductID from the canonical 'products' table.
    Prefer the already-written interim parquet; if missing, fall back to products.csv.
    """
    prod_parquet = INTERIM / "products.parquet"
    if prod_parquet.exists():
        try:
            s = pd.read_parquet(prod_parquet)["ProductID"].dropna().astype("int64")
            return set(s.unique())
        except Exception:
            pass

    # Fallback: read raw csv with contract if parquet not ready
    prod_csv = RAW / "products.csv"
    prod_yml = CONTRACTS / "products.yaml"
    if prod_csv.exists() and prod_yml.exists():
        from pandas import read_csv
        prodf = read_csv(prod_csv, usecols=["ProductID"])
        return set(pd.to_numeric(prodf["ProductID"], errors="coerce").dropna().astype("int64").unique())

    # Last resort: empty set (no filtering)
    return None

def ingest_one(name: str, csv_name: str, yml_name: str, persist: bool) -> None:
    csv_path = RAW / csv_name
    yml_path = CONTRACTS / yml_name

    if not csv_path.exists():
        print(f"[WARN] {name}: missing {csv_path}")
        return
    if not yml_path.exists():
        print(f"[ERROR] {name}: missing contract {yml_path}")
        return

    # read CSV fast; let coercion happen per-contract below
    df = pd.read_csv(csv_path)

    # strict column presence check (no renaming here; your headers are clean)
    contract = load_contract(yml_path)

    # coerce dtypes per contract (only for columns that exist)
    for col, spec in contract.get("columns", {}).items():
        if col in df.columns:
            df[col] = coerce_column(df[col], spec.get("type", "string"))

    # normalize values (e.g., gender)
    df = normalize_values(df, contract)

    # validate
    errs = validate_against_contract(df, contract)
    if errs:
        print(f"[ERROR] {name}:\n - " + "\n - ".join(errs))
        return
    
    # Filter stocks to Valid ProductID present in products
    if name == "stocks":
        valid_pids = _valid_product_ids()
        if valid_pids is not None:
            before = len(df)
            # coerce ProductID just in case
            df["ProductID"] = pd.to_numeric(df["ProductID"], errors="coerce").astype("Int64")
            df = df[df["ProductID"].isin(valid_pids)].copy()
            removed = before - len(df)
            print(f"[clean] stocks: filtered {removed:,} rows with ProductID not in products")

    # success
    print(f"[OK] {name}: rows={len(df):,}, cols={len(df.columns)}")

    if persist:
        INTERIM.mkdir(parents=True, exist_ok=True)
        out_path = INTERIM / f"{name}.parquet"
        df.to_parquet(out_path, index=False)
        print(f"      → wrote {out_path}")

def main(dry_run: bool = False) -> None:
    for name, (csv_name, yml_name) in DATASETS.items():
        ingest_one(name, csv_name, yml_name, persist=not dry_run)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="Validate only; do not write parquet")
    args = ap.parse_args()
    main(dry_run=args.dry_run)
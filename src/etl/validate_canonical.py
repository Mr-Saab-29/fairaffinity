from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"
OUTDIR_CLEAN = PROCESSED / "clean"
PROCESSED.mkdir(parents=True, exist_ok=True)
OUTDIR_CLEAN.mkdir(parents=True, exist_ok=True)


TABLES = {
    "clients": {
        "path": INTERIM / "clients.parquet",
        "pk": ["ClientID"],
    },
    "products": {
        "path": INTERIM / "products.parquet",
        "pk": ["ProductID"],
    },
    "stores": {
        "path": INTERIM / "stores.parquet",
        "pk": ["StoreID"],
    },
    "transactions": {
        "path": INTERIM / "transactions.parquet",
        "pk": None,  # multiple per client/product/date
    },
    "stocks": {
        "path": INTERIM / "stocks.parquet",
        "pk": ["StoreCountry", "ProductID"],  # not strictly unique, but useful to check
    },
}

def null_summary(df: pd.DataFrame) -> Dict[str, float]:
    """Calculate percentage of nulls in each column."""
    return (df.isna().mean() * 100).round(2).to_dict()

def load_tables() -> Dict[str, pd.DataFrame]:
    """Load all tables into a dictionary."""
    dfs = {}
    for name, info in TABLES.items():
        p = info["path"]
        if not p.exists():
            raise FileNotFoundError(f"Missing table file: {p}")
        dfs[name] = pd.read_parquet(p)
    return dfs

def dedupe_table(name: str, df: pd.DataFrame, pk: List[str] | None) -> Tuple[pd.DataFrame, int]:
    """
    Table-aware deduplication:
      - transactions: drop only exact duplicates across ALL columns.
      - clients/products/stores: drop duplicates on PK, keep first.
      - stocks: aggregate to one row per (StoreCountry, ProductID) summing Quantity.
    """
    before = len(df)

    if name == "transactions":
        clean = df.drop_duplicates()
        removed = before - len(clean)
        return clean, removed

    if name in {"clients", "products", "stores"} and pk:
        clean = df.drop_duplicates(subset=pk, keep="first")
        removed = before - len(clean)
        return clean, removed

    if name == "stocks":
        qcol = "Quantity" if "Quantity" in df.columns else "stock_qty"
        tmp = df.copy()
        tmp[qcol] = pd.to_numeric(tmp[qcol], errors="coerce").fillna(0.0)
        clean = (
            tmp.groupby(["StoreCountry", "ProductID"], as_index=False)[qcol]
               .sum()
        )
        removed = before - len(clean)
        return clean, removed

    # Fallback: exact-row dedupe
    clean = df.drop_duplicates()
    removed = before - len(clean)
    return clean, removed

def pk_dupes(df: pd.DataFrame, pk: List[str]) -> int:
    if not pk:
        return 0
    return int(df.duplicated(subset=pk, keep=False).sum())

def key_coverage(
    transactions: pd.DataFrame,
    clients: pd.DataFrame,
    products: pd.DataFrame,
    stores: pd.DataFrame,
) -> Dict[str, int]:
    cov = {
        "orphan_clients":  int((~transactions["ClientID"].isin(clients["ClientID"])).sum()),
        "orphan_products": int((~transactions["ProductID"].isin(products["ProductID"])).sum()),
        "orphan_stores":   int((~transactions["StoreID"].isin(stores["StoreID"])).sum()),
    }
    return cov

def stocks_coverage(stocks: pd.DataFrame, products: pd.DataFrame) -> int:
    return int((~stocks["ProductID"].isin(products["ProductID"])).sum())

def suspicious_txn_dupes(transactions: pd.DataFrame) -> int:
    """
    Report-only: count rows in transactions that are duplicated on the composite key
    (ClientID, ProductID, SaleTransactionDate, StoreID). We DO NOT drop these here.
    """
    key_cols = ["ClientID", "ProductID", "SaleTransactionDate", "StoreID"]
    existing = [c for c in key_cols if c in transactions.columns]
    if len(existing) < 4:
        return 0
    return int(transactions.duplicated(subset=existing, keep=False).sum())

def main(fix: bool = False, drop_orphans: bool = False) -> None:
    dfs = load_tables()

    report: Dict[str, dict] = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "tables": {},
        "key_consistency": {},
    }

    # -------- per-table checks + optional dedupe ----------
    for name, meta in TABLES.items():
        df = dfs[name].copy()
        info: Dict[str, object] = {}

        info["rows_before"] = int(len(df))
        info["cols"] = int(df.shape[1])
        info["duplicate_rows_full"] = int(df.duplicated().sum())

        pk = meta["pk"]
        info["duplicate_rows_on_pk"] = pk_dupes(df, pk) if pk else 0

        # table-aware dedupe if --fix
        removed = 0
        if fix:
            df_clean, removed = dedupe_table(name, df, pk)
            dfs[name] = df_clean
        info["rows_removed_dedup"] = int(removed)
        info["rows_after"] = int(len(dfs[name]))

        # null % (top 15)
        nulls = null_summary(dfs[name])
        info["null_percent_top"] = dict(sorted(nulls.items(), key=lambda x: -x[1])[:15])

        report["tables"][name] = info

    # -------- cross-table key consistency ----------
    kc = key_coverage(dfs["transactions"], dfs["clients"], dfs["products"], dfs["stores"])
    report["key_consistency"] = kc
    report["stocks_products_not_in_products"] = stocks_coverage(dfs["stocks"], dfs["products"])

    # Report-only suspicious composite-duplicates in transactions
    report["transactions_suspicious_dupes"] = suspicious_txn_dupes(dfs["transactions"])

    # Optionally drop orphan transactions (AFTER reporting)
    if fix and drop_orphans:
        t = dfs["transactions"]
        before = len(t)
        t = t[
            t["ClientID"].isin(dfs["clients"]["ClientID"])
            & t["ProductID"].isin(dfs["products"]["ProductID"])
            & t["StoreID"].isin(dfs["stores"]["StoreID"])
        ].copy()
        removed = before - len(t)
        dfs["transactions"] = t
        report["transactions_orphans_dropped"] = int(removed)

    # -------- persist cleaned copies if --fix ----------
    if fix:
        for name, dfc in dfs.items():
            out = OUTDIR_CLEAN / f"{name}.parquet"
            dfc.to_parquet(out, index=False)

    # -------- persist reports ----------
    (PROCESSED / "validation_report.json").write_text(json.dumps(report, indent=2))

    table_rows = []
    for name, info in report["tables"].items():
        table_rows.append({
            "table": name,
            "rows_before": info["rows_before"],
            "rows_after": info["rows_after"],
            "duplicate_rows_full": info["duplicate_rows_full"],
            "duplicate_rows_on_pk": info["duplicate_rows_on_pk"],
        })
    pd.DataFrame(table_rows).to_csv(PROCESSED / "validation_summary.csv", index=False)

    # Console summary
    print("\n=== Validation Summary ===")
    for r in table_rows:
        print(
            f"{r['table']:13s} rows_before={r['rows_before']:,} | "
            f"rows_after={r['rows_after']:,} | dup_full={r['duplicate_rows_full']:,} | "
            f"dup_on_pk={r['duplicate_rows_on_pk']:,}"
        )
    print("\nKey consistency (transactions orphans):", report["key_consistency"])
    print("Stocks products not in products:", report["stocks_products_not_in_products"])
    print("Suspicious txn composite dupes (report-only):", report["transactions_suspicious_dupes"])
    if fix:
        print(f"\nCleaned copies written to: {OUTDIR_CLEAN}")
    print(f"Reports written to: {PROCESSED/'validation_report.json'} and {PROCESSED/'validation_summary.csv'}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fix", action="store_true",
                    help="Write de-duplicated clean copies to data/processed/canonical_clean")
    ap.add_argument("--drop-orphans", action="store_true",
                    help="When --fix, also drop orphan rows in transactions")
    args = ap.parse_args()
    main(fix=args.fix, drop_orphans=args.drop_orphans)
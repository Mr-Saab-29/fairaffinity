from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from src.utils.io_helpers import load_interactions
from src.utils.dates import normalize_txn_date

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
REPORTS = PROC/ "reports"
REPORTS.mkdir(parents=True, exist_ok=True)

def split_by_dates(train_end : str, val_end: str, test_end : str | None = None) -> dict:
    """Split interactions data into train, validation, and test sets based on date ranges.

    Args:
        train_end (str): End date for training set.
        val_end (str): End date for validation set.
        test_end (str | None): End date for test set. If None, no test set is created.
    
        Returns a dictionary of counts for logging
    """
    df = load_interactions()
    df['txn_date'] = normalize_txn_date(df['txn_date'])

    train_end_date = pd.to_datetime(train_end)
    val_end_date = pd.to_datetime(val_end)
    test_end_date = pd.to_datetime(test_end) if test_end else df['txn_date'].max()

    #Sanity Check on Dates
    if not train_end_date < val_end_date < test_end_date:
        raise ValueError("Ensure train_end < val_end < test_end")
    
    m_train = df['txn_date'] <= train_end_date
    m_val = (df['txn_date'] > train_end_date) & (df['txn_date'] <= val_end_date)
    m_test = (df['txn_date'] > val_end_date) & (df['txn_date'] <= test_end_date)

    out_train = PROC/ f"interactions_train_{train_end_date.date()}.parquet"
    out_val = PROC / f"interactions_val_{val_end_date.date()}.parquet"
    out_test = PROC / f"interactions_test_{test_end_date.date()}.parquet"

    df.loc[m_train].to_parquet(out_train, index=False)
    df.loc[m_val].to_parquet(out_val, index=False)
    df.loc[m_test].to_parquet(out_test, index=False)

    summary = {
        "range_min": str(df["txn_date"].min()),
        "range_max": str(df["txn_date"].max()),
        "train_end": str(train_end_date),
        "val_end": str(val_end_date),
        "test_end": str(test_end_date),
        "rows_total": int(len(df)),
        "rows_train": int(m_train.sum()),
        "rows_val": int(m_val.sum()),
        "rows_test": int(m_test.sum()),
        "uniq_clients_train": int(df.loc[m_train, "ClientID"].nunique()),
        "uniq_clients_val": int(df.loc[m_val, "ClientID"].nunique()),
        "uniq_clients_test": int(df.loc[m_test, "ClientID"].nunique()),
        "uniq_products_train": int(df.loc[m_train, "ProductID"].nunique()),
        "uniq_products_val": int(df.loc[m_val, "ProductID"].nunique()),
        "uniq_products_test": int(df.loc[m_test, "ProductID"].nunique()),
    }
    pd.DataFrame([summary]).to_csv(REPORTS / "time_splits_summary.csv", index=False)

    print(
        "[OK] time split written:\n"
        f" - train: {out_train} ({summary['rows_train']:,} rows)\n"
        f" - val  : {out_val}   ({summary['rows_val']:,} rows)\n"
        f" - test : {out_test}  ({summary['rows_test']:,} rows)\n"
        f"Summary -> {REPORTS / 'time_splits_summary.csv'}"
    )
    return summary

def main() -> None:
    parser = argparse.ArgumentParser(description="Split interactions data by date ranges.")
    parser.add_argument("--train_end", type=str, required=True, help="End date for training set (YYYY-MM-DD).")
    parser.add_argument("--val_end", type=str, required=True, help="End date for validation set (YYYY-MM-DD).")
    parser.add_argument("--test_end", type=str, default=None, help="End date for test set (YYYY-MM-DD). If None, uses max date in data.")

    args = parser.parse_args()
    split_by_dates(args.train_end, args.val_end, args.test_end)

if __name__ == "__main__":
    main()
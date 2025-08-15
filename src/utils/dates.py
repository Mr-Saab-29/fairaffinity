from __future__ import annotations
import pandas as pd

def normalize_txn_date(series: pd.Series) -> pd.Series:
    """Parse to UTC then drop tz so dtype is plain datetime64[ns]."""
    return pd.to_datetime(series, errors="coerce", utc=True).dt.tz_localize(None)

def apply_cutoff(df: pd.DataFrame, cutoff: str | None) -> pd.DataFrame:
    """Filter rows to strictly before cutoff if provided (YYYY-MM-DD)."""
    if cutoff:
        cutoff_ts = pd.to_datetime(cutoff, errors="coerce")
        if pd.isna(cutoff_ts):
            raise ValueError(f"Invalid cutoff date: {cutoff}")
        return df[df["txn_date"] < cutoff_ts].copy()
    return df
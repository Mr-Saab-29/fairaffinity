from __future__ import annotations
from pathlib import Path
import pandas as pd
from src.utils.dates import normalize_txn_date

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"

def load_interactions() -> pd.DataFrame:
    """Load interactions data from processed directory."""
    for name in [
        "interactions_collapsed_c_p_d_s.parquet",
        "interactions.parquet",
    ]:
        path = PROC / name
        if path.exists():
            df = pd.read_parquet(path)
            df['txn_date'] = normalize_txn_date(df['txn_date'])
            return df
    raise FileNotFoundError("No interactions data found in processed directory.")
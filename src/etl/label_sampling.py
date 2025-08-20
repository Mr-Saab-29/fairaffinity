from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Literal

import numpy as np
import pandas as pd

from src.utils.io_helpers import load_interactions
from src.utils.dates import normalize_txn_date

from src.features.user_features import build_user_features
from src.features.category_features import build_category_features
from src.features.rfm_features import build_user_rfm, build_client_product_recency
from src.features.product_features import build_product_features

# ---------------------------------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
MODEL_DIR = PROC / 'model'
REPORTS = PROC / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------------------------------
# Configure dataclass
# ---------------------------------------------------------------------------------------------------
HardNegMode = Literal['none', 'pop', 'pop in artefact']
PosTargetMode = Literal['binary', 'count_txn', 'sum_qty']

@dataclass
class SplitCfg:
    """ Configuration for one split (train, validation, test) """
    name : str
    cutoff : pd.Timestamp
    label_days : int
    negatives_per_pos : int
    hard_negatives : HardNegMode
    pos_target_mode : PosTargetMode
    random_state : int = 42

# ---------------------------------------------------------------------------------------------------
# Helper Functions : Window, Positives, Negatives and Feature Artifact IO
# ---------------------------------------------------------------------------------------------------

def _history_window(
        full: pd.DataFrame, cutoff: pd.Timestamp, label_days: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split full DataFrame into history and label windows.
    - history : txn_date <= cutoff
    - window : (cutoff, cutoff + delta], where labels/ targets are measured

    Args:
        full (pd.DataFrame): Full DataFrame with transactions.
        cutoff (pd.Timestamp): Cutoff date for splitting.
        label_days (int): Number of days after cutoff to consider for labels.
    
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Tuple containing history DataFrame and window DataFrame.
    """
    df = full.copy()
    df['txn_date'] = normalize_txn_date(df['txn_date'])
    history = df[df['txn_date'] <= cutoff]
    hi = cutoff + pd.Timedelta(days=label_days)
    window = df[(df['txn_date'] > cutoff) & (df['txn_date'] <= hi)]
    return history, window

def _positives(window: pd.DataFrame, mode: PosTargetMode) -> pd.DataFrame:
    """Build positives (one row per CLientID, ProductID) from the label window.
    - binary: label = 1
    - count_txn: targetr = # of transactions in the window
    - sum_qty: target = sum of Quantity in the window

    Args:
        window (pd.DataFrame): DataFrame containing transactions in the label window.
        mode (PosTargetMode): Mode for positive targets, can be 'binary', 'count_txn', or 'sum_qty'.

    Returns:
        pd.DataFrame: Dataframe with columns [(ClientID, ProductID), label, target]
    """
    if window.empty:
        return pd.DataFrame(columns=['ClientID', 'ProductID', 'label', 'target'])
    
    if mode == 'binary':
        pos = window[['ClientID', 'ProductID']].drop_duplicates()
        pos['label'] = 1
        pos['target'] = 1
        return pos
    
    agg = {
        'count_txn': ('txn_date', 'count'),
        'sum_qty': ('Quantity', 'sum')
    }
    pick = 'count_txn' if mode == 'count_txn' else 'sum_qty'

    g = (
        window.groupby(['ClientID', 'ProductID'], as_index=False)
        .agg(**agg)
    )
    g = g[g[pick] > 0].copy()
    g['label'] = 1
    g['target'] = g[pick]
    return g[['ClientID', 'ProductID', 'label', 'target']]
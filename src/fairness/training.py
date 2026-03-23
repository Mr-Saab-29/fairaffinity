from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.utils.io_helpers import load_interactions


def load_client_groups(group_col: str = "ClientGender") -> pd.DataFrame:
    """
    Build a client->group lookup from interactions.
    """
    hist = load_interactions()
    if "ClientID" not in hist.columns:
        raise ValueError("interactions must contain ClientID")
    if group_col not in hist.columns:
        raise ValueError(f"interactions must contain {group_col}")

    grp = hist[["ClientID", group_col]].drop_duplicates("ClientID").copy()
    if group_col == "ClientGender":
        grp[group_col] = grp[group_col].replace({"U": "Unisex"})
    grp[group_col] = grp[group_col].fillna("UNKNOWN")
    return grp


def attach_client_groups(
    split_df: pd.DataFrame,
    client_groups: pd.DataFrame,
    group_col: str,
) -> pd.DataFrame:
    if "ClientID" not in split_df.columns:
        raise ValueError("split_df must contain ClientID")
    out = split_df.merge(
        client_groups[["ClientID", group_col]],
        on="ClientID",
        how="left",
        validate="m:1",
    )
    out[group_col] = out[group_col].fillna("UNKNOWN")
    return out


def _weights_from_groups(
    groups: np.ndarray,
    eligible_groups: Iterable[str] | None = None,
    others_weight: float = 1.0,
    clip_min: float = 0.2,
    clip_max: float = 5.0,
) -> np.ndarray:
    """
    Inverse-frequency weighting inside eligible groups.
    """
    g = pd.Series(groups).fillna("UNKNOWN").astype(str)
    out = np.full(len(g), float(others_weight), dtype=float)

    if eligible_groups is None:
        elig = set(g.unique().tolist())
    else:
        elig = set(eligible_groups)
    mask = g.isin(elig)
    if not mask.any():
        return out

    cnt = g[mask].value_counts()
    scale = float(cnt.sum()) / max(float(len(cnt)), 1.0)
    grp_w = (scale / cnt).clip(lower=clip_min, upper=clip_max).to_dict()
    out[mask.to_numpy()] = g[mask].map(grp_w).to_numpy(dtype=float)
    return out


def compute_sample_weights(
    split_df: pd.DataFrame,
    group_col: str,
    eligible_groups: Iterable[str] | None = None,
    others_weight: float = 1.0,
    clip_min: float = 0.2,
    clip_max: float = 5.0,
) -> np.ndarray:
    if group_col not in split_df.columns:
        raise ValueError(f"split_df missing group column '{group_col}'")
    return _weights_from_groups(
        groups=split_df[group_col].to_numpy(),
        eligible_groups=eligible_groups,
        others_weight=others_weight,
        clip_min=clip_min,
        clip_max=clip_max,
    )


def compute_sample_weights_from_groups(
    groups: np.ndarray,
    eligible_groups: Iterable[str] | None = None,
    others_weight: float = 1.0,
    clip_min: float = 0.2,
    clip_max: float = 5.0,
) -> np.ndarray:
    return _weights_from_groups(
        groups=groups,
        eligible_groups=eligible_groups,
        others_weight=others_weight,
        clip_min=clip_min,
        clip_max=clip_max,
    )


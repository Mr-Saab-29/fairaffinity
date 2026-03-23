from __future__ import annotations

import numpy as np
import pandas as pd

from src.fairness.training import compute_sample_weights


def test_compute_sample_weights_inverse_frequency():
    df = pd.DataFrame(
        {
            "ClientID": [1, 2, 3, 4, 5, 6],
            "ClientGender": ["Male", "Male", "Female", "Female", "Female", "UNKNOWN"],
        }
    )
    w = compute_sample_weights(
        df,
        group_col="ClientGender",
        eligible_groups=["Male", "Female"],
        others_weight=1.0,
    )
    # Male appears 2x and Female 3x -> Male should receive larger balancing weight.
    w_male = float(np.mean(w[df["ClientGender"] == "Male"]))
    w_female = float(np.mean(w[df["ClientGender"] == "Female"]))
    w_other = float(np.mean(w[df["ClientGender"] == "UNKNOWN"]))
    assert w_male > w_female
    assert w_other == 1.0


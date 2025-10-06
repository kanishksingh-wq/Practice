import numpy as np
import pandas as pd
from typing import Dict


def psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    # Compute Population Stability Index for a single feature
    quantiles = np.linspace(0, 1, bins + 1)
    cuts = expected.quantile(quantiles).values
    cuts[0] = -np.inf
    cuts[-1] = np.inf
    e_counts = np.histogram(expected, bins=cuts)[0] / (len(expected) + 1e-12)
    a_counts = np.histogram(actual, bins=cuts)[0] / (len(actual) + 1e-12)
    a_counts = np.where(a_counts == 0, 1e-6, a_counts)
    e_counts = np.where(e_counts == 0, 1e-6, e_counts)
    return float(np.sum((e_counts - a_counts) * np.log(e_counts / a_counts)))


def compute_psi_frame(train_df: pd.DataFrame, test_df: pd.DataFrame, features, bins: int = 10) -> Dict[str, float]:
    out = {}
    for f in features:
        if pd.api.types.is_numeric_dtype(train_df[f]):
            out[f] = psi(train_df[f].astype(float), test_df[f].astype(float), bins=bins)
    return out

# Imports ------------------------------------------
from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests
# --------------------------------------------------


def adjust_pvalues(pvalues: pd.Series, method: str = 'fdr_bh') -> pd.Series:
    values = pvalues.astype(float).to_numpy()
    finite = np.isfinite(values)

    adjusted = pd.Series(np.nan, index = pvalues.index, dtype = float, name = pvalues.name)

    if not finite.any():
        return adjusted

    finite_values = values[finite]

    if np.any((finite_values < 0.0) | (finite_values > 1.0)):
        raise ValueError('P-values must be between zero and one.')

    adjusted_values = multipletests(finite_values, method = method)[1]
    adjusted.iloc[np.flatnonzero(finite)] = adjusted_values

    return adjusted.clip(lower = 0.0, upper = 1.0)
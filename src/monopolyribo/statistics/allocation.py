# Imports ------------------------------------------
from __future__ import annotations

from typing import Any

import pandas as pd
# --------------------------------------------------


def allocation_wide_counts(dataset: Any, fractions: list[str]) -> pd.DataFrame:
    if not fractions:
        raise ValueError('At least one allocation fraction is required.')

    metadata = dataset.metadata
    fraction_tables: list[pd.DataFrame] = []

    for fraction in fractions:
        sample_indices = metadata.index[metadata[dataset.fraction] == fraction]
        fraction_counts = dataset.filtered_counts.loc[sample_indices].copy()
        subject_ids = metadata.loc[sample_indices, dataset.subject].astype(str)

        if subject_ids.duplicated().any():
            raise ValueError(f'Multiple samples were found for the same subject in fraction {fraction!r}.')

        fraction_counts.index = subject_ids
        fraction_counts.columns = pd.MultiIndex.from_product([fraction_counts.columns, [fraction]])
        fraction_tables.append(fraction_counts)

    # Allocation models require every retained subject to have all selected fractions.
    return pd.concat(fraction_tables, axis = 1).dropna()


def default_fraction_weights(fractions: list[str], weights: dict[str, float] | None) -> dict[str, float]:
    if weights is not None:
        missing_fractions = [fraction for fraction in fractions if fraction not in weights]

        if missing_fractions:
            raise ValueError(f'Missing weights for fractions: {missing_fractions}.')

        return {fraction: float(weights[fraction]) for fraction in fractions}

    return {fraction: float(index) for index, fraction in enumerate(fractions)}
# Imports ------------------------------------------
from __future__ import annotations

import pandas as pd

from .exceptions import MonoPolyInputError
# --------------------------------------------------


def filter_counts(counts: pd.DataFrame, min_count: int, min_samples: int) -> pd.Index:
    _validate_filter_inputs(counts, min_count, min_samples)

    samples_at_threshold = (counts >= min_count).sum(axis = 0)

    return counts.columns[samples_at_threshold >= min_samples]


def filtering_table(counts: pd.DataFrame, kept: pd.Index, min_count: int, min_samples: int) -> pd.DataFrame:
    _validate_filter_inputs(counts, min_count, min_samples)

    if not isinstance(kept, pd.Index):
        raise TypeError('Kept features must be provided as a pandas Index.')

    unknown_features = kept.difference(counts.columns)

    if not unknown_features.empty:
        raise MonoPolyInputError(
            f'Kept features are absent from the count matrix: {unknown_features.tolist()[:5]}.'
        )

    samples_at_threshold = (counts >= min_count).sum(axis = 0)

    return pd.DataFrame(
        {
            'feature_id': counts.columns,
            'n_samples_at_min_count': samples_at_threshold.to_numpy(),
            'min_count': min_count,
            'min_samples': min_samples,
            'kept': counts.columns.isin(kept)
        },
        index = counts.columns
    )


def _validate_filter_inputs(counts: pd.DataFrame, min_count: int, min_samples: int) -> None:
    if not isinstance(counts, pd.DataFrame):
        raise TypeError('Counts must be provided as a pandas DataFrame.')

    if counts.empty:
        raise MonoPolyInputError('The count matrix must not be empty.')

    if not isinstance(min_count, int) or isinstance(min_count, bool):
        raise MonoPolyInputError('The minimum count threshold must be an integer.')

    if min_count < 0:
        raise MonoPolyInputError('The minimum count threshold must be nonnegative.')

    if not isinstance(min_samples, int) or isinstance(min_samples, bool):
        raise MonoPolyInputError('The minimum sample threshold must be an integer.')

    if min_samples < 1:
        raise MonoPolyInputError('The minimum sample threshold must be positive.')

    if min_samples > counts.shape[0]:
        raise MonoPolyInputError(
            'The minimum sample threshold cannot exceed the number of samples.'
        )
# Imports ------------------------------------------
from __future__ import annotations

import numpy as np
import pandas as pd

from .exceptions import MonoPolyInputError
# --------------------------------------------------


EXPOSURE_COLUMNS = {
    'gradient_area': 'gradient_area',
    'rna_yield': 'rna_yield',
    'spike_in': 'spike_in_factor'
}


def median_ratio_size_factors(counts: pd.DataFrame) -> pd.Series:
    _validate_count_matrix(counts)

    count_array = counts.to_numpy(dtype = float)

    with np.errstate(divide = 'ignore', invalid = 'ignore'):
        log_counts = np.where(count_array > 0.0, np.log(count_array), np.nan)

    valid_counts = np.sum(np.isfinite(log_counts), axis = 0)
    log_geometric_means = np.divide(
        np.nansum(log_counts, axis = 0),
        valid_counts,
        out = np.full(counts.shape[1], np.nan, dtype = float),
        where = valid_counts > 0
    )
    geometric_means = np.exp(log_geometric_means)

    valid_features = np.isfinite(geometric_means) & (geometric_means > 0.0)

    if not valid_features.any():
        return _library_size_factors(counts)

    ratios = count_array[:, valid_features] / geometric_means[valid_features]

    with np.errstate(invalid = 'ignore'):
        positive_ratios = np.where(ratios > 0.0, ratios, np.nan)
        size_factors = np.nanmedian(positive_ratios, axis = 1)

    size_factors = pd.Series(
        size_factors,
        index = counts.index,
        name = 'size_factor'
    )

    invalid_size_factors = (
        ~np.isfinite(size_factors.to_numpy())
        | (size_factors <= 0.0)
    )

    if invalid_size_factors.any():
        fallback_size_factors = _library_size_factors(counts)
        size_factors.loc[invalid_size_factors] = fallback_size_factors.loc[invalid_size_factors]

    median_size_factor = float(np.median(size_factors.to_numpy()))

    if not np.isfinite(median_size_factor) or median_size_factor <= 0.0:
        raise MonoPolyInputError('Size factors could not be normalized to a positive median.')

    return (size_factors / median_size_factor).rename('size_factor')


def recovery_factors(
    metadata: pd.DataFrame,
    mode: str,
    fraction_weights: str | pd.Series | dict[str, float] | None = None
) -> pd.Series:
    if not isinstance(metadata, pd.DataFrame):
        raise TypeError('Metadata must be provided as a pandas DataFrame.')

    if metadata.empty:
        raise MonoPolyInputError('Metadata must not be empty.')

    if mode == 'relative_library':
        return pd.Series(
            1.0,
            index = metadata.index,
            name = 'recovery_factor'
        )

    if mode == 'custom_exposure':
        exposure = _custom_exposure(metadata, fraction_weights)
    else:
        exposure_column = EXPOSURE_COLUMNS.get(mode)

        if exposure_column is None:
            raise MonoPolyInputError(
                f'Unsupported fraction measurement {mode!r}.'
            )

        if exposure_column not in metadata.columns:
            raise MonoPolyInputError(
                f'Metadata are missing the required exposure column {exposure_column!r}.'
)

        exposure = pd.to_numeric(
            metadata[exposure_column],
            errors = 'coerce'
        )

    exposure = exposure.reindex(metadata.index)
    _validate_exposure_factors(exposure)

    median_exposure = float(exposure.median())

    if not np.isfinite(median_exposure) or median_exposure <= 0.0:
        raise MonoPolyInputError('Exposure factors must have a finite positive median.')

    return (exposure / median_exposure).rename('recovery_factor')


def _library_size_factors(counts: pd.DataFrame) -> pd.Series:
    library_sizes = counts.sum(axis = 1).astype(float)
    positive_library_sizes = library_sizes.where(library_sizes > 0.0)
    median_library_size = float(positive_library_sizes.median())

    if not np.isfinite(median_library_size) or median_library_size <= 0.0:
        raise MonoPolyInputError(
            'Size factors could not be estimated because all library sizes are zero.'
        )

    size_factors = library_sizes / median_library_size
    size_factors = size_factors.where(size_factors > 0.0, 1.0)

    return size_factors.rename('size_factor')


def _custom_exposure(metadata: pd.DataFrame, fraction_weights: str | pd.Series | dict[str, float] | None) -> pd.Series:
    if isinstance(fraction_weights, str):
        if fraction_weights not in metadata.columns:
            raise MonoPolyInputError(f'Metadata are missing the custom exposure column {fraction_weights!r}.')

        return pd.to_numeric(
            metadata[fraction_weights],
            errors = 'coerce'
        )

    if isinstance(fraction_weights, pd.Series):
        return pd.to_numeric(
            fraction_weights.reindex(metadata.index),
            errors = 'coerce'
        )

    if isinstance(fraction_weights, dict):
        return pd.to_numeric(
            pd.Series(fraction_weights).reindex(metadata.index),
            errors = 'coerce'
        )

    raise MonoPolyInputError(
        'Custom exposure requires a metadata column name, pandas Series, or '
        'sample-indexed dictionary.'
    )


def _validate_count_matrix(counts: pd.DataFrame) -> None:
    if not isinstance(counts, pd.DataFrame):
        raise TypeError('Counts must be provided as a pandas DataFrame.')

    if counts.empty:
        raise MonoPolyInputError('The count matrix must not be empty.')

    if counts.index.has_duplicates:
        raise MonoPolyInputError('The count matrix must contain unique sample identifiers.')

    if counts.columns.has_duplicates:
        raise MonoPolyInputError('The count matrix must contain unique feature identifiers.')

    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in counts.dtypes):
        raise MonoPolyInputError('The count matrix must contain only numeric columns.')

    count_array = counts.to_numpy(dtype = float)

    if not np.all(np.isfinite(count_array)):
        raise MonoPolyInputError('The count matrix must contain only finite values.')

    if np.any(count_array < 0.0):
        raise MonoPolyInputError('The count matrix must contain only nonnegative values.')


def _validate_exposure_factors(exposure: pd.Series) -> None:
    exposure_array = exposure.to_numpy(dtype = float)

    if exposure.isna().any() or not np.all(np.isfinite(exposure_array)):
        raise MonoPolyInputError(
            'Exposure factors must contain finite values for every sample.'
        )

    if np.any(exposure_array <= 0.0):
        raise MonoPolyInputError(
            'Exposure factors must contain only positive values.'
        )

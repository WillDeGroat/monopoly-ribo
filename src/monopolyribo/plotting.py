# Imports ------------------------------------------
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
# --------------------------------------------------


def plot_pca(counts: pd.DataFrame) -> tuple[Figure, Axes]:
    _validate_numeric_dataframe(counts, 'The count matrix')

    if counts.shape[0] < 2:
        raise ValueError('PCA requires at least two samples.')

    if counts.shape[1] < 2:
        raise ValueError('PCA requires at least two features.')

    if (counts < 0.0).any().any():
        raise ValueError('The count matrix must contain only nonnegative values.')

    transformed_counts = np.log2(counts.astype(float) + 1.0)
    centered_counts = transformed_counts - transformed_counts.mean(axis = 0)

    left_vectors, singular_values, _ = np.linalg.svd(
        centered_counts.to_numpy(),
        full_matrices = False
    )

    if len(singular_values) < 2:
        raise ValueError('PCA requires at least two nonzero dimensions.')

    scores = left_vectors[:, :2] * singular_values[:2]

    total_variance = np.sum(np.square(singular_values))
    explained_variance = np.square(singular_values[:2]) / total_variance if total_variance > 0.0 else np.zeros(2)

    figure, axes = plt.subplots()
    axes.scatter(scores[:, 0], scores[:, 1])
    axes.set_xlabel(f'PC1 ({explained_variance[0]:.1%})')
    axes.set_ylabel(f'PC2 ({explained_variance[1]:.1%})')
    axes.set_title('Sample PCA')

    return figure, axes


def volcano_plot(results: pd.DataFrame) -> tuple[Figure, Axes]:
    _validate_required_columns(results, {'effect', 'pvalue'}, 'Volcano plot results')

    effect = pd.to_numeric(results['effect'], errors = 'coerce')
    pvalue = pd.to_numeric(results['pvalue'], errors = 'coerce')
    valid = np.isfinite(effect) & np.isfinite(pvalue) & (pvalue >= 0.0) & (pvalue <= 1.0)

    if not valid.any():
        raise ValueError('Volcano plot results contain no valid effect and p-value pairs.')

    negative_log10_pvalue = -np.log10(pvalue[valid].clip(lower = 1e-300))

    figure, axes = plt.subplots()
    axes.scatter(effect[valid], negative_log10_pvalue)
    axes.axvline(0.0, linewidth = 1.0)
    axes.set_xlabel('Effect')
    axes.set_ylabel('-log10(p-value)')
    axes.set_title('Volcano Plot')

    return figure, axes


def fraction_profiles(
    counts: pd.DataFrame,
    metadata: pd.DataFrame,
    fraction: str
) -> tuple[Figure, Axes]:
    _validate_numeric_dataframe(counts, 'The count matrix')

    if fraction not in metadata.columns:
        raise ValueError(f'Fraction column {fraction!r} was not found in the metadata.')

    if not counts.index.equals(metadata.index):
        raise ValueError('The count matrix and metadata must have matching sample indices.')

    if metadata[fraction].isna().any():
        raise ValueError('Fraction metadata must not contain missing values.')

    sample_totals = counts.sum(axis = 1)
    fraction_means = sample_totals.groupby(metadata[fraction], sort = False).mean()

    if fraction_means.empty:
        raise ValueError('No fraction profiles are available to plot.')

    figure, axes = plt.subplots()
    fraction_means.plot(ax = axes, marker = 'o')
    axes.set_xlabel(fraction)
    axes.set_ylabel('Mean total count')
    axes.set_title('Fraction Profile')

    return figure, axes


def effect_comparison(first: pd.DataFrame, second: pd.DataFrame) -> tuple[Figure, Axes]:
    _validate_required_columns(first, {'effect'}, 'The first result table')
    _validate_required_columns(second, {'effect'}, 'The second result table')

    if first.index.has_duplicates or second.index.has_duplicates:
        raise ValueError('Effect comparison tables must have unique feature indices.')

    shared_features = first.index.intersection(second.index)

    if shared_features.empty:
        raise ValueError('The result tables do not contain any shared features.')

    first_effect = pd.to_numeric(first.loc[shared_features, 'effect'], errors = 'coerce')
    second_effect = pd.to_numeric(second.loc[shared_features, 'effect'], errors = 'coerce')
    valid = np.isfinite(first_effect) & np.isfinite(second_effect)

    if not valid.any():
        raise ValueError('The result tables contain no valid shared effect estimates.')

    figure, axes = plt.subplots()
    axes.scatter(first_effect[valid], second_effect[valid])
    axes.axhline(0.0, linewidth = 1.0)
    axes.axvline(0.0, linewidth = 1.0)
    axes.set_xlabel('First effect')
    axes.set_ylabel('Second effect')
    axes.set_title('Effect Comparison')

    return figure, axes


def loso_effects(results: pd.DataFrame) -> tuple[Figure, Axes]:
    _validate_required_columns(
        results,
        {'full_effect', 'subset_effect'},
        'Leave-one-subject-out results'
    )

    full_effect = pd.to_numeric(results['full_effect'], errors = 'coerce')
    subset_effect = pd.to_numeric(results['subset_effect'], errors = 'coerce')
    valid = np.isfinite(full_effect) & np.isfinite(subset_effect)

    if not valid.any():
        raise ValueError('Leave-one-subject-out results contain no valid effect pairs.')

    minimum_effect = min(full_effect[valid].min(), subset_effect[valid].min())
    maximum_effect = max(full_effect[valid].max(), subset_effect[valid].max())

    figure, axes = plt.subplots()
    axes.scatter(full_effect[valid], subset_effect[valid])
    axes.plot(
        [minimum_effect, maximum_effect],
        [minimum_effect, maximum_effect],
        linestyle = '--',
        linewidth = 1.0
    )
    axes.set_xlabel('Full-data effect')
    axes.set_ylabel('Leave-one-subject-out effect')
    axes.set_title('Leave-One-Subject-Out Effects')

    return figure, axes


def allocation_probability_plot(results: pd.DataFrame) -> tuple[Figure, Axes]:
    if not isinstance(results, pd.DataFrame):
        raise TypeError('Allocation results must be provided as a pandas DataFrame.')

    numeric_results = results.select_dtypes(include = 'number')

    if numeric_results.empty:
        raise ValueError('Allocation results contain no numeric columns to plot.')

    finite_columns = [
        column
        for column in numeric_results.columns
        if np.isfinite(numeric_results[column].to_numpy(dtype = float)).any()
    ]

    if not finite_columns:
        raise ValueError('Allocation results contain no finite numeric values to plot.')

    figure, axes = plt.subplots()
    numeric_results[finite_columns].plot(ax = axes)
    axes.set_xlabel('Observation')
    axes.set_ylabel('Value')
    axes.set_title('Allocation Profile')

    return figure, axes


def _validate_numeric_dataframe(dataframe: pd.DataFrame, name: str) -> None:
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError(f'{name} must be provided as a pandas DataFrame.')

    if dataframe.empty:
        raise ValueError(f'{name} must not be empty.')

    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in dataframe.dtypes):
        raise ValueError(f'{name} must contain only numeric columns.')

    if not np.all(np.isfinite(dataframe.to_numpy(dtype = float))):
        raise ValueError(f'{name} must contain only finite values.')


def _validate_required_columns(results: pd.DataFrame, required_columns: set[str], name: str) -> None:
    if not isinstance(results, pd.DataFrame):
        raise TypeError(f'{name} must be provided as a pandas DataFrame.')

    missing_columns = required_columns.difference(results.columns)

    if missing_columns:
        raise ValueError(f'{name} are missing required columns: {sorted(missing_columns)}.')
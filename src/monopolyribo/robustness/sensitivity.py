# Imports ------------------------------------------
from __future__ import annotations

from typing import Any
from warnings import warn

import pandas as pd

from ..contrasts import FractionContrast
# --------------------------------------------------


SENSITIVITY_RESULT_COLUMNS = [
    'feature_id',
    'effect',
    'standard_error',
    'pvalue',
    'padj',
    'converged',
    'engine',
    'contrast'
]

REQUIRED_RESULT_COLUMNS = {
    'feature_id',
    'effect',
    'standard_error',
    'pvalue',
    'padj',
    'converged'
}


def filter_sensitivity(dataset: Any, contrast: FractionContrast, engine: str = 'nb_interaction', min_counts: list[int] | None = None) -> pd.DataFrame:
    from ..dataset import MonoPolyDataSet
    from ..stats import MonoPolyStats

    thresholds = [dataset.min_count] if min_counts is None else min_counts

    if not thresholds:
        raise ValueError('At least one minimum count threshold is required.')

    if any(not isinstance(min_count, int) or isinstance(min_count, bool) for min_count in thresholds):
        raise ValueError('Minimum count thresholds must be integers.')

    if any(min_count < 0 for min_count in thresholds):
        raise ValueError('Minimum count thresholds must be nonnegative.')

    result_tables: list[pd.DataFrame] = []

    for min_count in thresholds:
        try:
            sensitivity_dataset = MonoPolyDataSet(
                counts = dataset.counts,
                metadata = dataset.metadata,
                subject = dataset.subject,
                condition = dataset.condition,
                case = dataset.case,
                control = dataset.control,
                fraction = dataset.fraction,
                fraction_order = dataset.fraction_order,
                covariates = dataset.covariates,
                abundance_fraction = dataset.abundance_fraction,
                allocation_fractions = dataset.allocation_fractions,
                engines = [engine],
                fraction_measurement = dataset.fraction_measurement,
                fraction_weights = dataset.fraction_weights,
                min_count = min_count,
                min_samples = dataset.min_samples,
                n_cpus = dataset.n_cpus,
                seed = dataset.seed,
                quiet = dataset.quiet
            ).fit()

            results = MonoPolyStats(
                sensitivity_dataset,
                contrast = contrast,
                engine = engine
            ).summary()

            missing_columns = REQUIRED_RESULT_COLUMNS.difference(results.columns)

            if missing_columns:
                raise ValueError(
                    f'Sensitivity results are missing required columns: {sorted(missing_columns)}.'
                )

            sensitivity_results = results[
                [
                    'feature_id',
                    'effect',
                    'standard_error',
                    'pvalue',
                    'padj',
                    'converged'
                ]
            ].copy()

            sensitivity_results['engine'] = engine
            sensitivity_results['contrast'] = contrast.label
            sensitivity_results['min_count'] = min_count
            sensitivity_results['min_samples'] = dataset.min_samples
            result_tables.append(sensitivity_results)

        except Exception as exc:
            warn(
                f'Filter sensitivity analysis failed for min_count={min_count}: {exc}',
                RuntimeWarning,
                stacklevel = 2
            )

    if not result_tables:
        return pd.DataFrame(
            columns = SENSITIVITY_RESULT_COLUMNS + ['min_count', 'min_samples']
        )

    return pd.concat(result_tables, axis = 0, ignore_index = True, sort = False)


def normalization_sensitivity(dataset: Any, contrast: FractionContrast, engine: str = 'nb_interaction') -> pd.DataFrame:
    from ..stats import MonoPolyStats

    results = MonoPolyStats(
        dataset,
        contrast = contrast,
        engine = engine
    ).summary()

    missing_columns = REQUIRED_RESULT_COLUMNS.difference(results.columns)

    if missing_columns:
        raise ValueError(
            f'Sensitivity results are missing required columns: {sorted(missing_columns)}.'
        )

    sensitivity_results = results[
        [
            'feature_id',
            'effect',
            'standard_error',
            'pvalue',
            'padj',
            'converged'
        ]
    ].copy()

    sensitivity_results['engine'] = engine
    sensitivity_results['contrast'] = contrast.label
    sensitivity_results['normalization'] = 'median_ratio'

    return sensitivity_results.reset_index(drop = True)
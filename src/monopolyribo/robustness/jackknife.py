# Imports ------------------------------------------
from __future__ import annotations

from typing import Any
from warnings import warn

import numpy as np
import pandas as pd

from ..contrasts import FractionContrast
# --------------------------------------------------


REQUIRED_RESULT_COLUMNS = {
    'feature_id',
    'effect',
    'padj',
    'converged'
}

LOSO_DETAIL_COLUMNS = [
    'feature_id',
    'engine',
    'contrast',
    'omitted_subject',
    'full_effect',
    'subset_effect',
    'effect_change',
    'absolute_effect_change',
    'effect_retention',
    'same_direction',
    'full_rank',
    'subset_rank',
    'converged'
]

LOSO_SUMMARY_COLUMNS = [
    'engine',
    'contrast',
    'median_subset_effect',
    'minimum_subset_effect',
    'maximum_subset_effect',
    'direction_concordance',
    'median_effect_retention',
    'largest_influence_subject',
    'largest_absolute_effect_change',
    'rank_stability',
    'n_successful_fits',
    'n_attempted_fits',
    'stability_class'
]


def leave_one_subject_out(dataset: Any, contrast: FractionContrast, engine: str = 'nb_interaction') -> tuple[pd.DataFrame, pd.DataFrame]:
    from ..dataset import MonoPolyDataSet
    from ..stats import MonoPolyStats

    subject_ids = pd.unique(dataset.metadata[dataset.subject])

    if len(subject_ids) < 2:
        raise ValueError('Leave-one-subject-out analysis requires at least two subjects.')

    full_results = MonoPolyStats(
        dataset,
        contrast = contrast,
        engine = engine
    ).summary()

    _validate_results(full_results)
    full_results = full_results.set_index('feature_id', drop = False)
    full_ranks = _result_ranks(full_results)

    result_rows: list[dict[str, Any]] = []

    for omitted_subject in subject_ids:
        keep_samples = dataset.metadata[dataset.subject] != omitted_subject

        try:
            subset_dataset = MonoPolyDataSet(
                counts = dataset.counts.loc[keep_samples],
                metadata = dataset.metadata.loc[keep_samples],
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
                min_count = dataset.min_count,
                min_samples = dataset.min_samples,
                n_cpus = dataset.n_cpus,
                seed = dataset.seed,
                quiet = dataset.quiet
            ).fit()

            subset_results = MonoPolyStats(
                subset_dataset,
                contrast = contrast,
                engine = engine
            ).summary()

            _validate_results(subset_results)
            subset_results = subset_results.set_index('feature_id', drop = False)
            subset_ranks = _result_ranks(subset_results)

            shared_features = full_results.index.intersection(subset_results.index)

            for feature_id in shared_features:
                full_effect = float(full_results.loc[feature_id, 'effect'])
                subset_effect = float(subset_results.loc[feature_id, 'effect'])
                effect_change = subset_effect - full_effect

                result_rows.append(
                    {
                        'feature_id': feature_id,
                        'engine': engine,
                        'contrast': contrast.label,
                        'omitted_subject': omitted_subject,
                        'full_effect': full_effect,
                        'subset_effect': subset_effect,
                        'effect_change': effect_change,
                        'absolute_effect_change': abs(effect_change),
                        'effect_retention': _effect_retention(full_effect, subset_effect),
                        'same_direction': _same_direction(full_effect, subset_effect),
                        'full_rank': full_ranks.loc[feature_id],
                        'subset_rank': subset_ranks.loc[feature_id],
                        'converged': bool(subset_results.loc[feature_id, 'converged'])
                    }
                )

        except Exception as exc:
            warn(
                f'Leave-one-subject-out analysis failed when omitting subject '
                f'{omitted_subject!r}: {exc}',
                RuntimeWarning,
                stacklevel = 2
            )

    if not result_rows:
        detail = pd.DataFrame(columns = LOSO_DETAIL_COLUMNS)
        summary = pd.DataFrame(columns = LOSO_SUMMARY_COLUMNS)
        return detail, summary

    detail = pd.DataFrame(result_rows, columns = LOSO_DETAIL_COLUMNS)
    summary = _summarize_leave_one_subject_out(detail)

    return detail, summary


def _validate_results(results: pd.DataFrame) -> None:
    missing_columns = REQUIRED_RESULT_COLUMNS.difference(results.columns)

    if missing_columns:
        raise ValueError(
            f'Leave-one-subject-out results are missing required columns: '
            f'{sorted(missing_columns)}.'
        )

    if results['feature_id'].duplicated().any():
        raise ValueError('Leave-one-subject-out results must contain unique feature identifiers.')


def _result_ranks(results: pd.DataFrame) -> pd.Series:
    return results['padj'].rank(method = 'min', na_option = 'bottom').astype(int)


def _effect_retention(full_effect: float, subset_effect: float) -> float:
    if not np.isfinite(full_effect) or not np.isfinite(subset_effect):
        return np.nan

    if np.isclose(full_effect, 0.0):
        return np.nan

    return subset_effect / full_effect


def _same_direction(full_effect: float, subset_effect: float) -> bool | float:
    if not np.isfinite(full_effect) or not np.isfinite(subset_effect):
        return np.nan

    return bool(np.sign(full_effect) == np.sign(subset_effect))


def _summarize_leave_one_subject_out(detail: pd.DataFrame) -> pd.DataFrame:
    successful = detail[detail['converged']].copy()

    if successful.empty:
        return pd.DataFrame(columns = LOSO_SUMMARY_COLUMNS)

    summary = successful.groupby('feature_id').agg(
        engine = ('engine', 'first'),
        contrast = ('contrast', 'first'),
        median_subset_effect = ('subset_effect', 'median'),
        minimum_subset_effect = ('subset_effect', 'min'),
        maximum_subset_effect = ('subset_effect', 'max'),
        direction_concordance = ('same_direction', 'mean'),
        median_effect_retention = ('effect_retention', 'median'),
        largest_absolute_effect_change = ('absolute_effect_change', 'max'),
        rank_stability = ('subset_rank', 'std'),
        n_successful_fits = ('converged', 'sum')
    )

    attempted_fits = detail.groupby('feature_id').size()
    summary['engine'] = successful.groupby('feature_id')['engine'].first().reindex(summary.index)
    summary['n_attempted_fits'] = attempted_fits.reindex(summary.index).astype(int)

    largest_influence_indices = successful.groupby('feature_id')['absolute_effect_change'].idxmax()
    largest_influence_subjects = successful.loc[
        largest_influence_indices,
        ['feature_id', 'omitted_subject']
    ].set_index('feature_id')['omitted_subject']

    summary['largest_influence_subject'] = largest_influence_subjects.reindex(summary.index)
    summary['stability_class'] = np.where(
        summary['direction_concordance'] >= 0.9,
        'robust',
        'single_subject_sensitive'
    )

    return summary[LOSO_SUMMARY_COLUMNS]
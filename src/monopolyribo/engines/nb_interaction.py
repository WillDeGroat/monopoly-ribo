# Imports ------------------------------------------
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import optimize, stats

from ..statistics import (
    adjust_pvalues,
    allocation_wide_counts,
    default_fraction_weights,
    dirichlet_multinomial_nll
)
from .base import EngineFit
# --------------------------------------------------


LIKELIHOOD_RATIO_TOLERANCE = 1e-6


class DirichletMultinomialEngine:
    name: str = 'dirichlet_multinomial'

    def fit(self, dataset: Any) -> EngineFit:
        fractions = dataset.allocation_fractions or [
            fraction
            for fraction in dataset.fraction_order
            if fraction != dataset.abundance_fraction
        ]

        if len(fractions) < 2:
            raise ValueError(
                'The Dirichlet-multinomial engine requires at least two allocation fractions.'
            )

        fraction_weights = default_fraction_weights(
            fractions,
            dataset.fraction_weights
        )

        wide_counts = allocation_wide_counts(
            dataset,
            fractions
        )

        subject_conditions = _subject_conditions(dataset)
        aligned_conditions = subject_conditions.reindex(wide_counts.index)
        condition_indicator = _condition_indicator(
            aligned_conditions,
            dataset.case,
            dataset.control
        )

        result_rows: list[dict[str, Any]] = []

        for feature_id in dataset.filtered_counts.columns:
            feature_counts = np.column_stack(
                [
                    wide_counts[(feature_id, fraction)].to_numpy(dtype = float)
                    for fraction in fractions
                ]
            )

            result_rows.append(
                _fit_feature(
                    feature_id = feature_id,
                    counts = feature_counts,
                    condition_indicator = condition_indicator,
                    fractions = fractions,
                    fraction_weights = fraction_weights,
                    dataset = dataset
                )
            )

        results = pd.DataFrame(result_rows).set_index('feature_id')
        results['padj'] = adjust_pvalues(results['pvalue'])

        return EngineFit(
            name = self.name,
            results = {'allocation': results},
            metadata = {
                'fraction_measurement': dataset.fraction_measurement,
                'fractions': fractions,
                'fraction_weights': fraction_weights,
                'case': dataset.case,
                'control': dataset.control,
                'condition_coding': 'case=1, control=0'
            }
        )


def _subject_conditions(dataset: Any) -> pd.Series:
    subject_metadata = dataset.metadata[
        [dataset.subject, dataset.condition]
    ].copy()

    condition_counts = subject_metadata.groupby(
        dataset.subject,
        observed = True
    )[dataset.condition].nunique()

    if (condition_counts != 1).any():
        raise ValueError('Each subject must be associated with exactly one condition.')

    return (
        subject_metadata.drop_duplicates(dataset.subject)
        .set_index(dataset.subject)[dataset.condition]
        .astype(str)
    )


def _condition_indicator(
    subject_conditions: pd.Series,
    case: str,
    control: str
) -> np.ndarray:
    if subject_conditions.isna().any():
        raise ValueError('Condition metadata are missing for one or more subjects.')

    observed_conditions = set(subject_conditions.astype(str).unique())
    expected_conditions = {case, control}

    if observed_conditions != expected_conditions:
        raise ValueError(
            'The Dirichlet-multinomial engine requires subject conditions to match '
            f'the configured case and control exactly. Expected '
            f'{sorted(expected_conditions)!r}, observed {sorted(observed_conditions)!r}.'
        )

    return subject_conditions.astype(str).eq(case).to_numpy(dtype = float)


def _fit_feature(
    feature_id: str,
    counts: np.ndarray,
    condition_indicator: np.ndarray,
    fractions: list[str],
    fraction_weights: dict[str, float],
    dataset: Any
) -> dict[str, Any]:
    informative = counts.sum(axis = 1) > 0.0
    counts = counts[informative]
    condition_indicator = condition_indicator[informative]

    n_informative = counts.shape[0]
    n_fractions = len(fractions)

    if n_informative < 3 or len(np.unique(condition_indicator)) < 2:
        return _result_row(
            feature_id = feature_id,
            effect = np.nan,
            standard_error = np.nan,
            statistic = np.nan,
            pvalue = np.nan,
            converged = False,
            warning_code = 'low_information',
            dataset = dataset,
            n_informative = n_informative
        )

    pooled_probabilities = (
        counts.sum(axis = 0) + 0.5
    ) / (
        counts.sum() + 0.5 * n_fractions
    )

    reference_probability = pooled_probabilities[0]
    intercepts = np.log(pooled_probabilities[1:] / reference_probability)
    condition_effects = np.zeros(n_fractions - 1, dtype = float)
    log_concentration = np.log(20.0)

    full_initial_parameters = np.concatenate(
        [
            intercepts,
            condition_effects,
            [log_concentration]
        ]
    )

    reduced_initial_parameters = np.concatenate(
        [
            intercepts,
            [log_concentration]
        ]
    )

    full_model = optimize.minimize(
        dirichlet_multinomial_nll,
        full_initial_parameters,
        args = (
            counts,
            condition_indicator,
            n_fractions,
            True
        ),
        method = 'L-BFGS-B',
        bounds = (
            [(-8.0, 8.0)] * (2 * (n_fractions - 1))
            + [(-3.0, 8.0)]
        )
    )

    reduced_model = optimize.minimize(
        dirichlet_multinomial_nll,
        reduced_initial_parameters,
        args = (
            counts,
            condition_indicator,
            n_fractions,
            False
        ),
        method = 'L-BFGS-B',
        bounds = (
            [(-8.0, 8.0)] * (n_fractions - 1)
            + [(-3.0, 8.0)]
        )
    )

    if not _valid_optimization(full_model) or not _valid_optimization(reduced_model):
        return _result_row(
            feature_id = feature_id,
            effect = np.nan,
            standard_error = np.nan,
            statistic = np.nan,
            pvalue = np.nan,
            converged = False,
            warning_code = 'nonconverged',
            dataset = dataset,
            n_informative = n_informative
        )

    full_nll = dirichlet_multinomial_nll(
        full_model.x,
        counts,
        condition_indicator,
        n_fractions,
        True,
        penalty_strength = 0.0
    )

    reduced_nll = dirichlet_multinomial_nll(
        reduced_model.x,
        counts,
        condition_indicator,
        n_fractions,
        False,
        penalty_strength = 0.0
    )

    likelihood_ratio = 2.0 * (reduced_nll - full_nll)

    if likelihood_ratio < -LIKELIHOOD_RATIO_TOLERANCE:
        return _result_row(
            feature_id = feature_id,
            effect = np.nan,
            standard_error = np.nan,
            statistic = np.nan,
            pvalue = np.nan,
            converged = False,
            warning_code = 'invalid_likelihood_ratio',
            dataset = dataset,
            n_informative = n_informative
        )

    likelihood_ratio = max(0.0, likelihood_ratio)

    pvalue = float(
        stats.chi2.sf(
            likelihood_ratio,
            n_fractions - 1
        )
    )

    # Case is coded as 1 and control as 0, so these are case-versus-control effects.
    condition_effects = full_model.x[
        n_fractions - 1:2 * (n_fractions - 1)
    ]

    weight_differences = np.array(
        [
            fraction_weights[fraction] - fraction_weights[fractions[0]]
            for fraction in fractions[1:]
        ],
        dtype = float
    )

    effect = float(
        np.dot(
            weight_differences,
            condition_effects
        ) / np.log(2.0)
    )

    warning_code = (
        'relative_library_caution'
        if dataset.fraction_measurement == 'relative_library'
        else ''
    )

    return _result_row(
        feature_id = feature_id,
        effect = effect,
        standard_error = np.nan,
        statistic = likelihood_ratio,
        pvalue = pvalue,
        converged = True,
        warning_code = warning_code,
        dataset = dataset,
        n_informative = n_informative
    )


def _valid_optimization(result: optimize.OptimizeResult) -> bool:
    return bool(
        result.success
        and np.isfinite(result.fun)
        and np.all(np.isfinite(result.x))
    )


def _result_row(
    feature_id: str,
    effect: float,
    standard_error: float,
    statistic: float,
    pvalue: float,
    converged: bool,
    warning_code: str,
    dataset: Any,
    n_informative: int
) -> dict[str, Any]:
    return {
        'feature_id': feature_id,
        'contrast': f'{dataset.case}_vs_{dataset.control}:global_condition_allocation',
        'engine': DirichletMultinomialEngine.name,
        'effect': effect,
        'effect_scale': 'weighted_log2_allocation_shift',
        'standard_error': standard_error,
        'statistic': statistic,
        'pvalue': pvalue,
        'base_mean': dataset.counts[feature_id].mean(),
        'n_samples': dataset.counts.shape[0],
        'n_subjects': dataset.metadata[dataset.subject].nunique(),
        'n_informative': n_informative,
        'converged': converged,
        'warning_code': warning_code
    }
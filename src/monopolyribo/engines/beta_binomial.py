# Imports ------------------------------------------
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import optimize, special, stats

from ..statistics import adjust_pvalues, allocation_wide_counts, beta_binomial_nll
from .base import EngineFit
# --------------------------------------------------


class BetaBinomialEngine:
    name: str = 'beta_binomial'

    def fit(self, dataset: Any) -> EngineFit:
        fractions = dataset.allocation_fractions or [
            fraction for fraction in dataset.fraction_order
            if fraction != dataset.abundance_fraction
        ]

        if len(fractions) != 2:
            raise ValueError('The beta-binomial engine requires exactly two allocation fractions.')

        denominator_fraction = fractions[0]
        numerator_fraction = fractions[1]
        contrast_label = (
            f'{dataset.case}_vs_{dataset.control}:'
            f'{numerator_fraction}_vs_{denominator_fraction}'
        )

        wide_counts = allocation_wide_counts(dataset, [denominator_fraction, numerator_fraction])

        subject_conditions = _subject_conditions(dataset)
        aligned_conditions = subject_conditions.reindex(wide_counts.index)
        
        condition_indicator = _condition_indicator(
            aligned_conditions,
            dataset.case,
            dataset.control
        )

        result_rows: list[dict[str, Any]] = []

        for feature_id in dataset.filtered_counts.columns:
            numerator_counts = wide_counts[(feature_id, numerator_fraction)].to_numpy(dtype = float)
            denominator_counts = wide_counts[(feature_id, denominator_fraction)].to_numpy(dtype = float)

            total_counts = numerator_counts + denominator_counts
            informative = total_counts > 0.0

            result_rows.append(
                _fit_feature(
                    feature_id,
                    numerator_counts[informative],
                    total_counts[informative],
                    condition_indicator[informative],
                    dataset,
                    contrast_label
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
                'case': dataset.case,
                'control': dataset.control,
                'condition_coding': 'case=1, control=0'
            }
        )


def _subject_conditions(dataset: Any) -> pd.Series:
    subject_metadata = dataset.metadata[[dataset.subject, dataset.condition]]
    condition_counts = subject_metadata.groupby(dataset.subject)[dataset.condition].nunique()

    if (condition_counts > 1).any():
        raise ValueError('Each subject must be associated with exactly one condition.')

    return (
        subject_metadata.drop_duplicates(dataset.subject)
        .set_index(dataset.subject)[dataset.condition]
        .astype(str)
    )


def _condition_indicator(subject_conditions: pd.Series, case: str, control: str) -> np.ndarray:
    if subject_conditions.isna().any():
        raise ValueError('Condition metadata are missing for one or more subjects.')

    observed_conditions = set(subject_conditions.astype(str).unique())
    expected_conditions = {case, control}

    if observed_conditions != expected_conditions:
        raise ValueError(
            'The beta-binomial engine requires subject conditions to match '
            f'the configured case and control exactly. Expected '
            f'{sorted(expected_conditions)!r}, observed '
            f'{sorted(observed_conditions)!r}.'
        )

    return subject_conditions.astype(str).eq(case).to_numpy(dtype = float)


def _fit_feature(
    feature_id: str,
    numerator_counts: np.ndarray,
    total_counts: np.ndarray,
    condition_indicator: np.ndarray,
    dataset: Any,
    contrast_label: str
) -> dict[str, Any]:
    n_informative = len(numerator_counts)

    if n_informative < 3 or len(np.unique(condition_indicator)) < 2:
        return _result_row(
            feature_id,
            contrast_label,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            False,
            'low_information',
            dataset,
            n_informative
        )

    pooled_allocation = (numerator_counts.sum() + 0.5) / (total_counts.sum() + 1.0)

    initial_parameters = np.array(
        [
            special.logit(np.clip(pooled_allocation, 0.01, 0.99)),
            0.0,
            np.log(0.05)
        ],
        dtype = float
    )

    optimization = optimize.minimize(
        beta_binomial_nll,
        initial_parameters,
        args = (numerator_counts, total_counts, condition_indicator),
        method = 'L-BFGS-B',
        bounds = [(-10.0, 10.0), (-10.0, 10.0), (-8.0, 3.0)]
    )

    if not optimization.success or not np.all(np.isfinite(optimization.x)):
        return _result_row(
            feature_id,
            contrast_label,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            False,
            'nonconverged',
            dataset,
            n_informative
        )

    effect = float(optimization.x[1] / np.log(2.0))
    inverse_hessian = _inverse_hessian(optimization.hess_inv)
    effect_variance = float(inverse_hessian[1, 1])

    if not np.isfinite(effect_variance) or effect_variance <= 0.0:
        return _result_row(
            feature_id,
            contrast_label,
            effect,
            np.nan,
            np.nan,
            np.nan,
            False,
            'invalid_hessian',
            dataset,
            n_informative
        )

    standard_error = float(np.sqrt(effect_variance) / np.log(2.0))
    statistic = effect / standard_error
    pvalue = float(2.0 * stats.norm.sf(abs(statistic)))

    warning_code = (
        'relative_library_caution'
        if dataset.fraction_measurement == 'relative_library'
        else ''
    )

    return _result_row(
        feature_id,
        contrast_label,
        effect,
        standard_error,
        statistic,
        pvalue,
        True,
        warning_code,
        dataset,
        n_informative
    )


def _inverse_hessian(hessian: Any) -> np.ndarray:
    if hasattr(hessian, 'todense'):
        inverse_hessian = np.asarray(hessian.todense(), dtype = float)
    else:
        inverse_hessian = np.asarray(hessian, dtype = float)

    if inverse_hessian.shape != (3, 3):
        raise ValueError('The beta-binomial inverse Hessian must have shape (3, 3).')

    if not np.all(np.isfinite(inverse_hessian)):
        raise ValueError('The beta-binomial inverse Hessian must contain only finite values.')

    return inverse_hessian


def _result_row(
    feature_id: str,
    contrast_label: str,
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
        'contrast': contrast_label,
        'engine': BetaBinomialEngine.name,
        'effect': effect,
        'effect_scale': 'log2_allocation_odds_ratio',
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

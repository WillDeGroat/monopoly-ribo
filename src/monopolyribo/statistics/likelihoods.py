# Imports ------------------------------------------
from __future__ import annotations

import numpy as np
from scipy import special
# --------------------------------------------------


RIDGE_PENALTY = 1e-4


def beta_binomial_nll(parameters: np.ndarray, numerator_counts: np.ndarray, total_counts: np.ndarray, condition_indicator: np.ndarray) -> float:
    if parameters.shape != (3,):
        raise ValueError('The beta-binomial model requires exactly three parameters.')

    if numerator_counts.ndim != 1 or total_counts.ndim != 1 or condition_indicator.ndim != 1:
        raise ValueError('The beta-binomial input arrays must be one-dimensional.')

    if not (numerator_counts.shape == total_counts.shape == condition_indicator.shape):
        raise ValueError('The beta-binomial input arrays must have matching shapes.')

    if not (
        np.all(np.isfinite(parameters))
        and np.all(np.isfinite(numerator_counts))
        and np.all(np.isfinite(total_counts))
        and np.all(np.isfinite(condition_indicator))
    ):
        raise ValueError('The beta-binomial inputs must contain only finite values.')

    if np.any(numerator_counts < 0.0) or np.any(total_counts < 0.0):
        raise ValueError('Beta-binomial counts must be nonnegative.')

    if np.any(numerator_counts > total_counts):
        raise ValueError('Numerator counts cannot exceed total counts.')

    intercept, condition_effect, log_dispersion = parameters

    allocation_probability = special.expit(intercept + condition_effect * condition_indicator)
    dispersion = np.exp(log_dispersion)
    alpha = allocation_probability / dispersion
    beta = (1.0 - allocation_probability) / dispersion

    log_likelihood = (
        special.gammaln(total_counts + 1.0)
        - special.gammaln(numerator_counts + 1.0)
        - special.gammaln(total_counts - numerator_counts + 1.0)
        + special.betaln(numerator_counts + alpha, total_counts - numerator_counts + beta)
        - special.betaln(alpha, beta)
    )

    coefficient_penalty = RIDGE_PENALTY * float(np.sum(np.square(parameters[:2])))

    return -float(np.sum(log_likelihood)) + coefficient_penalty


def dirichlet_multinomial_nll(
    parameters: np.ndarray,
    counts: np.ndarray,
    condition_indicator: np.ndarray,
    n_fractions: int,
    free: bool = True,
    penalty_strength: float = RIDGE_PENALTY
) -> float:
    if n_fractions < 2:
        raise ValueError('The Dirichlet-multinomial model requires at least two fractions.')

    if counts.ndim != 2:
        raise ValueError('Dirichlet-multinomial counts must be a two-dimensional array.')

    if condition_indicator.ndim != 1:
        raise ValueError('The condition indicator must be a one-dimensional array.')

    if counts.shape[1] != n_fractions:
        raise ValueError('The number of count columns must match the number of fractions.')

    if counts.shape[0] != condition_indicator.shape[0]:
        raise ValueError('Counts and condition indicators must contain the same number of observations.')

    if not (
        np.all(np.isfinite(parameters))
        and np.all(np.isfinite(counts))
        and np.all(np.isfinite(condition_indicator))
    ):
        raise ValueError('The Dirichlet-multinomial inputs must contain only finite values.')

    if np.any(counts < 0.0):
        raise ValueError('Dirichlet-multinomial counts must be nonnegative.')

    if not np.isfinite(penalty_strength) or penalty_strength < 0.0:
        raise ValueError('The penalty strength must be a finite nonnegative value.')

    n_coefficients = n_fractions - 1
    expected_parameters = 2 * n_coefficients + 1 if free else n_coefficients + 1

    if parameters.shape != (expected_parameters,):
        raise ValueError(f'The Dirichlet-multinomial model requires {expected_parameters} parameters when free is {free}.')

    intercepts = parameters[:n_coefficients]

    if free:
        condition_effects = parameters[n_coefficients:2 * n_coefficients]
    else:
        condition_effects = np.zeros(n_coefficients, dtype = float)

    log_concentration = parameters[-1]

    # The first fraction is the reference category and has a fixed linear predictor of zero.
    linear_predictors = np.column_stack(
        [
            np.zeros(len(condition_indicator), dtype = float),
            intercepts[None, :] + condition_indicator[:, None] * condition_effects[None, :]
        ]
    )

    fraction_probabilities = special.softmax(linear_predictors, axis = 1)
    concentration = np.exp(log_concentration)
    alpha = fraction_probabilities * concentration
    total_counts = counts.sum(axis = 1)

    log_likelihood = (
        special.gammaln(total_counts + 1.0)
        - special.gammaln(counts + 1.0).sum(axis = 1)
        + special.gammaln(concentration)
        - special.gammaln(total_counts + concentration)
        + (special.gammaln(counts + alpha) - special.gammaln(alpha)).sum(axis = 1)
    )

    coefficient_penalty = penalty_strength * float(np.sum(np.square(parameters[:-1])))

    return -float(np.sum(log_likelihood)) + coefficient_penalty
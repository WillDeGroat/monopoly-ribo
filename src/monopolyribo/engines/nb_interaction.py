# Imports ------------------------------------------
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from ..contrasts import FractionContrast
from ..design import design_matrix
from ..exceptions import NonEstimableContrastError
from ..normalization import median_ratio_size_factors
from ..statistics import adjust_pvalues
from .base import EngineFit
# --------------------------------------------------


ESTIMABILITY_TOLERANCE = 1e-8


class NBInteractionEngine:
    name: str = 'nb_interaction'

    def fit(self, dataset: Any) -> EngineFit:
        counts = dataset.filtered_counts.astype(float)

        if counts.empty:
            raise ValueError('The interaction model requires a nonempty filtered count matrix.')

        if not np.all(np.isfinite(counts.to_numpy())):
            raise ValueError('The filtered count matrix must contain only finite values.')

        if (counts < 0.0).any().any():
            raise ValueError('The filtered count matrix must contain only nonnegative values.')

        if not counts.index.equals(dataset.metadata.index):
            raise ValueError('The filtered count matrix and metadata must have matching sample indices.')

        if not dataset.fraction_order:
            raise ValueError('The interaction model requires at least one ordered fraction.')

        reference_fraction = dataset.fraction_order[0]

        if dataset.abundance_fraction is not None and dataset.abundance_fraction != reference_fraction:
            raise ValueError(
                'The abundance fraction must be the reference level in the ordered fraction list.'
            )

        size_factors = median_ratio_size_factors(counts).reindex(counts.index)

        if size_factors.isna().any():
            raise ValueError('Size factors are missing for one or more samples.')

        if not np.all(np.isfinite(size_factors.to_numpy())) or (size_factors <= 0.0).any():
            raise ValueError('Size factors must contain only finite positive values.')

        normalized_counts = counts.div(size_factors, axis = 0)
        response = np.log2(normalized_counts + 1.0)

        if not np.all(np.isfinite(response.to_numpy())):
            raise ValueError('The normalized response matrix must contain only finite values.')

        design = design_matrix(
            dataset.metadata,
            dataset.subject,
            dataset.condition,
            dataset.fraction,
            dataset.fraction_order,
            dataset.covariates
        )

        if not design.index.equals(response.index):
            raise ValueError('The design matrix and count matrix must have matching sample indices.')

        design_array = design.to_numpy(dtype = float)
        response_array = response.to_numpy(dtype = float)

        if not np.all(np.isfinite(design_array)):
            raise ValueError('The design matrix must contain only finite values.')

        coefficients, _, design_rank, _ = np.linalg.lstsq(
            design_array,
            response_array,
            rcond = None
        )

        residual_degrees_freedom = design_array.shape[0] - design_rank

        if residual_degrees_freedom <= 0:
            raise ValueError('The interaction model requires positive residual degrees of freedom.')

        fitted_values = design_array @ coefficients
        residuals = response_array - fitted_values
        residual_variance = np.sum(np.square(residuals), axis = 0) / residual_degrees_freedom

        coefficient_table = pd.DataFrame(
            coefficients.T,
            index = response.columns,
            columns = design.columns
        )

        covariance_base = np.linalg.pinv(design_array.T @ design_array)
        estimability_projection = np.linalg.pinv(design_array) @ design_array
        design_rank_deficient = design_rank < design_array.shape[1]

        valid_residual_variance = np.isfinite(residual_variance) & (residual_variance >= 0.0)

        diagnostics = pd.DataFrame(
            {
                'feature_id': response.columns,
                'engine': self.name,
                'converged': valid_residual_variance,
                'warning_code': np.where(
                    valid_residual_variance,
                    '',
                    'invalid_residual_variance'
                )
            },
            index = response.columns
        )

        return EngineFit(
            name = self.name,
            models = {
                'coef': coefficient_table,
                'vcov_base': covariance_base,
                'sigma2': pd.Series(residual_variance, index = response.columns),
                'design_columns': list(design.columns),
                'design_rank': int(design_rank),
                'df_residual': int(residual_degrees_freedom),
                'estimability_projection': estimability_projection,
                'size_factor': size_factors,
                'base_mean': normalized_counts.mean(axis = 0),
                'reference_fraction': reference_fraction
            },
            diagnostics = diagnostics,
            metadata = {
                'design': 'log2(count/size_factor+1) ~ subject + condition*fraction + covariates',
                'model_family': 'Gaussian linear model on log-normalized counts',
                'design_rank_deficient': bool(design_rank_deficient)
            }
        )


def _condition_column(columns: list[str], level: str) -> str | None:
    column = f'condition_{level}'
    return column if column in columns else None


def _interaction_column(columns: list[str], level: str, fraction: str) -> str | None:
    prefix = f'condition_{level}:'
    suffix = f':fraction_{fraction}'

    return next(
        (
            column
            for column in columns
            if column.startswith(prefix) and column.endswith(suffix)
        ),
        None
    )


def _add_interaction_term(
    contrast_vector: np.ndarray,
    column_indices: dict[str, int],
    columns: list[str],
    contrast: FractionContrast,
    fraction: str,
    coefficient: float
) -> None:
    column = _interaction_column(columns, contrast.case, fraction)
    signed_coefficient = coefficient

    if column is None:
        column = _interaction_column(columns, contrast.control, fraction)
        signed_coefficient = -coefficient

    if column is None:
        raise NonEstimableContrastError(
            f'Contrast {contrast.label!r} is not estimable because no interaction '
            f'coefficient was found for fraction {fraction!r}.'
        )

    contrast_vector[column_indices[column]] += signed_coefficient


def _abundance_contrast_vector(
    fit: EngineFit,
    dataset: Any,
    columns: list[str],
    column_indices: dict[str, int],
    contrast: FractionContrast
) -> np.ndarray:
    reference_fraction = fit.models['reference_fraction']

    if dataset.abundance_fraction is None:
        raise NonEstimableContrastError(
            f'Contrast {contrast.label!r} requires an abundance fraction.'
        )

    if dataset.abundance_fraction != reference_fraction:
        raise NonEstimableContrastError(
            f'Contrast {contrast.label!r} is not estimable as an abundance contrast because '
            f'the abundance fraction {dataset.abundance_fraction!r} is not the reference '
            f'fraction {reference_fraction!r}.'
        )

    contrast_vector = np.zeros(len(columns), dtype = float)
    condition_column = _condition_column(columns, contrast.case)
    sign = 1.0

    if condition_column is None:
        condition_column = _condition_column(columns, contrast.control)
        sign = -1.0

    if condition_column is None:
        raise NonEstimableContrastError(
            f'Contrast {contrast.label!r} is not estimable because no condition '
            'coefficient was found for the case or control level.'
        )

    contrast_vector[column_indices[condition_column]] = sign

    return contrast_vector


def _redistribution_contrast_vector(
    columns: list[str],
    column_indices: dict[str, int],
    contrast: FractionContrast
) -> np.ndarray:
    if contrast.numerator is None or contrast.denominator is None:
        raise NonEstimableContrastError(
            f'Contrast {contrast.label!r} requires numerator and denominator fractions.'
        )

    contrast_vector = np.zeros(len(columns), dtype = float)

    _add_interaction_term(
        contrast_vector,
        column_indices,
        columns,
        contrast,
        contrast.numerator,
        1.0
    )

    _add_interaction_term(
        contrast_vector,
        column_indices,
        columns,
        contrast,
        contrast.denominator,
        -1.0
    )

    return contrast_vector


def _validate_estimability(fit: EngineFit, contrast: FractionContrast, contrast_vector: np.ndarray) -> None:
    projection = fit.models['estimability_projection']
    projected_vector = projection @ contrast_vector

    if not np.allclose(
        projected_vector,
        contrast_vector,
        rtol = ESTIMABILITY_TOLERANCE,
        atol = ESTIMABILITY_TOLERANCE
    ):
        raise NonEstimableContrastError(
            f'Contrast {contrast.label!r} is not estimable under the fitted design matrix.'
        )


def nb_contrast(fit: EngineFit, dataset: Any, contrast: FractionContrast) -> pd.DataFrame:
    columns = fit.models['design_columns']
    column_indices = {column: index for index, column in enumerate(columns)}

    if contrast.kind == 'abundance':
        contrast_vector = _abundance_contrast_vector(
            fit,
            dataset,
            columns,
            column_indices,
            contrast
        )
        effect_scale = 'log2_fold_change'
    elif contrast.kind in {'redistribution', 'fraction_vs_input'}:
        contrast_vector = _redistribution_contrast_vector(
            columns,
            column_indices,
            contrast
        )
        effect_scale = 'log2_redistribution_ratio'
    else:
        raise NonEstimableContrastError(
            f'Contrast kind {contrast.kind!r} is not supported by the interaction model.'
        )

    _validate_estimability(fit, contrast, contrast_vector)

    coefficients = fit.models['coef']
    residual_variance = fit.models['sigma2'].reindex(coefficients.index)
    residual_degrees_freedom = fit.models['df_residual']

    variance_factor = float(contrast_vector @ fit.models['vcov_base'] @ contrast_vector)

    if not np.isfinite(variance_factor) or variance_factor <= 0.0:
        raise NonEstimableContrastError(
            f'Contrast {contrast.label!r} has a nonpositive or nonfinite variance.'
        )

    effects = coefficients.to_numpy() @ contrast_vector
    contrast_variance = residual_variance.to_numpy() * variance_factor

    valid_effect = np.isfinite(effects)
    valid_variance = np.isfinite(contrast_variance) & (contrast_variance > 0.0)
    valid_statistics = valid_effect & valid_variance

    standard_errors = np.full(len(coefficients), np.nan, dtype = float)
    statistics = np.full(len(coefficients), np.nan, dtype = float)
    pvalues = np.full(len(coefficients), np.nan, dtype = float)

    standard_errors[valid_statistics] = np.sqrt(contrast_variance[valid_statistics])
    statistics[valid_statistics] = effects[valid_statistics] / standard_errors[valid_statistics]
    pvalues[valid_statistics] = 2.0 * stats.t.sf(np.abs(statistics[valid_statistics]), df = residual_degrees_freedom)

    warning_codes = np.full(len(coefficients), '', dtype = object)
    warning_codes[~valid_variance] = 'invalid_contrast_variance'
    warning_codes[valid_variance & ~valid_effect] = 'invalid_contrast_effect'

    base_mean = fit.models['base_mean'].reindex(coefficients.index)
    informative_counts = (
        dataset.filtered_counts.reindex(columns = coefficients.index)
        .gt(0.0)
        .sum(axis = 0)
        .to_numpy()
    )

    results = pd.DataFrame(
        {
            'feature_id': coefficients.index,
            'contrast': contrast.label,
            'engine': NBInteractionEngine.name,
            'effect': effects,
            'effect_scale': effect_scale,
            'standard_error': standard_errors,
            'statistic': statistics,
            'pvalue': pvalues,
            'base_mean': base_mean,
            'n_samples': dataset.filtered_counts.shape[0],
            'n_subjects': dataset.metadata[dataset.subject].nunique(),
            'n_informative': informative_counts,
            'converged': valid_statistics,
            'warning_code': warning_codes
        },
        index = coefficients.index
    )

    results['padj'] = adjust_pvalues(results['pvalue'])

    return results
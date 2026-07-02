# Imports ------------------------------------------
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from ..contrasts import FractionContrast
from ..design import abundance_design_matrix, redistribution_design_matrix
from ..exceptions import NonEstimableContrastError
from ..statistics import adjust_pvalues
from .base import EngineFit
# --------------------------------------------------


ESTIMABILITY_TOLERANCE = 1e-8
LEVERAGE_TOLERANCE = 1e-10


class NBInteractionEngine:
    name: str = 'nb_interaction'

    def fit(self, dataset: Any) -> EngineFit:
        counts = dataset.filtered_counts.astype(float)

        _validate_counts(counts, dataset.metadata)

        exposure = dataset.effective_exposure.reindex(counts.index).astype(float)

        if exposure.isna().any():
            raise ValueError(
                'Effective exposures are missing for one or more samples.'
            )

        if (
            not np.all(np.isfinite(exposure.to_numpy()))
            or (exposure <= 0.0).any()
        ):
            raise ValueError(
                'Effective exposures must contain only finite positive values.'
            )

        normalized_counts = counts.div(exposure, axis = 0)
        response = np.log2(normalized_counts + 1.0)

        if not np.all(np.isfinite(response.to_numpy())):
            raise ValueError(
                'The normalized response matrix must contain only finite values.'
            )

        models: dict[str, Any] = {
            'case': dataset.case,
            'control': dataset.control,
            'reference_fraction': dataset.fraction_order[0]
        }

        if dataset.abundance_fraction is not None:
            abundance_mask = (
                dataset.metadata[dataset.fraction].astype(str)
                == dataset.abundance_fraction
            )

            abundance_metadata = dataset.metadata.loc[abundance_mask].copy()
            abundance_response = response.loc[abundance_mask].copy()
            abundance_counts = counts.loc[abundance_mask].copy()
            abundance_normalized_counts = normalized_counts.loc[
                abundance_mask
            ].copy()

            abundance_design = abundance_design_matrix(
                abundance_metadata,
                dataset.subject,
                dataset.condition,
                dataset.case,
                dataset.control,
                dataset.covariates
            )

            models['abundance'] = _fit_linear_model(
                abundance_response,
                abundance_counts,
                abundance_normalized_counts,
                abundance_metadata,
                abundance_design,
                dataset.subject,
                covariance_type = 'hc3'
            )

        redistribution_design = redistribution_design_matrix(
            dataset.metadata,
            dataset.subject,
            dataset.condition,
            dataset.case,
            dataset.control,
            dataset.fraction,
            dataset.fraction_order,
            dataset.covariates
        )

        models['redistribution'] = _fit_linear_model(
            response,
            counts,
            normalized_counts,
            dataset.metadata,
            redistribution_design,
            dataset.subject,
            covariance_type = 'cluster'
        )

        diagnostics = pd.DataFrame(
            {
                'feature_id': response.columns,
                'engine': self.name,
                'converged': True,
                'warning_code': ''
            },
            index = response.columns
        )

        return EngineFit(
            name = self.name,
            models = models,
            diagnostics = diagnostics,
            metadata = {
                'abundance_design': (
                    'abundance_fraction ~ condition + covariates'
                ),
                'abundance_covariance': (
                    'HC3 heteroskedasticity-robust covariance'
                ),
                'redistribution_design': (
                    'response ~ subject + fraction + condition:fraction '
                    '+ covariate:fraction'
                ),
                'redistribution_covariance': (
                    'CR1 subject-clustered sandwich covariance'
                ),
                'model_family': (
                    'Gaussian linear models on log2 '
                    'exposure-normalized counts'
                ),
                'design_rank_deficient': False
            }
        )


def nb_contrast(
    fit: EngineFit,
    dataset: Any,
    contrast: FractionContrast
) -> pd.DataFrame:
    if contrast.kind == 'abundance':
        model_name = 'abundance'
        effect_scale = 'log2_fold_change'

        if model_name not in fit.models:
            raise NonEstimableContrastError(
                f'Contrast {contrast.label!r} requires a configured '
                'abundance fraction.'
            )

        if contrast.fraction != dataset.abundance_fraction:
            raise NonEstimableContrastError(
                f'Contrast {contrast.label!r} is not an abundance contrast '
                'for the configured abundance fraction '
                f'{dataset.abundance_fraction!r}.'
            )
    elif contrast.kind in {'redistribution', 'fraction_vs_input'}:
        model_name = 'redistribution'
        effect_scale = 'log2_redistribution_ratio'
    else:
        raise NonEstimableContrastError(
            f'Contrast kind {contrast.kind!r} is not supported by the '
            'interaction model.'
        )

    model = fit.models[model_name]
    columns = model['design_columns']
    column_indices = {
        column: index
        for index, column in enumerate(columns)
    }
    orientation = _contrast_orientation(fit, contrast)

    if contrast.kind == 'abundance':
        contrast_vector = _abundance_contrast_vector(
            fit,
            columns,
            column_indices,
            orientation,
            contrast
        )
    else:
        contrast_vector = _redistribution_contrast_vector(
            fit,
            dataset,
            columns,
            column_indices,
            orientation,
            contrast
        )

    _validate_estimability(
        model,
        contrast,
        contrast_vector
    )

    coefficients = model['coef']
    effects = coefficients.to_numpy() @ contrast_vector
    contrast_variance = _contrast_variance(
        model,
        contrast_vector
    )

    valid_effect = np.isfinite(effects)
    valid_variance = (
        np.isfinite(contrast_variance)
        & (contrast_variance > 0.0)
    )
    valid_statistics = valid_effect & valid_variance

    standard_errors = np.full(
        len(coefficients),
        np.nan,
        dtype = float
    )
    statistics = np.full(
        len(coefficients),
        np.nan,
        dtype = float
    )
    pvalues = np.full(
        len(coefficients),
        np.nan,
        dtype = float
    )

    standard_errors[valid_statistics] = np.sqrt(
        contrast_variance[valid_statistics]
    )
    statistics[valid_statistics] = (
        effects[valid_statistics]
        / standard_errors[valid_statistics]
    )
    pvalues[valid_statistics] = 2.0 * stats.t.sf(
        np.abs(statistics[valid_statistics]),
        df = model['test_df']
    )

    warning_codes = np.full(
        len(coefficients),
        '',
        dtype = object
    )
    warning_codes[~valid_variance] = 'invalid_contrast_variance'
    warning_codes[
        valid_variance & ~valid_effect
    ] = 'invalid_contrast_effect'

    base_mean = model['base_mean'].reindex(
        coefficients.index
    )
    informative_counts = (
        model['counts']
        .reindex(columns = coefficients.index)
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
            'n_samples': model['n_samples'],
            'n_subjects': model['n_subjects'],
            'n_informative': informative_counts,
            'converged': valid_statistics,
            'warning_code': warning_codes
        },
        index = coefficients.index
    )
    results['padj'] = adjust_pvalues(
        results['pvalue']
    )

    return results


def _fit_linear_model(
    response: pd.DataFrame,
    counts: pd.DataFrame,
    normalized_counts: pd.DataFrame,
    metadata: pd.DataFrame,
    design: pd.DataFrame,
    subject: str,
    covariance_type: str
) -> dict[str, Any]:
    if not design.index.equals(response.index):
        raise ValueError(
            'The design matrix and response matrix must have matching '
            'sample indices.'
        )

    if not counts.index.equals(response.index):
        raise ValueError(
            'The count matrix and response matrix must have matching '
            'sample indices.'
        )

    if not normalized_counts.index.equals(response.index):
        raise ValueError(
            'The normalized count matrix and response matrix must have '
            'matching sample indices.'
        )

    if not metadata.index.equals(response.index):
        raise ValueError(
            'The metadata and response matrix must have matching sample '
            'indices.'
        )

    design_array = design.to_numpy(dtype = float)
    response_array = response.to_numpy(dtype = float)

    coefficients, _, design_rank, _ = np.linalg.lstsq(
        design_array,
        response_array,
        rcond = None
    )

    if design_rank < design_array.shape[1]:
        raise ValueError(
            'The fitted design matrix must have full column rank.'
        )

    residual_degrees_freedom = (
        design_array.shape[0] - design_rank
    )

    if residual_degrees_freedom <= 0:
        raise ValueError(
            'The fitted model requires positive residual degrees of freedom.'
        )

    fitted_values = design_array @ coefficients
    residuals = response_array - fitted_values
    bread = np.linalg.inv(
        design_array.T @ design_array
    )
    estimability_projection = (
        np.linalg.pinv(design_array) @ design_array
    )
    leverage = np.sum(
        (design_array @ bread) * design_array,
        axis = 1
    )

    model: dict[str, Any] = {
        'coef': pd.DataFrame(
            coefficients.T,
            index = response.columns,
            columns = design.columns
        ),
        'design_array': design_array,
        'design_columns': list(design.columns),
        'bread': bread,
        'residuals': residuals,
        'estimability_projection': estimability_projection,
        'df_residual': int(residual_degrees_freedom),
        'base_mean': normalized_counts.mean(axis = 0),
        'counts': counts.copy(),
        'n_samples': int(len(metadata)),
        'n_subjects': int(
            metadata[subject].astype(str).nunique()
        ),
        'covariance_type': covariance_type
    }

    if covariance_type == 'hc3':
        if np.any(
            leverage >= 1.0 - LEVERAGE_TOLERANCE
        ):
            raise ValueError(
                'The abundance model has leverage values too close to one '
                'for HC3 inference.'
            )

        model['leverage'] = leverage
        model['test_df'] = int(
            residual_degrees_freedom
        )

        return model

    if covariance_type != 'cluster':
        raise ValueError(
            f'Unsupported covariance type {covariance_type!r}.'
        )

    subject_values = metadata[subject].astype(str)
    cluster_codes, cluster_levels = pd.factorize(
        subject_values,
        sort = True
    )
    n_clusters = len(cluster_levels)

    if n_clusters < 4:
        raise ValueError(
            'Subject-clustered inference requires at least four subjects.'
        )

    correction = (
        n_clusters
        / (n_clusters - 1.0)
        * (design_array.shape[0] - 1.0)
        / residual_degrees_freedom
    )

    model['cluster_codes'] = cluster_codes
    model['cluster_correction'] = float(correction)
    model['test_df'] = int(n_clusters - 1)

    return model


def _contrast_orientation(
    fit: EngineFit,
    contrast: FractionContrast
) -> float:
    fitted_case = fit.models['case']
    fitted_control = fit.models['control']

    if (
        contrast.case == fitted_case
        and contrast.control == fitted_control
    ):
        return 1.0

    if (
        contrast.case == fitted_control
        and contrast.control == fitted_case
    ):
        return -1.0

    raise NonEstimableContrastError(
        f'Contrast {contrast.label!r} uses condition levels that do not '
        f'match the fitted case {fitted_case!r} and control '
        f'{fitted_control!r}.'
    )


def _abundance_contrast_vector(
    fit: EngineFit,
    columns: list[str],
    column_indices: dict[str, int],
    orientation: float,
    contrast: FractionContrast
) -> np.ndarray:
    condition_column = (
        f'condition_{fit.models["case"]}'
    )

    if condition_column not in column_indices:
        raise NonEstimableContrastError(
            f'Contrast {contrast.label!r} is not estimable because '
            f'condition column {condition_column!r} is absent from the '
            'abundance model.'
        )

    contrast_vector = np.zeros(
        len(columns),
        dtype = float
    )
    contrast_vector[
        column_indices[condition_column]
    ] = orientation

    return contrast_vector


def _redistribution_contrast_vector(
    fit: EngineFit,
    dataset: Any,
    columns: list[str],
    column_indices: dict[str, int],
    orientation: float,
    contrast: FractionContrast
) -> np.ndarray:
    if (
        contrast.numerator is None
        or contrast.denominator is None
    ):
        raise NonEstimableContrastError(
            f'Contrast {contrast.label!r} requires numerator and '
            'denominator fractions.'
        )

    known_fractions = set(
        dataset.fraction_order
    )

    for fraction in [
        contrast.numerator,
        contrast.denominator
    ]:
        if fraction not in known_fractions:
            raise NonEstimableContrastError(
                f'Contrast {contrast.label!r} refers to unknown fraction '
                f'{fraction!r}.'
            )

    contrast_vector = np.zeros(
        len(columns),
        dtype = float
    )
    reference_fraction = fit.models[
        'reference_fraction'
    ]
    condition_column = (
        f'condition_{fit.models["case"]}'
    )

    fraction_coefficients = [
        (contrast.numerator, 1.0),
        (contrast.denominator, -1.0)
    ]

    for fraction, coefficient in fraction_coefficients:
        if fraction == reference_fraction:
            continue

        interaction_column = (
            f'{condition_column}:fraction_{fraction}'
        )

        if interaction_column not in column_indices:
            raise NonEstimableContrastError(
                f'Contrast {contrast.label!r} is not estimable because '
                f'interaction column {interaction_column!r} is absent '
                'from the redistribution model.'
            )

        contrast_vector[
            column_indices[interaction_column]
        ] += orientation * coefficient

    if not np.any(contrast_vector):
        raise NonEstimableContrastError(
            f'Contrast {contrast.label!r} has an empty redistribution '
            'contrast vector.'
        )

    return contrast_vector


def _validate_estimability(
    model: dict[str, Any],
    contrast: FractionContrast,
    contrast_vector: np.ndarray
) -> None:
    projection = model[
        'estimability_projection'
    ]
    projected_vector = projection @ contrast_vector

    if not np.allclose(
        projected_vector,
        contrast_vector,
        rtol = ESTIMABILITY_TOLERANCE,
        atol = ESTIMABILITY_TOLERANCE
    ):
        raise NonEstimableContrastError(
            f'Contrast {contrast.label!r} is not estimable under the '
            'fitted design matrix.'
        )


def _contrast_variance(
    model: dict[str, Any],
    contrast_vector: np.ndarray
) -> np.ndarray:
    design_array = model['design_array']
    bread = model['bread']
    residuals = model['residuals']

    influence_weights = (
        design_array @ bread @ contrast_vector
    )

    if model['covariance_type'] == 'hc3':
        leverage_adjustment = (
            1.0 - model['leverage']
        )
        adjusted_residuals = (
            residuals
            / leverage_adjustment[:, np.newaxis]
        )

        return np.sum(
            np.square(
                influence_weights[:, np.newaxis]
                * adjusted_residuals
            ),
            axis = 0
        )

    cluster_scores: list[np.ndarray] = []

    for cluster_code in np.unique(
        model['cluster_codes']
    ):
        cluster_mask = (
            model['cluster_codes'] == cluster_code
        )
        cluster_score = np.sum(
            influence_weights[
                cluster_mask,
                np.newaxis
            ]
            * residuals[cluster_mask],
            axis = 0
        )
        cluster_scores.append(cluster_score)

    return model[
        'cluster_correction'
    ] * np.sum(
        np.square(
            np.vstack(cluster_scores)
        ),
        axis = 0
    )


def _validate_counts(
    counts: pd.DataFrame,
    metadata: pd.DataFrame
) -> None:
    if counts.empty:
        raise ValueError(
            'The interaction model requires a nonempty filtered count '
            'matrix.'
        )

    if not np.all(
        np.isfinite(counts.to_numpy())
    ):
        raise ValueError(
            'The filtered count matrix must contain only finite values.'
        )

    if (counts < 0.0).any().any():
        raise ValueError(
            'The filtered count matrix must contain only nonnegative values.'
        )

    if not counts.index.equals(metadata.index):
        raise ValueError(
            'The filtered count matrix and metadata must have matching '
            'sample indices.'
        )

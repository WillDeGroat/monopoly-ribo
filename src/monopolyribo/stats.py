# Imports ------------------------------------------
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .contrasts import FractionContrast
from .engines import ENGINE_CLASSES
from .engines.nb_interaction import nb_contrast
from .exceptions import MonoPolyInputError
from .validation import validate_conditions
# --------------------------------------------------


SUPPORTED_STATS_ENGINES = {
    'beta_binomial',
    'dirichlet_multinomial',
    'joint_latent',
    'joint_latent_mle',
    'nb_interaction'
}


class MonoPolyStats:
    def __init__(
        self,
        dataset: Any,
        contrast: FractionContrast | None = None,
        case: str | None = None,
        control: str | None = None,
        numerator_fraction: str | None = None,
        denominator_fraction: str | None = None,
        engine: str = 'nb_interaction',
        alpha: float = 0.05
    ) -> None:
        _validate_engine(engine)
        _validate_alpha(alpha)

        if contrast is None:
            contrast = _build_redistribution_contrast(
                case,
                control,
                numerator_fraction,
                denominator_fraction
            )
        elif any(
            value is not None
            for value in [
                case,
                control,
                numerator_fraction,
                denominator_fraction
            ]
        ):
            raise MonoPolyInputError(
                'Provide either a FractionContrast or individual contrast arguments, not both.'
            )

        validate_conditions(
            dataset.metadata,
            dataset.condition,
            contrast.case,
            contrast.control
        )

        self.dataset = dataset
        self.contrast = contrast
        self.engine = engine
        self.alpha = alpha
        self.results_df: pd.DataFrame | None = None

    def summary(self) -> pd.DataFrame:
        fit = self._get_fit()

        if self.engine == 'nb_interaction':
            results = nb_contrast(
                fit,
                self.dataset,
                self.contrast
            )
        elif self.engine == 'beta_binomial':
            results = self._beta_binomial_results(fit)
        elif self.engine == 'dirichlet_multinomial':
            results = self._dirichlet_multinomial_results(fit)
        elif self.engine in {'joint_latent_mle', 'joint_latent'}:
            results = self._joint_latent_results(fit)
        else:
            raise MonoPolyInputError(
                f'Statistical summaries are not supported for engine {self.engine!r}.'
            )

        results = results.copy()
        results['significant'] = (
            results['padj'].notna()
            & np.isfinite(results['padj'])
            & (results['padj'] <= self.alpha)
        )

        self.results_df = results

        return results

    def _get_fit(self) -> Any:
        fit_name = 'joint_latent_mle' if self.engine == 'joint_latent' else self.engine
        fit = self.dataset.fits.get(fit_name)

        if fit is not None:
            return fit

        if fit_name not in ENGINE_CLASSES:
            raise MonoPolyInputError(
                f'Unknown statistical engine {fit_name!r}.'
            )

        engine_class = ENGINE_CLASSES[fit_name]
        fit = engine_class().fit(self.dataset)
        self.dataset.fits[fit_name] = fit

        return fit

    def _beta_binomial_results(self, fit: Any) -> pd.DataFrame:
        if self.contrast.kind not in {'redistribution', 'fraction_vs_input'}:
            raise MonoPolyInputError(
                'The beta-binomial engine supports only redistribution and fraction-versus-input contrasts.'
            )

        allocation_fractions = self.dataset.allocation_fractions or []

        if len(allocation_fractions) != 2:
            raise MonoPolyInputError(
                'The beta-binomial engine requires exactly two allocation fractions.'
            )

        denominator_fraction = allocation_fractions[0]
        numerator_fraction = allocation_fractions[1]

        _validate_condition_orientation(
            self.dataset,
            self.contrast
        )

        if (
            self.contrast.numerator != numerator_fraction
            or self.contrast.denominator != denominator_fraction
        ):
            raise MonoPolyInputError(
                f'The beta-binomial engine was fitted for '
                f'{numerator_fraction!r} versus {denominator_fraction!r}, '
                f'but contrast {self.contrast.label!r} was requested.'
            )

        return _result_table(fit, 'allocation')

    def _dirichlet_multinomial_results(self, fit: Any) -> pd.DataFrame:
        if self.contrast.kind != 'omnibus_interaction':
            raise MonoPolyInputError(
                'The Dirichlet-multinomial engine supports only omnibus interaction contrasts.'
            )

        _validate_condition_orientation(
            self.dataset,
            self.contrast
        )

        return _result_table(fit, 'allocation')

    def _joint_latent_results(self, fit: Any) -> pd.DataFrame:
        results = _result_table(fit, 'joint')
        matching = results['contrast'] == self.contrast.label

        if not matching.any():
            raise MonoPolyInputError(
                f'The joint latent fit does not contain contrast {self.contrast.label!r}.'
            )

        return results.loc[matching].copy()


def _build_redistribution_contrast(
    case: str | None,
    control: str | None,
    numerator_fraction: str | None,
    denominator_fraction: str | None
) -> FractionContrast:
    missing_arguments = [
        name
        for name, value in {
            'case': case,
            'control': control,
            'numerator_fraction': numerator_fraction,
            'denominator_fraction': denominator_fraction
        }.items()
        if value is None
    ]

    if missing_arguments:
        raise MonoPolyInputError(
            f'Missing contrast arguments: {missing_arguments}. '
            'Provide a FractionContrast or all individual contrast arguments.'
        )

    return FractionContrast.redistribution(
        case,
        control,
        numerator_fraction,
        denominator_fraction
    )


def _validate_engine(engine: str) -> None:
    if not isinstance(engine, str) or not engine:
        raise MonoPolyInputError(
            'The statistical engine name must be a nonempty string.'
        )

    if engine not in SUPPORTED_STATS_ENGINES:
        raise MonoPolyInputError(
            f'Unsupported statistical engine {engine!r}. '
            f'Expected one of {sorted(SUPPORTED_STATS_ENGINES)}.'
        )


def _validate_alpha(alpha: float) -> None:
    if isinstance(alpha, bool) or not isinstance(alpha, int | float):
        raise MonoPolyInputError(
            'The significance threshold must be numeric.'
        )

    if not np.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise MonoPolyInputError(
            'The significance threshold must be between zero and one.'
        )


def _validate_condition_orientation(dataset: Any, contrast: FractionContrast) -> None:
    condition_levels = sorted(
        pd.unique(dataset.metadata[dataset.condition].astype(str))
    )

    if len(condition_levels) != 2:
        raise MonoPolyInputError(
            'The requested engine requires exactly two condition levels.'
        )

    expected_control = condition_levels[0]
    expected_case = condition_levels[1]

    if contrast.case != expected_case or contrast.control != expected_control:
        raise MonoPolyInputError(
            f'The fitted engine uses case {expected_case!r} and control '
            f'{expected_control!r}, but contrast {contrast.label!r} was requested.'
        )


def _result_table(fit: Any, result_name: str) -> pd.DataFrame:
    if result_name not in fit.results:
        raise MonoPolyInputError(
            f'The fitted engine does not contain a {result_name!r} result table.'
        )

    results = fit.results[result_name]

    if not isinstance(results, pd.DataFrame):
        raise MonoPolyInputError(
            f'The {result_name!r} engine result must be a pandas DataFrame.'
        )

    required_columns = {
        'feature_id',
        'contrast',
        'effect',
        'pvalue',
        'padj',
        'converged'
    }
    missing_columns = required_columns.difference(results.columns)

    if missing_columns:
        raise MonoPolyInputError(
            f'The {result_name!r} engine results are missing required columns: '
            f'{sorted(missing_columns)}.'
        )

    return results
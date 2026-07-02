# Imports ------------------------------------------
from __future__ import annotations

import pandas as pd

from .classification import integrate
from .contrasts import FractionContrast
from .dataset import MonoPolyDataSet
from .exceptions import MonoPolyInputError
from .results import MonoPolyResult
from .robustness import leave_one_subject_out
from .stats import MonoPolyStats
# --------------------------------------------------


RESULT_COLUMNS = [
    'feature_id',
    'contrast',
    'engine',
    'effect',
    'effect_scale',
    'standard_error',
    'statistic',
    'pvalue',
    'base_mean',
    'n_samples',
    'n_subjects',
    'n_informative',
    'converged',
    'warning_code',
    'padj',
    'significant'
]


def analyze(
    counts: pd.DataFrame,
    metadata: pd.DataFrame,
    subject: str,
    condition: str,
    case: str,
    control: str,
    fraction: str,
    fraction_order: list[str],
    covariates: list[str] | None = None,
    abundance_fraction: str | None = None,
    allocation_fractions: list[str] | None = None,
    engines: list[str] | None = None,
    fraction_measurement: str = 'relative_library',
    fraction_weights: str | pd.Series | dict[str, float] | None = None,
    min_count: int = 10,
    min_samples: int = 3,
    n_cpus: int | None = None,
    seed: int = 0,
    quiet: bool = False
) -> MonoPolyResult:
    configured_abundance_fraction = abundance_fraction or fraction_order[0]
    configured_allocation_fractions = allocation_fractions or [
        fraction_name
        for fraction_name in fraction_order
        if fraction_name != configured_abundance_fraction
    ]

    dataset = MonoPolyDataSet(
        counts = counts,
        metadata = metadata,
        subject = subject,
        condition = condition,
        case = case,
        control = control,
        fraction = fraction,
        fraction_order = fraction_order,
        covariates = covariates,
        abundance_fraction = configured_abundance_fraction,
        allocation_fractions = configured_allocation_fractions,
        engines = engines,
        fraction_measurement = fraction_measurement,
        fraction_weights = fraction_weights,
        min_count = min_count,
        min_samples = min_samples,
        n_cpus = n_cpus,
        seed = seed,
        quiet = quiet
    ).fit()

    abundance_results = _abundance_results(
        dataset,
        case,
        control,
        configured_abundance_fraction
    )
    redistribution_results = _redistribution_results(
        dataset,
        case,
        control,
        configured_allocation_fractions
    )
    fraction_effects = _fraction_effect_results(
        dataset,
        case,
        control,
        configured_abundance_fraction,
        configured_allocation_fractions
    )
    allocation_results = _allocation_results(dataset)
    joint_results = _joint_results(dataset)

    integrated_results = integrate(
        abundance_results,
        redistribution_results,
        allocation_results,
        joint_results
    )
    classification_results = integrated_results[
        ['regulatory_class', 'classification_reason']
    ].copy()
    stability_results = _stability_results(
        dataset,
        case,
        control,
        configured_allocation_fractions
    )

    return MonoPolyResult(
        abundance = abundance_results,
        fraction_effects = fraction_effects,
        redistribution = redistribution_results,
        allocation = allocation_results,
        joint = joint_results,
        integrated = integrated_results,
        classification = classification_results,
        stability = stability_results,
        diagnostics = dataset.diagnostics.copy(),
        metadata = {
            'case': case,
            'control': control,
            'abundance_fraction': configured_abundance_fraction,
            'allocation_fractions': configured_allocation_fractions,
            'fraction_measurement': fraction_measurement,
            'engines': list(dataset.fits),
            'min_count': min_count,
            'min_samples': min_samples,
            'seed': seed
        }
    )


def _abundance_results(dataset: MonoPolyDataSet, case: str, control: str, abundance_fraction: str) -> pd.DataFrame:
    if 'nb_interaction' not in dataset.fits:
        return _empty_results()

    contrast = FractionContrast.abundance(
        case,
        control,
        abundance_fraction
    )

    return MonoPolyStats(
        dataset,
        contrast = contrast,
        engine = 'nb_interaction'
    ).summary()


def _redistribution_results( dataset: MonoPolyDataSet, case: str, control: str, allocation_fractions: list[str]) -> pd.DataFrame:
    if 'nb_interaction' not in dataset.fits or len(allocation_fractions) < 2:
        return _empty_results()

    contrast = FractionContrast.redistribution(
        case,
        control,
        allocation_fractions[-1],
        allocation_fractions[0]
    )

    return MonoPolyStats(
        dataset,
        contrast = contrast,
        engine = 'nb_interaction'
    ).summary()


def _fraction_effect_results(
    dataset: MonoPolyDataSet,
    case: str,
    control: str,
    abundance_fraction: str,
    allocation_fractions: list[str]
) -> dict[str, pd.DataFrame]:
    if 'nb_interaction' not in dataset.fits:
        return {}

    fraction_effects: dict[str, pd.DataFrame] = {}

    for allocation_fraction in allocation_fractions:
        contrast = FractionContrast.fraction_vs_input(
            case,
            control,
            allocation_fraction,
            abundance_fraction
        )
        fraction_effects[allocation_fraction] = MonoPolyStats(
            dataset,
            contrast = contrast,
            engine = 'nb_interaction'
        ).summary()

    return fraction_effects


def _allocation_results(dataset: MonoPolyDataSet) -> pd.DataFrame:
    fitted_engines = [
        engine_name
        for engine_name in ['beta_binomial', 'dirichlet_multinomial']
        if engine_name in dataset.fits
    ]

    if not fitted_engines:
        return _empty_results()

    if len(fitted_engines) > 1:
        raise MonoPolyInputError(
            'A complete analysis cannot contain both beta-binomial and '
            'Dirichlet-multinomial allocation fits.'
        )

    allocation_fit = dataset.fits[fitted_engines[0]]

    if 'allocation' not in allocation_fit.results:
        raise MonoPolyInputError(
            f'Engine {fitted_engines[0]!r} does not contain allocation results.'
        )

    return allocation_fit.results['allocation'].copy()


def _joint_results(dataset: MonoPolyDataSet) -> pd.DataFrame:
    fitted_engines = [
        engine_name
        for engine_name in ['joint_latent_mle', 'joint_latent']
        if engine_name in dataset.fits
    ]

    if not fitted_engines:
        return _empty_results()

    if len(fitted_engines) > 1:
        raise MonoPolyInputError(
            'A complete analysis cannot contain both joint_latent and '
            'joint_latent_mle fits.'
        )

    joint_fit = dataset.fits[fitted_engines[0]]

    if 'joint' not in joint_fit.results:
        raise MonoPolyInputError(
            f'Engine {fitted_engines[0]!r} does not contain joint results.'
        )

    return joint_fit.results['joint'].copy()


def _stability_results(dataset: MonoPolyDataSet, case: str, control: str, allocation_fractions: list[str]) -> pd.DataFrame:
    if 'nb_interaction' not in dataset.fits or len(allocation_fractions) < 2:
        return pd.DataFrame()

    contrast = FractionContrast.redistribution(
        case,
        control,
        allocation_fractions[-1],
        allocation_fractions[0]
    )
    _, stability_results = leave_one_subject_out(
        dataset,
        contrast,
        engine = 'nb_interaction'
    )

    return stability_results


def _empty_results() -> pd.DataFrame:
    return pd.DataFrame(columns = RESULT_COLUMNS).set_index('feature_id')
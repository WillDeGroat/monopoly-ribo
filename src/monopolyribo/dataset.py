# Imports ------------------------------------------
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .engines import ENGINE_CLASSES
from .exceptions import MonoPolyInputError
from .filtering import filter_counts, filtering_table
from .normalization import median_ratio_size_factors, recovery_factors
from .validation import (
    align_metadata,
    validate_complete_subject_fractions,
    validate_counts,
    validate_metadata
)
# --------------------------------------------------


SUPPORTED_FRACTION_MEASUREMENTS = {
    'custom_exposure',
    'gradient_area',
    'relative_library',
    'rna_yield',
    'spike_in'
}


class MonoPolyDataSet:
    def __init__(
        self,
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
    ) -> None:
        validate_counts(counts)
        aligned_metadata, alignment_actions = align_metadata(counts, metadata)

        covariates = [] if covariates is None else list(covariates)
        fraction_order = list(fraction_order)
        allocation_fractions = (
            None
            if allocation_fractions is None
            else list(allocation_fractions)
        )

        validate_metadata(
            aligned_metadata,
            subject,
            condition,
            fraction,
            fraction_order,
            covariates
        )
        _validate_condition_configuration(
            aligned_metadata,
            subject,
            condition,
            case,
            control
        )
        _validate_fraction_configuration(
            fraction_order,
            abundance_fraction,
            allocation_fractions
        )
        _validate_fraction_measurement(fraction_measurement, fraction_weights)
        _validate_filtering_parameters(min_count, min_samples)
        _validate_runtime_parameters(n_cpus, seed, quiet)
        _validate_unique_subject_fractions(
            aligned_metadata,
            subject,
            fraction
        )

        validate_complete_subject_fractions(
            aligned_metadata,
            subject,
            fraction,
            fraction_order
        )

        self.counts = counts.astype(int).copy()
        self.metadata = aligned_metadata.copy()
        self.subject = subject
        self.condition = condition
        self.case = case
        self.control = control
        self.fraction = fraction
        self.fraction_order = fraction_order
        self.covariates = covariates
        self.abundance_fraction = abundance_fraction
        self.allocation_fractions = allocation_fractions
        self.fraction_measurement = fraction_measurement
        self.fraction_weights = fraction_weights
        self.min_count = min_count
        self.min_samples = min_samples
        self.n_cpus = n_cpus
        self.seed = seed
        self.quiet = quiet

        kept_features = filter_counts(self.counts, min_count, min_samples)
        self.filtered_counts = self.counts.loc[:, kept_features].copy()
        self.filters = filtering_table(
            self.counts,
            kept_features,
            min_count,
            min_samples
        )

        if self.filtered_counts.shape[1] == 0:
            raise MonoPolyInputError('Filtering removed all features from the count matrix.')

        self.size_factor = median_ratio_size_factors(
            self.filtered_counts
        ).reindex(self.counts.index)

        if self.size_factor.isna().any():
            raise MonoPolyInputError('Size factors are missing for one or more samples.')

        if not np.all(np.isfinite(self.size_factor.to_numpy())):
            raise MonoPolyInputError('Size factors must contain only finite values.')

        if (self.size_factor <= 0.0).any():
            raise MonoPolyInputError('Size factors must contain only positive values.')

        self.recovery_factor = recovery_factors(
            self.metadata,
            fraction_measurement,
            fraction_weights
        ).reindex(self.counts.index)

        if self.recovery_factor.isna().any():
            raise MonoPolyInputError('Recovery factors are missing for one or more samples.')

        if not np.all(np.isfinite(self.recovery_factor.to_numpy())):
            raise MonoPolyInputError('Recovery factors must contain only finite values.')

        if (self.recovery_factor <= 0.0).any():
            raise MonoPolyInputError('Recovery factors must contain only positive values.')

        self.effective_exposure = self.size_factor * self.recovery_factor

        if not np.all(np.isfinite(self.effective_exposure.to_numpy())):
            raise MonoPolyInputError('Effective exposures must contain only finite values.')

        if (self.effective_exposure <= 0.0).any():
            raise MonoPolyInputError('Effective exposures must contain only positive values.')

        configured_engines = self._default_engines() if engines is None else list(engines)
        self.engines = _validate_engines(configured_engines)

        self.fits: dict[str, Any] = {}

        self.input_diagnostics = pd.DataFrame({'action': alignment_actions})
        self.input_diagnostics['diagnostic_type'] = 'input_alignment'
        self.diagnostics = self.input_diagnostics.copy()

    def _default_engines(self) -> list[str]:
        engines = ['nb_interaction']

        if self.allocation_fractions:
            if len(self.allocation_fractions) == 2:
                engines.append('beta_binomial')
            else:
                engines.append('dirichlet_multinomial')

        if (
            self.abundance_fraction is not None
            and self.allocation_fractions is not None
            and len(self.allocation_fractions) >= 2
        ):
            engines.append('joint_latent_mle')

        return engines

    def fit(self) -> MonoPolyDataSet:
        self.fits = {}

        for engine_name in self.engines:
            engine_class = ENGINE_CLASSES[engine_name]
            self.fits[engine_name] = engine_class().fit(self)

        diagnostic_tables: list[pd.DataFrame] = []

        if not self.input_diagnostics.empty:
            diagnostic_tables.append(self.input_diagnostics)

        for engine_name, fit in self.fits.items():
            if fit.diagnostics.empty:
                continue

            engine_diagnostics = fit.diagnostics.copy()

            if 'engine' not in engine_diagnostics.columns:
                engine_diagnostics['engine'] = engine_name

            engine_diagnostics['diagnostic_type'] = 'engine_fit'
            diagnostic_tables.append(engine_diagnostics)

        if diagnostic_tables:
            self.diagnostics = pd.concat(
                diagnostic_tables,
                axis = 0,
                ignore_index = True,
                sort = False
            )
        else:
            self.diagnostics = pd.DataFrame()

        return self


def _validate_condition_configuration(metadata: pd.DataFrame, subject: str, condition: str, case: str, control: str) -> None:
    if not isinstance(case, str) or not case:
        raise MonoPolyInputError('The case condition must be a nonempty string.')

    if not isinstance(control, str) or not control:
        raise MonoPolyInputError('The control condition must be a nonempty string.')

    if case == control:
        raise MonoPolyInputError('The case and control conditions must be different.')

    observed_conditions = set(metadata[condition].astype(str))
    expected_conditions = {case, control}

    if observed_conditions != expected_conditions:
        raise MonoPolyInputError(
            f'The condition column must contain exactly case {case!r} and control '
            f'{control!r}; observed {sorted(observed_conditions)}.'
        )

    subject_conditions = (
        metadata.assign(
            __subject = metadata[subject].astype(str),
            __condition = metadata[condition].astype(str)
        )
        .drop_duplicates(['__subject', '__condition'])
    )
    conditions_per_subject = subject_conditions.groupby(
        '__subject',
        observed = True
    )['__condition'].nunique()

    if (conditions_per_subject != 1).any():
        invalid_subjects = conditions_per_subject[conditions_per_subject != 1].index.tolist()[:5]
        raise MonoPolyInputError(
            'Each subject must belong to exactly one condition. '
            f'Examples of invalid subjects: {invalid_subjects}.'
        )

    subjects_per_condition = subject_conditions.groupby(
        '__condition',
        observed = True
    )['__subject'].nunique()

    if (subjects_per_condition.reindex([control, case], fill_value = 0) < 2).any():
        raise MonoPolyInputError(
            'At least two subjects are required in both the case and control conditions.'
        )


def _validate_unique_subject_fractions(metadata: pd.DataFrame, subject: str, fraction: str) -> None:
    duplicated = metadata.assign(
        __subject = metadata[subject].astype(str),
        __fraction = metadata[fraction].astype(str)
    ).duplicated(['__subject', '__fraction'], keep = False)

    if duplicated.any():
        duplicate_pairs = (
            metadata.assign(
                __subject = metadata[subject].astype(str),
                __fraction = metadata[fraction].astype(str)
            )
            .loc[duplicated, ['__subject', '__fraction']]
            .drop_duplicates()
            .head(5)
            .to_dict('records')
        )
        raise MonoPolyInputError(
            'Exactly one sample is required for each subject and fraction. '
            f'Examples of duplicated pairs: {duplicate_pairs}.'
        )


def _validate_fraction_configuration(fraction_order: list[str], abundance_fraction: str | None, allocation_fractions: list[str] | None) -> None:
    if not fraction_order:
        raise MonoPolyInputError('At least one fraction must be included in fraction_order.')

    if len(fraction_order) != len(set(fraction_order)):
        raise MonoPolyInputError('Fraction order must contain unique fraction names.')

    if abundance_fraction is not None and abundance_fraction not in fraction_order:
        raise MonoPolyInputError('The abundance fraction must be included in fraction_order.')

    if abundance_fraction is not None and fraction_order[0] != abundance_fraction:
        raise MonoPolyInputError(
            'The abundance fraction must be the first entry in fraction_order.'
        )

    if allocation_fractions is None:
        return

    if len(allocation_fractions) < 2:
        raise MonoPolyInputError(
            'At least two allocation fractions are required when allocation analysis is configured.'
        )

    if len(allocation_fractions) != len(set(allocation_fractions)):
        raise MonoPolyInputError('Allocation fractions must contain unique fraction names.')

    unknown_fractions = [
        fraction
        for fraction in allocation_fractions
        if fraction not in fraction_order
    ]

    if unknown_fractions:
        raise MonoPolyInputError(
            f'Allocation fractions are missing from fraction_order: {unknown_fractions}.'
        )

    if abundance_fraction in allocation_fractions:
        raise MonoPolyInputError('The abundance fraction cannot also be an allocation fraction.')


def _validate_fraction_measurement(fraction_measurement: str, fraction_weights: str | pd.Series | dict[str, float] | None) -> None:
    if fraction_measurement not in SUPPORTED_FRACTION_MEASUREMENTS:
        raise MonoPolyInputError(
            f'Unsupported fraction measurement {fraction_measurement!r}. '
            f'Expected one of {sorted(SUPPORTED_FRACTION_MEASUREMENTS)}.'
        )

    if fraction_measurement != 'relative_library' and fraction_weights is None:
        raise MonoPolyInputError(
            f'Fraction weights are required for fraction measurement {fraction_measurement!r}.'
        )


def _validate_filtering_parameters(min_count: int, min_samples: int) -> None:
    if not isinstance(min_count, int) or isinstance(min_count, bool):
        raise MonoPolyInputError('The minimum count threshold must be an integer.')

    if min_count < 0:
        raise MonoPolyInputError('The minimum count threshold must be nonnegative.')

    if not isinstance(min_samples, int) or isinstance(min_samples, bool):
        raise MonoPolyInputError('The minimum sample threshold must be an integer.')

    if min_samples < 1:
        raise MonoPolyInputError('The minimum sample threshold must be positive.')


def _validate_runtime_parameters(n_cpus: int | None, seed: int, quiet: bool) -> None:
    if n_cpus is not None:
        if not isinstance(n_cpus, int) or isinstance(n_cpus, bool):
            raise MonoPolyInputError('The CPU count must be an integer or None.')

        if n_cpus < 1:
            raise MonoPolyInputError('The CPU count must be positive.')

    if not isinstance(seed, int) or isinstance(seed, bool):
        raise MonoPolyInputError('The random seed must be an integer.')

    if not isinstance(quiet, bool):
        raise MonoPolyInputError('The quiet parameter must be a boolean.')


def _validate_engines(engines: list[str]) -> list[str]:
    if not engines:
        raise MonoPolyInputError('At least one statistical engine must be configured.')

    if any(not isinstance(engine, str) for engine in engines):
        raise MonoPolyInputError('Engine names must be strings.')

    unknown_engines = [
        engine
        for engine in engines
        if engine not in ENGINE_CLASSES
    ]

    if unknown_engines:
        raise MonoPolyInputError(f'Unknown statistical engines: {unknown_engines}.')

    return list(dict.fromkeys(engines))
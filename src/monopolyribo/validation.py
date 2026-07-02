# Imports ------------------------------------------
from __future__ import annotations

import numpy as np
import pandas as pd

from .exceptions import (
    InvalidCountMatrixError,
    MissingFractionError,
    MonoPolyInputError,
    UnpairedFractionError
)
# --------------------------------------------------


def validate_counts(counts: pd.DataFrame) -> None:
    if not isinstance(counts, pd.DataFrame):
        raise InvalidCountMatrixError('Counts must be provided as a pandas DataFrame.')

    if counts.empty:
        raise InvalidCountMatrixError('The count matrix must not be empty.')

    if not counts.index.is_unique:
        duplicate_samples = counts.index[counts.index.duplicated()].unique().tolist()[:5]

        raise InvalidCountMatrixError(
            f'The count matrix contains duplicate sample identifiers: {duplicate_samples}.'
        )

    if not counts.columns.is_unique:
        duplicate_features = counts.columns[counts.columns.duplicated()].unique().tolist()[:5]

        raise InvalidCountMatrixError(
            f'The count matrix contains duplicate feature identifiers: {duplicate_features}.'
        )

    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in counts.dtypes):
        raise InvalidCountMatrixError('The count matrix must contain only numeric columns.')

    count_values = counts.to_numpy(dtype = float)

    if not np.all(np.isfinite(count_values)):
        raise InvalidCountMatrixError('The count matrix must contain only finite values.')

    if np.any(count_values < 0.0):
        raise InvalidCountMatrixError('The count matrix must contain only nonnegative values.')

    if not np.all(np.equal(count_values, np.floor(count_values))):
        raise InvalidCountMatrixError('The count matrix must contain only integer values.')


def align_metadata(counts: pd.DataFrame, metadata: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    if not isinstance(metadata, pd.DataFrame):
        raise MonoPolyInputError('Metadata must be provided as a pandas DataFrame.')

    if metadata.empty:
        raise MonoPolyInputError('Metadata must not be empty.')

    if not metadata.index.is_unique:
        duplicate_samples = metadata.index[metadata.index.duplicated()].unique().tolist()[:5]

        raise MonoPolyInputError(
            f'Metadata contain duplicate sample identifiers: {duplicate_samples}.'
        )

    missing_metadata = counts.index.difference(metadata.index).tolist()
    extra_metadata = metadata.index.difference(counts.index).tolist()

    if missing_metadata or extra_metadata:
        raise MonoPolyInputError(
            f'Count and metadata sample identifiers do not match. '
            f'Missing metadata: {missing_metadata[:5]}. '
            f'Extra metadata: {extra_metadata[:5]}.'
        )

    aligned_metadata = metadata.loc[counts.index].copy()
    alignment_actions = []

    if not metadata.index.equals(counts.index):
        alignment_actions.append('metadata_aligned_to_counts')

    return aligned_metadata, alignment_actions


def validate_metadata(
    metadata: pd.DataFrame,
    subject: str,
    condition: str,
    fraction: str,
    fraction_order: list[str],
    covariates: list[str] | None = None
) -> None:
    _validate_column_name('subject', subject)
    _validate_column_name('condition', condition)
    _validate_column_name('fraction', fraction)

    covariates = [] if covariates is None else covariates

    if any(not isinstance(covariate, str) or not covariate for covariate in covariates):
        raise MonoPolyInputError('Covariate names must be nonempty strings.')

    required_columns = [subject, condition, fraction] + list(covariates)
    missing_columns = [column for column in required_columns if column not in metadata.columns]

    if missing_columns:
        raise MonoPolyInputError(
            f'Metadata are missing required columns: {missing_columns}.'
        )

    for column in required_columns:
        missing_samples = metadata.index[metadata[column].isna()].tolist()[:5]

        if missing_samples:
            raise MonoPolyInputError(
                f'Metadata column {column!r} contains missing values for samples '
                f'{missing_samples}.'
            )

    if metadata[subject].astype(str).str.strip().eq('').any():
        raise MonoPolyInputError('Subject identifiers must be nonempty.')

    if metadata[condition].astype(str).str.strip().eq('').any():
        raise MonoPolyInputError('Condition values must be nonempty.')

    if metadata[fraction].astype(str).str.strip().eq('').any():
        raise MonoPolyInputError('Fraction values must be nonempty.')

    observed_fractions = list(pd.unique(metadata[fraction].astype(str)))
    requested_fractions = [str(value) for value in fraction_order]

    absent_fractions = [
        requested_fraction
        for requested_fraction in requested_fractions
        if requested_fraction not in observed_fractions
    ]

    if absent_fractions:
        raise MissingFractionError(
            f'Requested fractions are absent from metadata: {absent_fractions}. '
            f'Observed fractions are {observed_fractions}.'
        )

    unknown_fractions = [
        observed_fraction
        for observed_fraction in observed_fractions
        if observed_fraction not in requested_fractions
    ]

    if unknown_fractions:
        raise MissingFractionError(
            f'Fraction order omits observed fractions: {unknown_fractions}.'
        )

    duplicate_covariates = [
        covariate
        for covariate in covariates
        if covariates.count(covariate) > 1
    ]

    if duplicate_covariates:
        raise MonoPolyInputError(
            f'Covariate names must be unique: {sorted(set(duplicate_covariates))}.'
        )

    protected_columns = {subject, condition, fraction}
    overlapping_covariates = [
        covariate
        for covariate in covariates
        if covariate in protected_columns
    ]

    if overlapping_covariates:
        raise MonoPolyInputError(
            f'Covariates cannot duplicate subject, condition, or fraction columns: '
            f'{overlapping_covariates}.'
        )


def validate_conditions(
    metadata: pd.DataFrame,
    condition: str,
    case: str | None,
    control: str | None
) -> None:
    if condition not in metadata.columns:
        raise MonoPolyInputError(
            f'Metadata are missing the condition column {condition!r}.'
        )

    if case is None or control is None:
        raise MonoPolyInputError('Both case and control conditions are required.')

    if not isinstance(case, str) or not case:
        raise MonoPolyInputError('The case condition must be a nonempty string.')

    if not isinstance(control, str) or not control:
        raise MonoPolyInputError('The control condition must be a nonempty string.')

    if case == control:
        raise MonoPolyInputError('The case and control conditions must be different.')

    condition_levels = set(metadata[condition].astype(str))
    missing_conditions = [
        requested_condition
        for requested_condition in [case, control]
        if requested_condition not in condition_levels
    ]

    if missing_conditions:
        raise MonoPolyInputError(
            f'Requested conditions are absent from metadata: {missing_conditions}. '
            f'Observed conditions are {sorted(condition_levels)}.'
        )


def validate_complete_subject_fractions(
    metadata: pd.DataFrame,
    subject: str,
    fraction: str,
    fractions: list[str]
) -> None:
    if not fractions:
        raise MonoPolyInputError('At least one fraction is required for completeness validation.')

    if len(fractions) != len(set(fractions)):
        raise MonoPolyInputError('Fractions must be unique for completeness validation.')

    if subject not in metadata.columns:
        raise MonoPolyInputError(
            f'Metadata are missing the subject column {subject!r}.'
        )

    if fraction not in metadata.columns:
        raise MonoPolyInputError(
            f'Metadata are missing the fraction column {fraction!r}.'
        )

    subject_fraction_counts = pd.crosstab(
        metadata[subject],
        metadata[fraction]
    ).reindex(columns = fractions, fill_value = 0)

    missing_fraction_subjects = subject_fraction_counts.index[
        (subject_fraction_counts == 0).any(axis = 1)
    ].tolist()

    if missing_fraction_subjects:
        raise UnpairedFractionError(
            f'Subjects have incomplete fraction sets: {missing_fraction_subjects[:5]}. '
            f'Each subject must contain fractions {fractions}.'
        )

    duplicate_fraction_subjects = subject_fraction_counts.index[
        (subject_fraction_counts > 1).any(axis = 1)
    ].tolist()

    if duplicate_fraction_subjects:
        raise UnpairedFractionError(
            f'Subjects contain duplicate samples for one or more fractions: '
            f'{duplicate_fraction_subjects[:5]}.'
        )


def _validate_column_name(name: str, value: str) -> None:
    if not isinstance(value, str) or not value:
        raise MonoPolyInputError(
            f'The {name} column name must be a nonempty string.'
        )
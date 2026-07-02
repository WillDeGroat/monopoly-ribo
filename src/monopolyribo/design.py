# Imports ------------------------------------------
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import linalg

from .exceptions import MonoPolyInputError
# --------------------------------------------------


def abundance_design_matrix(
    metadata: pd.DataFrame,
    subject: str,
    condition: str,
    case: str,
    control: str,
    covariates: list[str] | None = None
) -> pd.DataFrame:
    covariates = [] if covariates is None else list(covariates)

    _validate_common_inputs(
        metadata,
        subject,
        condition,
        case,
        control,
        covariates
    )
    _validate_one_row_per_subject(metadata, subject, 'abundance')

    design_metadata = metadata.copy()
    condition_effect = _condition_effect(
        design_metadata[condition],
        case,
        control
    )

    design_parts: list[pd.DataFrame | pd.Series] = [
        pd.Series(1.0, index = design_metadata.index, name = 'intercept'),
        condition_effect
    ]
    design_parts.extend(
        _subject_level_covariate_columns(
            design_metadata,
            subject,
            covariates
        )
    )

    return _finalize_design(design_parts, 'abundance')


def redistribution_design_matrix(
    metadata: pd.DataFrame,
    subject: str,
    condition: str,
    case: str,
    control: str,
    fraction: str,
    fraction_order: list[str],
    covariates: list[str] | None = None
) -> pd.DataFrame:
    covariates = [] if covariates is None else list(covariates)

    _validate_common_inputs(
        metadata,
        subject,
        condition,
        case,
        control,
        covariates
    )
    _validate_fraction_inputs(metadata, fraction, fraction_order)
    _validate_one_row_per_subject_fraction(metadata, subject, fraction)

    design_metadata = metadata.copy()
    design_metadata[subject] = design_metadata[subject].astype(str)
    design_metadata[fraction] = pd.Categorical(
        design_metadata[fraction].astype(str),
        categories = fraction_order,
        ordered = True
    )

    subject_effects = pd.get_dummies(
        design_metadata[subject],
        prefix = 'subject',
        drop_first = True,
        dtype = float
    )
    fraction_effects = pd.get_dummies(
        design_metadata[fraction],
        prefix = 'fraction',
        drop_first = True,
        dtype = float
    )
    condition_effect = _condition_effect(
        design_metadata[condition],
        case,
        control
    )

    design_parts: list[pd.DataFrame | pd.Series] = [
        pd.Series(1.0, index = design_metadata.index, name = 'intercept'),
        subject_effects,
        fraction_effects
    ]

    for fraction_column in fraction_effects.columns:
        interaction = condition_effect * fraction_effects[fraction_column]
        interaction.name = f'{condition_effect.name}:{fraction_column}'
        design_parts.append(interaction)

    covariate_columns = _subject_level_covariate_columns(
        design_metadata,
        subject,
        covariates
    )

    for covariate_column in covariate_columns:
        for fraction_column in fraction_effects.columns:
            interaction = covariate_column * fraction_effects[fraction_column]
            interaction.name = f'{covariate_column.name}:{fraction_column}'
            design_parts.append(interaction)

    return _finalize_design(design_parts, 'redistribution')


def _condition_effect(values: pd.Series, case: str, control: str) -> pd.Series:
    categorical = pd.Categorical(
        values.astype(str),
        categories = [control, case],
        ordered = True
    )

    if pd.isna(categorical).any():
        raise MonoPolyInputError(
            'The condition column contains values outside the configured case and control levels.'
        )

    return pd.Series(
        (categorical == case).astype(float),
        index = values.index,
        name = f'condition_{case}'
    )


def _subject_level_covariate_columns(metadata: pd.DataFrame, subject: str, covariates: list[str]) -> list[pd.Series]:
    columns: list[pd.Series] = []

    for covariate in covariates:
        within_subject_levels = metadata.groupby(
            metadata[subject].astype(str),
            observed = True
        )[covariate].nunique(dropna = False)

        if (within_subject_levels > 1).any():
            varying_subjects = within_subject_levels[within_subject_levels > 1].index.tolist()[:5]
            raise MonoPolyInputError(
                f'Covariate {covariate!r} must be constant within each subject. '
                f'Examples of subjects with multiple values: {varying_subjects}.'
            )

        if pd.api.types.is_numeric_dtype(metadata[covariate]):
            values = pd.to_numeric(metadata[covariate], errors = 'coerce').astype(float)

            if values.isna().any() or not np.all(np.isfinite(values.to_numpy())):
                raise MonoPolyInputError(
                    f'Numeric covariate {covariate!r} must contain only finite numeric values.'
                )

            subject_values = (
                pd.DataFrame(
                    {
                        subject: metadata[subject].astype(str),
                        covariate: values
                    },
                    index = metadata.index
                )
                .drop_duplicates(subset = subject)
                .set_index(subject)[covariate]
            )
            centered_subject_values = subject_values - subject_values.mean()
            centered_values = metadata[subject].astype(str).map(centered_subject_values)
            columns.append(centered_values.astype(float).rename(covariate))
            continue

        categorical_values = metadata[covariate].astype(str)
        categories = sorted(categorical_values.unique())
        categorical = pd.Categorical(
            categorical_values,
            categories = categories,
            ordered = True
        )
        effects = pd.get_dummies(
            categorical,
            prefix = covariate,
            drop_first = True,
            dtype = float
        )
        effects.index = metadata.index
        columns.extend(effects[column] for column in effects.columns)

    return columns


def _finalize_design(design_parts: list[pd.DataFrame | pd.Series], model_name: str) -> pd.DataFrame:
    design = pd.concat(design_parts, axis = 1)

    if design.columns.has_duplicates:
        duplicate_columns = design.columns[design.columns.duplicated()].unique().tolist()[:5]
        raise MonoPolyInputError(
            f'The {model_name} design matrix contains duplicate columns: {duplicate_columns}.'
        )

    constant_columns = [
        column
        for column in design.columns
        if column != 'intercept' and design[column].nunique(dropna = False) <= 1
    ]

    if constant_columns:
        design = design.drop(columns = constant_columns)

    design = design.astype(float)
    design_array = design.to_numpy()

    if not np.all(np.isfinite(design_array)):
        raise MonoPolyInputError(
            f'The {model_name} design matrix must contain only finite values.'
        )

    rank = np.linalg.matrix_rank(design_array)

    if rank < design_array.shape[1]:
        _, _, pivots = linalg.qr(
            design_array,
            mode = 'economic',
            pivoting = True
        )
        aliased_columns = [
            str(design.columns[index])
            for index in pivots[rank:]
        ]
        raise MonoPolyInputError(
            f'The {model_name} design matrix is rank deficient. '
            f'Aliased columns include: {aliased_columns[:5]}.'
        )

    return design


def _validate_common_inputs(metadata: pd.DataFrame, subject: str, condition: str, case: str, control: str, covariates: list[str]) -> None:
    if not isinstance(metadata, pd.DataFrame):
        raise TypeError('Metadata must be provided as a pandas DataFrame.')

    if metadata.empty:
        raise MonoPolyInputError('Metadata must not be empty.')

    required_columns = [subject, condition] + covariates
    missing_columns = [column for column in required_columns if column not in metadata.columns]

    if missing_columns:
        raise MonoPolyInputError(
            f'Metadata are missing columns required for the design matrix: {missing_columns}.'
        )

    if metadata[required_columns].isna().any().any():
        raise MonoPolyInputError(
            'Metadata columns used in the design matrix must not contain missing values.'
        )

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

    subject_condition_counts = metadata.assign(
        __subject = metadata[subject].astype(str),
        __condition = metadata[condition].astype(str)
    ).groupby('__subject', observed = True)['__condition'].nunique()

    if (subject_condition_counts > 1).any():
        invalid_subjects = subject_condition_counts[subject_condition_counts > 1].index.tolist()[:5]
        raise MonoPolyInputError(
            'Each subject must belong to exactly one condition. '
            f'Examples of invalid subjects: {invalid_subjects}.'
        )


def _validate_fraction_inputs(metadata: pd.DataFrame, fraction: str, fraction_order: list[str]) -> None:
    if fraction not in metadata.columns:
        raise MonoPolyInputError(f'Metadata are missing fraction column {fraction!r}.')

    if metadata[fraction].isna().any():
        raise MonoPolyInputError('The fraction column must not contain missing values.')

    if not fraction_order:
        raise MonoPolyInputError('The design matrix requires at least one ordered fraction.')

    if len(fraction_order) != len(set(fraction_order)):
        raise MonoPolyInputError('Fraction order must contain unique fraction names.')

    observed_fractions = set(metadata[fraction].astype(str))
    expected_fractions = set(fraction_order)
    missing_fractions = sorted(expected_fractions.difference(observed_fractions))
    unknown_fractions = sorted(observed_fractions.difference(expected_fractions))

    if missing_fractions:
        raise MonoPolyInputError(
            f'Fractions required by fraction_order are absent from metadata: {missing_fractions}.'
        )

    if unknown_fractions:
        raise MonoPolyInputError(
            f'Metadata contain fractions omitted from fraction_order: {unknown_fractions}.'
        )


def _validate_one_row_per_subject(metadata: pd.DataFrame, subject: str, model_name: str) -> None:
    duplicated = metadata[subject].astype(str).duplicated(keep = False)

    if duplicated.any():
        duplicate_subjects = metadata.loc[duplicated, subject].astype(str).unique().tolist()[:5]
        raise MonoPolyInputError(
            f'The {model_name} model requires exactly one sample per subject. '
            f'Examples of duplicated subjects: {duplicate_subjects}.'
        )


def _validate_one_row_per_subject_fraction(metadata: pd.DataFrame, subject: str, fraction: str) -> None:
    pairs = pd.DataFrame(
        {
            'subject': metadata[subject].astype(str),
            'fraction': metadata[fraction].astype(str)
        },
        index = metadata.index
    )
    duplicated = pairs.duplicated(['subject', 'fraction'], keep = False)

    if duplicated.any():
        duplicate_pairs = (
            pairs.loc[duplicated, ['subject', 'fraction']]
            .drop_duplicates()
            .head(5)
            .to_dict('records')
        )
        raise MonoPolyInputError(
            'The redistribution model requires exactly one sample per subject and fraction. '
            f'Examples of duplicated pairs: {duplicate_pairs}.'
        )
# Imports ------------------------------------------
from __future__ import annotations

import numpy as np
import pandas as pd

from .exceptions import MonoPolyInputError
# --------------------------------------------------


def design_matrix(
    metadata: pd.DataFrame,
    subject: str,
    condition: str,
    fraction: str,
    fraction_order: list[str],
    covariates: list[str] | None = None
) -> pd.DataFrame:
    _validate_design_inputs(
        metadata,
        subject,
        condition,
        fraction,
        fraction_order,
        covariates
    )

    design_metadata = metadata.copy()
    covariates = [] if covariates is None else list(covariates)

    design_metadata[subject] = design_metadata[subject].astype(str)
    design_metadata[condition] = design_metadata[condition].astype(str)
    design_metadata[fraction] = pd.Categorical(
        design_metadata[fraction].astype(str),
        categories = fraction_order,
        ordered = True
    )

    condition_levels = sorted(design_metadata[condition].unique())
    design_metadata[condition] = pd.Categorical(
        design_metadata[condition],
        categories = condition_levels,
        ordered = True
    )

    design_parts: list[pd.DataFrame | pd.Series] = [
        pd.Series(1.0, index = design_metadata.index, name = 'intercept')
    ]

    subject_effects = pd.get_dummies(
        design_metadata[subject],
        prefix = 'subject',
        drop_first = True,
        dtype = float
    )

    condition_effects = pd.get_dummies(
        design_metadata[condition],
        prefix = 'condition',
        drop_first = True,
        dtype = float
    )

    fraction_effects = pd.get_dummies(
        design_metadata[fraction],
        prefix = 'fraction',
        drop_first = True,
        dtype = float
    )

    design_parts.extend(
        [
            subject_effects,
            condition_effects,
            fraction_effects
        ]
    )

    for condition_column in condition_effects.columns:
        for fraction_column in fraction_effects.columns:
            interaction = condition_effects[condition_column] * fraction_effects[fraction_column]
            interaction.name = f'{condition_column}:{fraction_column}'
            design_parts.append(interaction)

    for covariate in covariates:
        if pd.api.types.is_numeric_dtype(design_metadata[covariate]):
            covariate_values = pd.to_numeric(
                design_metadata[covariate],
                errors = 'coerce'
            ).astype(float)

            design_parts.append(covariate_values.rename(covariate))
        else:
            covariate_effects = pd.get_dummies(
                design_metadata[covariate].astype(str),
                prefix = covariate,
                drop_first = True,
                dtype = float
            )

            design_parts.append(covariate_effects)

    design = pd.concat(design_parts, axis = 1)

    if design.columns.has_duplicates:
        duplicate_columns = design.columns[design.columns.duplicated()].unique().tolist()[:5]

        raise MonoPolyInputError(
            f'The design matrix contains duplicate columns: {duplicate_columns}.'
        )

    design_array = design.to_numpy(dtype = float)

    if not np.all(np.isfinite(design_array)):
        raise MonoPolyInputError('The design matrix must contain only finite values.')

    constant_columns = [
        column
        for column in design.columns
        if column != 'intercept' and design[column].nunique(dropna = False) <= 1
    ]

    if constant_columns:
        design = design.drop(columns = constant_columns)

    return design.astype(float)


def _validate_design_inputs(
    metadata: pd.DataFrame,
    subject: str,
    condition: str,
    fraction: str,
    fraction_order: list[str],
    covariates: list[str] | None
) -> None:
    if not isinstance(metadata, pd.DataFrame):
        raise TypeError('Metadata must be provided as a pandas DataFrame.')

    if metadata.empty:
        raise MonoPolyInputError('Metadata must not be empty.')

    required_columns = [subject, condition, fraction] + list(covariates or [])
    missing_columns = [column for column in required_columns if column not in metadata.columns]

    if missing_columns:
        raise MonoPolyInputError(
            f'Metadata are missing columns required for the design matrix: {missing_columns}.'
        )

    if metadata[required_columns].isna().any().any():
        raise MonoPolyInputError(
            'Metadata columns used in the design matrix must not contain missing values.'
        )

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

    if metadata[subject].astype(str).nunique() < 2:
        raise MonoPolyInputError('The design matrix requires at least two subjects.')

    if metadata[condition].astype(str).nunique() < 2:
        raise MonoPolyInputError('The design matrix requires at least two condition levels.')

    for covariate in covariates or []:
        if pd.api.types.is_numeric_dtype(metadata[covariate]):
            numeric_values = pd.to_numeric(metadata[covariate], errors = 'coerce')

            if numeric_values.isna().any():
                raise MonoPolyInputError(
                    f'Numeric covariate {covariate!r} must contain only numeric values.'
                )
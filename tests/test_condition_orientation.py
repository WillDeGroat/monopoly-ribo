# Imports ------------------------------------------
import numpy as np
import pandas as pd
import pytest

from monopolyribo.engines.beta_binomial import (
    _condition_indicator as beta_binomial_condition_indicator
)
from monopolyribo.engines.dirichlet_multinomial import (
    _condition_indicator as dirichlet_multinomial_condition_indicator
)
# --------------------------------------------------


CONDITION_INDICATOR_FUNCTIONS = [
    beta_binomial_condition_indicator,
    dirichlet_multinomial_condition_indicator
]


@pytest.mark.parametrize(
    'indicator_function',
    CONDITION_INDICATOR_FUNCTIONS
)
def test_condition_indicator_uses_explicit_orientation(indicator_function) -> None:
    subject_conditions = pd.Series(
        ['z_case', 'a_control', 'z_case', 'a_control'],
        index = ['s1', 's2', 's3', 's4']
    )

    condition_indicator = indicator_function(
        subject_conditions,
        case = 'z_case',
        control = 'a_control'
    )

    np.testing.assert_array_equal(
        condition_indicator,
        np.array([1.0, 0.0, 1.0, 0.0])
    )


@pytest.mark.parametrize(
    'indicator_function',
    CONDITION_INDICATOR_FUNCTIONS
)
def test_condition_indicator_does_not_use_alphabetical_orientation(indicator_function) -> None:
    subject_conditions = pd.Series(
        ['a_case', 'z_control', 'a_case', 'z_control'],
        index = ['s1', 's2', 's3', 's4']
    )

    condition_indicator = indicator_function(
        subject_conditions,
        case = 'a_case',
        control = 'z_control'
    )

    np.testing.assert_array_equal(
        condition_indicator,
        np.array([1.0, 0.0, 1.0, 0.0])
    )


@pytest.mark.parametrize(
    'indicator_function',
    CONDITION_INDICATOR_FUNCTIONS
)
def test_condition_indicator_rejects_unconfigured_levels(indicator_function) -> None:
    subject_conditions = pd.Series(
        ['case', 'control', 'other'],
        index = ['s1', 's2', 's3']
    )

    with pytest.raises(
        ValueError,
        match = 'configured case and control'
    ):
        indicator_function(
            subject_conditions,
            case = 'case',
            control = 'control'
        )


@pytest.mark.parametrize(
    'indicator_function',
    CONDITION_INDICATOR_FUNCTIONS
)
def test_condition_indicator_rejects_missing_values(indicator_function) -> None:
    subject_conditions = pd.Series(
        ['case', None, 'control'],
        index = ['s1', 's2', 's3']
    )

    with pytest.raises(
        ValueError,
        match = 'missing'
    ):
        indicator_function(
            subject_conditions,
            case = 'case',
            control = 'control'
        )
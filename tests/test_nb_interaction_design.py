# Imports ------------------------------------------
import numpy as np
import pandas as pd
import pytest

from monopolyribo import FractionContrast, MonoPolyDataSet, MonoPolyStats
from monopolyribo.datasets import load_example_data
from monopolyribo.exceptions import MonoPolyInputError
# --------------------------------------------------


FRACTION_ORDER = ['Input', 'Monosome', 'Polysome']
ALLOCATION_FRACTIONS = ['Monosome', 'Polysome']


def _fit_example_dataset() -> MonoPolyDataSet:
    counts, metadata = load_example_data()

    return MonoPolyDataSet(
        counts = counts,
        metadata = metadata,
        subject = 'subject',
        condition = 'condition',
        case = 'case',
        control = 'control',
        fraction = 'fraction',
        fraction_order = FRACTION_ORDER,
        abundance_fraction = 'Input',
        allocation_fractions = ALLOCATION_FRACTIONS,
        engines = ['nb_interaction']
    ).fit()


def test_interaction_models_are_full_rank() -> None:
    dataset = _fit_example_dataset()
    fit = dataset.fits['nb_interaction']

    assert fit.metadata['design_rank_deficient'] is False

    for model_name in ['abundance', 'redistribution']:
        model = fit.models[model_name]
        design = model['design_array']
        assert np.linalg.matrix_rank(design) == design.shape[1]


def test_abundance_contrast_reverses_sign() -> None:
    dataset = _fit_example_dataset()

    forward = MonoPolyStats(
        dataset,
        contrast = FractionContrast.abundance(
            'case',
            'control',
            'Input'
        )
    ).summary()
    reverse = MonoPolyStats(
        dataset,
        contrast = FractionContrast.abundance(
            'control',
            'case',
            'Input'
        )
    ).summary()

    pd.testing.assert_series_equal(
        forward['effect'],
        -reverse['effect'],
        check_names = False
    )
    pd.testing.assert_series_equal(
        forward['standard_error'],
        reverse['standard_error'],
        check_names = False
    )
    pd.testing.assert_series_equal(
        forward['pvalue'],
        reverse['pvalue'],
        check_names = False
    )


def test_fraction_vs_input_uses_zero_reference_interaction() -> None:
    dataset = _fit_example_dataset()
    contrast = FractionContrast.fraction_vs_input(
        'case',
        'control',
        'Monosome',
        'Input'
    )

    results = MonoPolyStats(
        dataset,
        contrast = contrast,
        engine = 'nb_interaction'
    ).summary()

    assert not results.empty
    assert results['effect'].notna().all()


def test_dataset_rejects_subjects_assigned_to_multiple_conditions() -> None:
    counts, metadata = load_example_data()
    subject = metadata['subject'].iloc[0]
    subject_rows = metadata['subject'] == subject
    metadata.loc[subject_rows, 'condition'] = ['case', 'control', 'case']

    with pytest.raises(
        MonoPolyInputError,
        match = 'Each subject must belong to exactly one condition'
    ):
        MonoPolyDataSet(
            counts = counts,
            metadata = metadata,
            subject = 'subject',
            condition = 'condition',
            case = 'case',
            control = 'control',
            fraction = 'fraction',
            fraction_order = FRACTION_ORDER
        )
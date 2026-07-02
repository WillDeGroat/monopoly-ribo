# Imports ------------------------------------------
import os
from pathlib import Path

import pandas as pd
import pytest

from monopolyribo import FractionContrast, MonoPolyDataSet, MonoPolyStats, analyze
from monopolyribo.datasets import load_example_data, simulate_ribofraction_data
from monopolyribo.exceptions import InvalidCountMatrixError
# --------------------------------------------------


FRACTION_ORDER = ['Input', 'Monosome', 'Polysome']
ALLOCATION_FRACTIONS = ['Monosome', 'Polysome']


def test_acceptance_analysis_runs_without_writing_files(tmp_path: Path) -> None:
    counts, metadata = load_example_data()
    files_before_analysis = set(os.listdir(tmp_path))

    result = analyze(
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
        engines = ['nb_interaction', 'beta_binomial', 'joint_latent_mle']
    )

    assert isinstance(result.abundance, pd.DataFrame)
    assert isinstance(result.fraction_effects, dict)
    assert isinstance(result.redistribution, pd.DataFrame)
    assert isinstance(result.allocation, pd.DataFrame)
    assert isinstance(result.joint, pd.DataFrame)
    assert isinstance(result.integrated, pd.DataFrame)
    assert isinstance(result.classification, pd.DataFrame)
    assert isinstance(result.stability, pd.DataFrame)
    assert isinstance(result.diagnostics, pd.DataFrame)

    assert result.abundance.index.is_unique
    assert result.redistribution.index.is_unique
    assert result.integrated.index.is_unique
    assert result['redistribution'].equals(result.redistribution)
    assert set(os.listdir(tmp_path)) == files_before_analysis


def test_advanced_api_matches_complete_analysis() -> None:
    counts, metadata = load_example_data()

    result = analyze(
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
    )

    dataset = MonoPolyDataSet(
        counts = counts,
        metadata = metadata,
        subject = 'subject',
        condition = 'condition',
        fraction = 'fraction',
        fraction_order = FRACTION_ORDER,
        abundance_fraction = 'Input',
        allocation_fractions = ALLOCATION_FRACTIONS,
        engines = ['nb_interaction']
    ).fit()

    contrast = FractionContrast.redistribution(
        case = 'case',
        control = 'control',
        numerator_fraction = 'Polysome',
        denominator_fraction = 'Monosome'
    )

    redistribution_results = MonoPolyStats(
        dataset,
        contrast = contrast,
        engine = 'nb_interaction'
    ).summary()

    pd.testing.assert_series_equal(
        result.redistribution['effect'],
        redistribution_results['effect']
    )


def test_multifraction_analysis_runs() -> None:
    fraction_order = ['Input', 'Free', 'Monosome', 'Light', 'Heavy']
    allocation_fractions = ['Free', 'Monosome', 'Light', 'Heavy']

    counts, metadata, _ = simulate_ribofraction_data(
        fraction_order = fraction_order,
        scenario = 'multi_fraction_shift'
    )

    result = analyze(
        counts = counts,
        metadata = metadata,
        subject = 'subject',
        condition = 'condition',
        case = 'case',
        control = 'control',
        fraction = 'fraction',
        fraction_order = fraction_order,
        abundance_fraction = 'Input',
        allocation_fractions = allocation_fractions,
        engines = ['nb_interaction', 'dirichlet_multinomial', 'joint_latent_mle'],
        fraction_weights = {
            'Free': 0.0,
            'Monosome': 1.0,
            'Light': 2.5,
            'Heavy': 6.0
        }
    )

    assert not result.allocation.empty
    assert result.allocation.index.is_unique
    assert result.integrated.index.is_unique
    assert set(result.fraction_effects) == set(allocation_fractions)


def test_validation_rejects_negative_counts() -> None:
    counts, metadata = load_example_data()
    counts.iloc[0, 0] = -1

    with pytest.raises(
        InvalidCountMatrixError,
        match = 'nonnegative'
    ):
        MonoPolyDataSet(
            counts = counts,
            metadata = metadata,
            subject = 'subject',
            condition = 'condition',
            fraction = 'fraction',
            fraction_order = FRACTION_ORDER
        )


def test_csv_writer_converts_columns_without_mutating_results(tmp_path: Path) -> None:
    counts, metadata = load_example_data()

    result = analyze(
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
    )

    original_columns = result.redistribution.columns.copy()
    output_directory = tmp_path / 'output'

    result.write_csv(output_directory)

    assert result.redistribution.columns.equals(original_columns)

    output_path = output_directory / 'redistribution.csv'

    assert output_path.exists()

    written_results = pd.read_csv(output_path)

    assert 'standardError' in written_results.columns
    assert 'standard_error' not in written_results.columns
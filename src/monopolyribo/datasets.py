# Imports ------------------------------------------
from __future__ import annotations

import numpy as np
import pandas as pd
# --------------------------------------------------


SUPPORTED_SCENARIOS = {
    'abundance_only',
    'buffering',
    'inversion',
    'monosome_retention',
    'multi_fraction_shift',
    'polysome_recruitment',
    'reinforcement'
}

REDISTRIBUTION_SCENARIOS = {
    'buffering',
    'inversion',
    'monosome_retention',
    'multi_fraction_shift',
    'polysome_recruitment',
    'reinforcement'
}


def simulate_ribofraction_data(
    n_subjects: int = 8,
    n_features: int = 30,
    fraction_order: list[str] | None = None,
    scenario: str = 'polysome_recruitment',
    seed: int = 0
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    _validate_simulation_parameters(
        n_subjects,
        n_features,
        fraction_order,
        scenario,
        seed
    )

    rng = np.random.default_rng(seed)
    fractions = ['Input', 'Monosome', 'Polysome'] if fraction_order is None else list(fraction_order)

    abundance_fraction = fractions[0]
    allocation_fractions = fractions[1:]
    allocation_weights = _allocation_shift_weights(allocation_fractions)

    subjects = [f'subject_{index:02d}' for index in range(n_subjects)]
    conditions = {
        subject: 'case' if index >= n_subjects // 2 else 'control'
        for index, subject in enumerate(subjects)
    }

    metadata_rows: list[dict[str, str]] = []

    for subject in subjects:
        for fraction in fractions:
            sample_id = f'{subject}_{fraction}'

            metadata_rows.append(
                {
                    'sample': sample_id,
                    'subject': subject,
                    'condition': conditions[subject],
                    'fraction': fraction
                }
            )

    metadata = pd.DataFrame(metadata_rows).set_index('sample')

    feature_ids = [f'feature_{index:03d}' for index in range(n_features)]
    baseline_means = rng.gamma(shape = 8.0, scale = 20.0, size = n_features)
    counts = np.zeros((len(metadata), n_features), dtype = int)
    truth_rows: list[dict[str, str | float]] = []

    affected_features = max(1, n_features // 4)

    for feature_index, feature_id in enumerate(feature_ids):
        abundance_effect, redistribution_effect = _scenario_effects(
            scenario,
            feature_index,
            affected_features
        )

        truth_rows.append(
            {
                'feature_id': feature_id,
                'abundance_effect': abundance_effect,
                'redistribution_effect': redistribution_effect
            }
        )

        for sample_index, (_, sample_metadata) in enumerate(metadata.iterrows()):
            condition = sample_metadata['condition']
            fraction = sample_metadata['fraction']
            mean_count = baseline_means[feature_index]

            if condition == 'case' and fraction == abundance_fraction:
                mean_count *= 2.0 ** abundance_effect

            if fraction != abundance_fraction:
                mean_count *= 0.7

                if condition == 'case':
                    mean_count *= 2.0 ** (
                        redistribution_effect * allocation_weights[fraction]
                    )

            counts[sample_index, feature_index] = rng.negative_binomial(
                20.0,
                20.0 / (20.0 + mean_count)
            )

    count_table = pd.DataFrame(
        counts,
        index = metadata.index,
        columns = feature_ids
    )

    truth_table = pd.DataFrame(truth_rows).set_index('feature_id')

    return count_table, metadata, truth_table


def load_example_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    counts, metadata, _ = simulate_ribofraction_data(seed = 1)
    return counts, metadata


def _allocation_shift_weights(allocation_fractions: list[str]) -> dict[str, float]:
    if not allocation_fractions:
        return {}

    weights = np.linspace(-0.5, 0.5, len(allocation_fractions))

    return {
        fraction: float(weight)
        for fraction, weight in zip(allocation_fractions, weights, strict = True)
    }


def _scenario_effects(scenario: str, feature_index: int, affected_features: int) -> tuple[float, float]:
    if feature_index >= affected_features:
        return 0.0, 0.0

    abundance_effect = 0.0
    redistribution_effect = 0.0

    if scenario in {
        'multi_fraction_shift',
        'polysome_recruitment',
        'reinforcement'
    }:
        redistribution_effect = 0.8

    if scenario in {
        'buffering',
        'inversion',
        'monosome_retention'
    }:
        redistribution_effect = -0.8

    if scenario in {
        'abundance_only',
        'buffering',
        'inversion',
        'reinforcement'
    }:
        abundance_effect = 0.7

    return abundance_effect, redistribution_effect


def _validate_simulation_parameters(n_subjects: int, n_features: int, fraction_order: list[str] | None, scenario: str, seed: int) -> None:
    if not isinstance(n_subjects, int) or isinstance(n_subjects, bool):
        raise ValueError('The number of subjects must be an integer.')

    if n_subjects < 2:
        raise ValueError('The simulation requires at least two subjects.')

    if not isinstance(n_features, int) or isinstance(n_features, bool):
        raise ValueError('The number of features must be an integer.')

    if n_features < 1:
        raise ValueError('The simulation requires at least one feature.')

    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError('The random seed must be an integer.')

    if scenario not in SUPPORTED_SCENARIOS:
        raise ValueError(
            f'Unsupported simulation scenario {scenario!r}. '
            f'Expected one of {sorted(SUPPORTED_SCENARIOS)}.'
        )

    if fraction_order is None:
        return

    if not fraction_order:
        raise ValueError('At least one fraction must be provided.')

    if any(not isinstance(fraction, str) or not fraction for fraction in fraction_order):
        raise ValueError('Fraction names must be nonempty strings.')

    if len(fraction_order) != len(set(fraction_order)):
        raise ValueError('Fraction names must be unique.')

    if scenario in REDISTRIBUTION_SCENARIOS and len(fraction_order) < 3:
        raise ValueError(
            'Redistribution scenarios require an abundance fraction and at least '
            'two allocation fractions.'
        )




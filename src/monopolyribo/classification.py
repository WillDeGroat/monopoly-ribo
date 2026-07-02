# Imports ------------------------------------------
from __future__ import annotations

import numpy as np
import pandas as pd
# --------------------------------------------------


CLASSIFICATION_COLUMNS = [
    'regulatory_class',
    'classification_reason'
]

INTEGRATED_RESULT_COLUMNS = [
    'feature_id',
    'abundance_effect',
    'abundance_padj',
    'redistribution_effect',
    'redistribution_padj',
    'allocation_effect',
    'allocation_padj',
    'joint_abundance_effect',
    'joint_abundance_padj',
    'joint_redistribution_effect',
    'joint_redistribution_padj'
]


def classify(integrated_results: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    _validate_alpha(alpha)
    _validate_integrated_results(integrated_results)

    classification_rows: list[dict[str, str]] = []

    for feature_id, feature_results in integrated_results.iterrows():
        abundance_effect = feature_results.get('abundance_effect', np.nan)
        abundance_padj = feature_results.get('abundance_padj', np.nan)
        redistribution_effect = feature_results.get('redistribution_effect', np.nan)
        redistribution_padj = feature_results.get('redistribution_padj', np.nan)

        regulatory_class, classification_reason = _classify_feature(
            abundance_effect,
            abundance_padj,
            redistribution_effect,
            redistribution_padj,
            alpha
        )

        classification_rows.append(
            {
                'feature_id': feature_id,
                'regulatory_class': regulatory_class,
                'classification_reason': classification_reason
            }
        )

    if not classification_rows:
        return pd.DataFrame(
            columns = ['feature_id'] + CLASSIFICATION_COLUMNS
        ).set_index('feature_id')

    return pd.DataFrame(classification_rows).set_index('feature_id')


def integrate(
    abundance_results: pd.DataFrame,
    redistribution_results: pd.DataFrame,
    allocation_results: pd.DataFrame,
    joint_results: pd.DataFrame
) -> pd.DataFrame:
    _validate_result_table(abundance_results, 'Abundance results')
    _validate_result_table(redistribution_results, 'Redistribution results')
    _validate_result_table(allocation_results, 'Allocation results')
    _validate_result_table(joint_results, 'Joint results')

    feature_index = (
        abundance_results.index
        .union(redistribution_results.index)
        .union(allocation_results.index)
        .union(joint_results.index)
    )

    integrated_results = pd.DataFrame(index = feature_index)
    integrated_results['feature_id'] = feature_index

    _add_result_columns(
        integrated_results,
        abundance_results,
        effect_column = 'abundance_effect',
        padj_column = 'abundance_padj',
        result_name = 'Abundance results'
    )

    _add_result_columns(
        integrated_results,
        redistribution_results,
        effect_column = 'redistribution_effect',
        padj_column = 'redistribution_padj',
        result_name = 'Redistribution results'
    )

    _add_result_columns(
        integrated_results,
        allocation_results,
        effect_column = 'allocation_effect',
        padj_column = 'allocation_padj',
        result_name = 'Allocation results'
    )

    if not joint_results.empty:
        _add_joint_result_columns(integrated_results, joint_results)

    classifications = classify(integrated_results)

    return integrated_results.join(classifications)


def _classify_feature(
    abundance_effect: float,
    abundance_padj: float,
    redistribution_effect: float,
    redistribution_padj: float,
    alpha: float
) -> tuple[str, str]:
    abundance_effect_valid = np.isfinite(abundance_effect)
    redistribution_effect_valid = np.isfinite(redistribution_effect)
    abundance_significant = _is_significant(abundance_padj, alpha)
    redistribution_significant = _is_significant(redistribution_padj, alpha)

    if not abundance_effect_valid and not redistribution_effect_valid:
        return 'low_information', 'missing_abundance_and_redistribution_effects'

    if abundance_significant and not abundance_effect_valid:
        return 'low_information', 'missing_abundance_effect'

    if redistribution_significant and not redistribution_effect_valid:
        return 'low_information', 'missing_redistribution_effect'

    if not abundance_significant and not redistribution_significant:
        return 'uncertain', 'no_significant_effects'

    if abundance_significant and not redistribution_significant:
        return 'abundance_only', 'abundance_significant_only'

    if redistribution_significant and not abundance_significant:
        if redistribution_effect > 0.0:
            return 'polysome_recruitment', 'positive_redistribution_only'

        if redistribution_effect < 0.0:
            return 'monosome_retention', 'negative_redistribution_only'

        return 'redistribution_only', 'zero_redistribution_effect'

    if np.sign(abundance_effect) == np.sign(redistribution_effect):
        return 'translational_reinforcement', 'abundance_and_redistribution_same_direction'

    if abs(redistribution_effect) < abs(abundance_effect):
        return 'translational_buffering', 'redistribution_opposes_but_does_not_exceed_abundance'

    return 'regulatory_inversion', 'redistribution_opposes_and_exceeds_abundance'


def _is_significant(adjusted_pvalue: float, alpha: float) -> bool:
    return bool(np.isfinite(adjusted_pvalue) and adjusted_pvalue <= alpha)


def _add_result_columns(
    integrated_results: pd.DataFrame,
    results: pd.DataFrame,
    effect_column: str,
    padj_column: str,
    result_name: str
) -> None:
    if results.empty:
        return

    _validate_required_columns(results, {'effect', 'padj'}, result_name)
    _validate_unique_feature_index(results, result_name)

    integrated_results[effect_column] = results['effect']
    integrated_results[padj_column] = results['padj']


def _add_joint_result_columns(integrated_results: pd.DataFrame, joint_results: pd.DataFrame) -> None:
    _validate_required_columns(
        joint_results,
        {'effect', 'effect_scale', 'padj'},
        'Joint results'
    )

    abundance_results = joint_results[
        joint_results['effect_scale'] == 'log2_fold_change'
    ]

    redistribution_results = joint_results[
        joint_results['effect_scale'].isin(
            {
                'log2_redistribution_ratio',
                'weighted_log2_allocation_shift'
            }
        )
    ]

    if not abundance_results.empty:
        _validate_unique_feature_index(
            abundance_results,
            'Joint abundance results'
        )

        integrated_results['joint_abundance_effect'] = abundance_results['effect']
        integrated_results['joint_abundance_padj'] = abundance_results['padj']

    if not redistribution_results.empty:
        _validate_unique_feature_index(
            redistribution_results,
            'Joint redistribution results'
        )

        integrated_results['joint_redistribution_effect'] = redistribution_results['effect']
        integrated_results['joint_redistribution_padj'] = redistribution_results['padj']


def _validate_result_table(results: pd.DataFrame, name: str) -> None:
    if not isinstance(results, pd.DataFrame):
        raise TypeError(f'{name} must be provided as a pandas DataFrame.')


def _validate_required_columns(results: pd.DataFrame, required_columns: set[str], name: str) -> None:
    missing_columns = required_columns.difference(results.columns)

    if missing_columns:
        raise ValueError(
            f'{name} are missing required columns: {sorted(missing_columns)}.'
        )


def _validate_unique_feature_index(results: pd.DataFrame, name: str) -> None:
    if results.index.has_duplicates:
        raise ValueError(f'{name} must contain unique feature indices.')


def _validate_integrated_results(integrated_results: pd.DataFrame) -> None:
    if not isinstance(integrated_results, pd.DataFrame):
        raise TypeError('Integrated results must be provided as a pandas DataFrame.')

    required_columns = {
        'abundance_effect',
        'abundance_padj',
        'redistribution_effect',
        'redistribution_padj'
    }
    missing_columns = required_columns.difference(integrated_results.columns)

    if missing_columns:
        raise ValueError(
            f'Integrated results are missing required columns: {sorted(missing_columns)}.'
        )


def _validate_alpha(alpha: float) -> None:
    if isinstance(alpha, bool) or not isinstance(alpha, int | float):
        raise ValueError('The significance threshold must be numeric.')

    if not np.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError('The significance threshold must be between zero and one.')
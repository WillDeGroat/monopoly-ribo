# Imports ------------------------------------------
from __future__ import annotations

from typing import Any

import pandas as pd

from ..contrasts import FractionContrast
from ..statistics import adjust_pvalues
from .base import EngineFit
from .nb_interaction import NBInteractionEngine, nb_contrast
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
    'padj'
]


class JointLatentEngine:
    name: str = 'joint_latent_mle'

    def fit(self, dataset: Any) -> EngineFit:
        interaction_fit = getattr(dataset, 'fits', {}).get('nb_interaction')

        if interaction_fit is None:
            interaction_fit = NBInteractionEngine().fit(dataset)

        control_condition, case_condition = _condition_levels(dataset)
        result_tables: list[pd.DataFrame] = []

        if dataset.abundance_fraction is not None:
            abundance_contrast = FractionContrast.abundance(
                case_condition,
                control_condition,
                dataset.abundance_fraction
            )

            abundance_results = nb_contrast(interaction_fit, dataset, abundance_contrast)
            abundance_results['engine'] = self.name
            abundance_results['effect_scale'] = 'log2_fold_change'
            abundance_results['padj'] = adjust_pvalues(abundance_results['pvalue'])
            result_tables.append(abundance_results)

        allocation_fractions = dataset.allocation_fractions or []

        if len(allocation_fractions) >= 2:
            denominator_fraction = allocation_fractions[0]
            numerator_fraction = allocation_fractions[-1]

            redistribution_contrast = FractionContrast.redistribution(
                case_condition,
                control_condition,
                numerator_fraction,
                denominator_fraction
            )

            redistribution_results = nb_contrast(
                interaction_fit,
                dataset,
                redistribution_contrast
            )

            redistribution_results['engine'] = self.name
            redistribution_results['padj'] = adjust_pvalues(redistribution_results['pvalue'])
            result_tables.append(redistribution_results)

        if result_tables:
            results = pd.concat(result_tables, axis = 0, sort = False)
            results['warning_code'] = _apply_fraction_warning(
                results['warning_code'],
                dataset.fraction_measurement
            )
        else:
            results = pd.DataFrame(columns = RESULT_COLUMNS).set_index('feature_id')

        return EngineFit(
            name = self.name,
            results = {'joint': results},
            metadata = {
                'backend': self.name,
                'penalty': 'inherits ridge-stabilized log-linear approximation',
                'multiple_testing': 'adjusted separately by contrast family'
            }
        )


def _condition_levels(dataset: Any) -> tuple[str, str]:
    condition_levels = sorted(pd.unique(dataset.metadata[dataset.condition].astype(str)))

    if len(condition_levels) != 2:
        raise ValueError('The joint latent engine requires exactly two condition levels.')

    control_condition = condition_levels[0]
    case_condition = condition_levels[1]

    return control_condition, case_condition


def _apply_fraction_warning(warning_codes: pd.Series, fraction_measurement: str) -> pd.Series:
    if fraction_measurement != 'relative_library':
        return warning_codes

    return warning_codes.mask(warning_codes.eq(''), 'relative_library_caution')
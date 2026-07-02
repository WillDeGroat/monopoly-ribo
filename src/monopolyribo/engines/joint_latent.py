# Imports ------------------------------------------
from __future__ import annotations

from typing import Any

import pandas as pd

from ..contrasts import FractionContrast
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

        result_tables: list[pd.DataFrame] = []

        if dataset.abundance_fraction is not None:
            abundance_contrast = FractionContrast.abundance(
                dataset.case,
                dataset.control,
                dataset.abundance_fraction
            )
            abundance_results = nb_contrast(
                interaction_fit,
                dataset,
                abundance_contrast
            )
            abundance_results['engine'] = self.name
            abundance_results['effect_scale'] = 'log2_fold_change'
            result_tables.append(abundance_results)

        allocation_fractions = dataset.allocation_fractions or []

        if len(allocation_fractions) >= 2:
            redistribution_contrast = FractionContrast.redistribution(
                dataset.case,
                dataset.control,
                allocation_fractions[-1],
                allocation_fractions[0]
            )
            redistribution_results = nb_contrast(
                interaction_fit,
                dataset,
                redistribution_contrast
            )
            redistribution_results['engine'] = self.name
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
                'method': (
                    'Shared abundance and redistribution summaries from the '
                    'robust interaction-model backend'
                ),
                'multiple_testing': 'Adjusted separately within each contrast family'
            }
        )


def _apply_fraction_warning(warning_codes: pd.Series, fraction_measurement: str) -> pd.Series:
    if fraction_measurement != 'relative_library':
        return warning_codes

    return warning_codes.mask(warning_codes.eq(''), 'relative_library_caution')

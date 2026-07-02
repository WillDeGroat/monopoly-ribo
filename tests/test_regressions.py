# Imports ------------------------------------------
import pandas as pd

from monopolyribo import FractionContrast, MonoPolyDataSet, MonoPolyResult, MonoPolyStats
from monopolyribo.datasets import load_example_data
from monopolyribo.robustness import leave_one_subject_out

# --------------------------------------------------


FRACTION_ORDER = ['Input', 'Monosome', 'Polysome']
ALLOCATION_FRACTIONS = ['Monosome', 'Polysome']
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
    'padj',
    'significant',
]


def _fit_dataset() -> MonoPolyDataSet:
    counts, metadata = load_example_data()

    return MonoPolyDataSet(
        counts=counts,
        metadata=metadata,
        subject='subject',
        condition='condition',
        case='case',
        control='control',
        fraction='fraction',
        fraction_order=FRACTION_ORDER,
        abundance_fraction='Input',
        allocation_fractions=ALLOCATION_FRACTIONS,
        engines=['nb_interaction'],
    ).fit()


def test_leave_one_subject_out_refits_valid_subset_datasets() -> None:
    dataset = _fit_dataset()
    contrast = FractionContrast.redistribution('case', 'control', 'Polysome', 'Monosome')

    detail, summary = leave_one_subject_out(dataset, contrast, engine='nb_interaction')

    assert not detail.empty
    assert not summary.empty
    assert set(detail['omitted_subject']) == set(dataset.metadata['subject'])
    assert summary['largest_absolute_effect_change'].notna().all()


def test_monopoly_result_copies_input_tables_and_metadata() -> None:
    empty_results = pd.DataFrame(columns=RESULT_COLUMNS).set_index('feature_id')
    classification = pd.DataFrame(
        {'regulatory_class': ['uncertain'], 'classification_reason': ['no_significant_effects']},
        index=pd.Index(['feature_001'], name='feature_id')
    )
    metadata = {'case': 'case'}

    result = MonoPolyResult(
        abundance=empty_results,
        fraction_effects={'Monosome': empty_results},
        redistribution=empty_results,
        allocation=empty_results,
        joint=empty_results,
        integrated=classification,
        classification=classification,
        stability=pd.DataFrame(),
        diagnostics=pd.DataFrame(),
        metadata=metadata,
    )

    empty_results['effect'] = 1.0
    classification.loc['feature_001', 'regulatory_class'] = 'changed'
    metadata['case'] = 'changed'

    assert result.abundance.empty
    assert result.fraction_effects['Monosome'].empty
    assert result.classification.loc['feature_001', 'regulatory_class'] == 'uncertain'
    assert result.metadata['case'] == 'case'


def test_summary_returns_independent_result_table() -> None:
    dataset = _fit_dataset()
    contrast = FractionContrast.redistribution('case', 'control', 'Polysome', 'Monosome')
    stats = MonoPolyStats(dataset, contrast=contrast)

    first = stats.summary()
    first['effect'] = 0.0
    second = stats.summary()

    assert not second['effect'].eq(0.0).all()


def test_filter_sensitivity_refits_valid_datasets() -> None:
    from monopolyribo.robustness import filter_sensitivity

    dataset = _fit_dataset()
    contrast = FractionContrast.redistribution('case', 'control', 'Polysome', 'Monosome')

    results = filter_sensitivity(dataset, contrast, engine='nb_interaction', min_counts=[5, 10])

    assert not results.empty
    assert set(results['min_count']) == {5, 10}
    assert results['effect'].notna().any()

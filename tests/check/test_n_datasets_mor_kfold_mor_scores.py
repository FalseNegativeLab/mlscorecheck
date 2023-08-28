"""
Testing the checking of scores on multiple datasets using kfold with
mean-of-ratios aggregation over the folds and mean-of-ratios aggregation
over the datasets
"""

import pytest

from mlscorecheck.check import check_n_datasets_mor_kfold_mor_scores
from mlscorecheck.aggregated import (Experiment,
                                    generate_experiment_specification)

def test_consistency():
    """
    Testing with a consistent setup
    """

    experiment_spec = generate_experiment_specification(aggregation='mor',
                                                        aggregation_ds='mor',
                                                        random_state=5)
    experiment = Experiment(**experiment_spec)
    sample = experiment.sample(random_state=5)
    scores = sample.calculate_scores(rounding_decimals=4)

    result = check_n_datasets_mor_kfold_mor_scores(datasets=experiment.to_dict()['datasets'],
                                                    scores=scores,
                                                    eps=1e-4)

    assert not result['inconsistency']

def test_failure():
    """
    Testing with an inconsistent setup
    """
    experiment_spec = generate_experiment_specification(aggregation='mor',
                                                        aggregation_ds='mor',
                                                        random_state=5)
    experiment = Experiment(**experiment_spec)
    scores = {'acc': 0.9, 'sens': 0.3, 'spec': 0.5, 'f1': 0.1}

    result = check_n_datasets_mor_kfold_mor_scores(datasets=experiment.to_dict()['datasets'],
                                                scores=scores,
                                                eps=1e-4)

    assert result['inconsistency']

def test_consistency_bounds():
    """
    Testing with a consistent setup and bounds
    """
    experiment_spec = generate_experiment_specification(aggregation='mor',
                                                        aggregation_ds='mor',
                                                        random_state=5)
    experiment = Experiment(**experiment_spec)
    sample = experiment.sample(random_state=5)
    bounds = sample.get_dataset_fold_bounds(feasible=True)
    experiment = experiment.add_dataset_fold_bounds(bounds)
    scores = sample.calculate_scores(rounding_decimals=4)

    result = check_n_datasets_mor_kfold_mor_scores(datasets=experiment.to_dict()['datasets'],
                                                scores=scores,
                                                eps=1e-4)

    assert not result['inconsistency']

def test_failure_bounds():
    """
    Testing with a inconsistent setup and bounds
    """
    experiment_spec = generate_experiment_specification(aggregation='mor',
                                                        aggregation_ds='mor',
                                                        random_state=5)
    experiment = Experiment(**experiment_spec)
    sample = experiment.sample(random_state=5)
    bounds = sample.get_dataset_fold_bounds(feasible=True)
    experiment = experiment.add_dataset_fold_bounds(bounds)
    scores = {'npv': 0.1, 'ppv': 0.1, 'f1': 0.9}

    result = check_n_datasets_mor_kfold_mor_scores(datasets=experiment.to_dict()['datasets'],
                                                scores=scores,
                                                eps=1e-4)

    assert result['message'] \
            == 'no scores suitable for aggregated consistency checks'

def test_exception():
    """
    Testing the throwing of an exception
    """

    with pytest.raises(ValueError):
        check_n_datasets_mor_kfold_mor_scores(datasets=[{'aggregation': 'rom'}],
                                                scores={},
                                                eps=1e-4)

def test_success_with_no_folding():
    """
    Testing the successful execution with folding specified
    """

    datasets = [{'p': 5, 'n': 10, 'n_folds': 3, 'n_repeats': 5, 'aggregation': 'mor'},
                {'p': 4, 'n': 20, 'n_folds': 5, 'n_repeats': 8, 'aggregation': 'mor'}]

    with pytest.raises(ValueError):
        Experiment(datasets=datasets, aggregation='mor')

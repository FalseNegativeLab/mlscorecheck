"""
Testing the checking of scores on multiple datasets using kfold with
ratio-of-means aggregation on both levels
"""

import pytest

from mlscorecheck.check import check_n_datasets_rom_kfold_rom_scores
from mlscorecheck.aggregated import (Experiment,
                                    generate_experiment_specification)

def test_consistency():
    """
    Testing with a consistent setup
    """

    experiment_spec = generate_experiment_specification(aggregation='rom',
                                                        aggregation_ds='rom',
                                                        random_state=5)
    experiment = Experiment(**experiment_spec)
    sample = experiment.sample(random_state=5)
    scores = sample.calculate_scores(rounding_decimals=4)

    result = check_n_datasets_rom_kfold_rom_scores(datasets=experiment.to_dict()['datasets'],
                                                    scores=scores,
                                                    eps=1e-4)

    assert not result['inconsistency']
    assert result['individual_results'] is not None
    assert 'aggregated_results' not in result

def test_failure():
    """
    Testing with an inconsistent setup
    """
    experiment_spec = generate_experiment_specification(aggregation='rom',
                                                        aggregation_ds='rom',
                                                        random_state=5)
    experiment = Experiment(**experiment_spec)
    scores = {'acc': 0.9, 'sens': 0.3, 'spec': 0.5, 'f1': 0.1}

    result = check_n_datasets_rom_kfold_rom_scores(datasets=experiment.to_dict()['datasets'],
                                                scores=scores,
                                                eps=1e-4)

    assert result['inconsistency']
    assert result['individual_results'] is not None
    assert 'aggregated_results' not in result

def test_consistency_bounds():
    """
    Testing with a consistent setup and bounds
    """
    experiment_spec = generate_experiment_specification(aggregation='rom',
                                                        aggregation_ds='rom',
                                                        random_state=5)
    experiment = Experiment(**experiment_spec)
    sample = experiment.sample(random_state=5)
    bounds = sample.get_dataset_fold_bounds(feasible=True)
    experiment = experiment.add_dataset_fold_bounds(bounds)
    scores = sample.calculate_scores(rounding_decimals=4)

    result = check_n_datasets_rom_kfold_rom_scores(datasets=experiment.to_dict()['datasets'],
                                                scores=scores,
                                                eps=1e-4)

    assert not result['inconsistency']
    assert result['individual_results'] is not None
    assert result['aggregated_results'] is not None

def test_failure_bounds():
    """
    Testing with a inconsistent setup and bounds
    """
    experiment_spec = generate_experiment_specification(aggregation='rom',
                                                        aggregation_ds='rom',
                                                        random_state=5)
    experiment = Experiment(**experiment_spec)

    sample = experiment.sample(random_state=5)
    bounds = sample.get_dataset_fold_bounds(feasible=True)
    experiment = experiment.add_dataset_fold_bounds(bounds)
    scores = {'npv': 0.1, 'ppv': 0.1, 'f1': 0.9}

    with pytest.warns():
        result = check_n_datasets_rom_kfold_rom_scores(datasets=experiment.to_dict()['datasets'],
                                                scores=scores,
                                                eps=1e-4)
    assert result['inconsistency']
    assert result['individual_results'] is not None
    assert result['aggregated_results']['message'] \
            == 'no scores suitable for aggregated consistency checks'

def test_exception():
    """
    Testing the throwing of an exception
    """

    with pytest.raises(ValueError):
        check_n_datasets_rom_kfold_rom_scores(datasets=[{'aggregation': 'mor'}],
                                                scores={},
                                                eps=1e-4)

def test_success_with_no_folding():
    """
    Testing the successful execution with folding specified
    """

    datasets = [{'p': 5, 'n': 10, 'n_folds': 3, 'n_repeats': 5, 'aggregation': 'rom'},
                {'p': 4, 'n': 20, 'n_folds': 5, 'n_repeats': 8, 'aggregation': 'rom'}]
    experiment = Experiment(datasets=datasets, aggregation='rom')
    sample = experiment.sample()
    scores = sample.calculate_scores(rounding_decimals=4)

    result = check_n_datasets_rom_kfold_rom_scores(datasets=datasets,
                                                    scores=scores,
                                                    eps=1e-4)

    assert not result['inconsistency']
    assert result['individual_results'] is not None
    assert 'aggregated_results' not in result

"""
Testing the checking of scores on 1 dataset using kfold with mean of ratios
aggregation
"""

import pytest

from mlscorecheck.check import check_1_dataset_kfold_mor_scores
from mlscorecheck.aggregated import (Dataset,
                                        generate_dataset_specification)

def test_consistency():
    """
    Testing with a consistent setup
    """
    dataset_spec = generate_dataset_specification(aggregation='mor',
                                                    random_state=5)
    dataset = Dataset(**dataset_spec) # pylint: disable=missing-kwoa
    sample = dataset.sample()
    scores = sample.calculate_scores(rounding_decimals=4)

    result = check_1_dataset_kfold_mor_scores(dataset=dataset_spec,
                                                scores=scores,
                                                eps=1e-4)

    assert not result['inconsistency']

def test_failure():
    """
    Testing with an inconsistent setup
    """
    dataset_spec = generate_dataset_specification(aggregation='mor',
                                                    random_state=5)
    scores = {'acc': 0.9, 'sens': 0.3, 'spec': 0.5, 'f1': 0.1}

    result = check_1_dataset_kfold_mor_scores(dataset=dataset_spec,
                                                scores=scores,
                                                eps=1e-4)

    assert result['inconsistency']

def test_consistency_aggregated():
    """
    Testing with a consistent setup and bounds
    """
    dataset_spec = generate_dataset_specification(aggregation='mor',
                                                    random_state=5)
    dataset = Dataset(**dataset_spec)  # pylint: disable=missing-kwoa
    sample = dataset.sample()
    dataset = dataset.add_fold_bounds(sample.get_fold_bounds(feasible=True))
    scores = sample.calculate_scores(rounding_decimals=4)

    result = check_1_dataset_kfold_mor_scores(dataset=dataset.to_dict(),
                                                scores=scores,
                                                eps=1e-4)

    assert not result['inconsistency']

def test_failure_aggregated():
    """
    Testing with a inconsistent setup and bounds
    """
    dataset_spec = generate_dataset_specification(aggregation='mor',
                                                    random_state=5)
    dataset = Dataset(**dataset_spec)  # pylint: disable=missing-kwoa
    sample = dataset.sample()
    dataset = dataset.add_fold_bounds(sample.get_fold_bounds(feasible=True))
    scores = {'npv': 0.1, 'ppv': 0.1, 'f1': 0.9}

    result = check_1_dataset_kfold_mor_scores(dataset=dataset.to_dict(),
                                                scores=scores,
                                                eps=1e-4)

    assert not result['inconsistency']

def test_exception():
    """
    Testing the throwing of an exception
    """

    with pytest.raises(ValueError):
        check_1_dataset_kfold_mor_scores(dataset={'aggregation': 'rom'},
                                            scores={},
                                            eps=1e-4)

def test_success_with_no_folding():
    """
    Testing the successful execution with folding specified
    """

    dataset_spec = {'p': 5, 'n': 10, 'n_folds': 3, 'n_repeats': 5}

    with pytest.raises(ValueError):
        check_1_dataset_kfold_mor_scores(dataset=dataset_spec,
                                        scores={'acc': 0.95},
                                        eps=1e-4)

    dataset_spec = {'p': 5, 'n': 10, 'n_folds': 3, 'n_repeats': 5,
                    'score_bounds': {'acc': (0, 1)}}

    with pytest.raises(ValueError):
        check_1_dataset_kfold_mor_scores(dataset=dataset_spec,
                                        scores={'acc': 0.95},
                                        eps=1e-4)

"""
Testing the checking of scores on 1 dataset using kfold with mean of ratios
aggregation
"""

import pytest

from mlscorecheck.check import check_1_dataset_known_folds_mor_scores
from mlscorecheck.aggregated import (generate_evaluation)

@pytest.mark.parametrize('random_seed', list(range(10)))
@pytest.mark.parametrize('rounding_decimals', [3, 4])
def test_consistency(random_seed: int, rounding_decimals: int):
    """
    Testing with a consistent setup

    Args:
        random_seed (int): the random seed to use
        rounding_decimals (int): the number of decimals to round to
    """
    evaluation, scores = generate_evaluation(aggregation='mor',
                                            random_state=random_seed,
                                            return_scores=True,
                                            rounding_decimals=rounding_decimals)

    result = check_1_dataset_known_folds_mor_scores(dataset=evaluation['dataset'],
                                                folding=evaluation['folding'],
                                                scores=scores,
                                                eps=10**(-rounding_decimals))

    assert not result['inconsistency']

@pytest.mark.parametrize('random_seed', list(range(10)))
@pytest.mark.parametrize('rounding_decimals', [3, 4])
def test_failure(random_seed: int, rounding_decimals: int):
    """
    Testing with an inconsistent setup

    Args:
        random_seed (int): the random seed to use
        rounding_decimals (int): the number of decimals to round to
    """
    evaluation, scores = generate_evaluation(aggregation='mor',
                                            random_state=random_seed,
                                            rounding_decimals=rounding_decimals,
                                            return_scores=True)
    scores = {'acc': 0.9, 'sens': 0.3, 'spec': 0.5, 'f1': 0.1}

    result = check_1_dataset_known_folds_mor_scores(dataset=evaluation['dataset'],
                                                folding=evaluation['folding'],
                                                scores=scores,
                                                eps=10**(-rounding_decimals))

    assert result['inconsistency']

@pytest.mark.parametrize('random_seed', list(range(10)))
@pytest.mark.parametrize('rounding_decimals', [3, 4])
def test_consistency_bounds(random_seed: int, rounding_decimals: int):
    """
    Testing with a consistent setup and bounds

    Args:
        random_seed (int): the random seed to use
        rounding_decimals (int): the number of decimals to round to
    """
    evaluation, scores = generate_evaluation(aggregation='mor',
                                            random_state=random_seed,
                                            return_scores=True,
                                            feasible_fold_score_bounds=True,
                                            rounding_decimals=rounding_decimals)

    result = check_1_dataset_known_folds_mor_scores(dataset=evaluation['dataset'],
                                                folding=evaluation['folding'],
                                                fold_score_bounds=evaluation['fold_score_bounds'],
                                                scores=scores,
                                                eps=10**(-rounding_decimals),
                                                timeout=1)

    assert not result['inconsistency']

@pytest.mark.parametrize('random_seed', list(range(10)))
@pytest.mark.parametrize('rounding_decimals', [3, 4])
def test_failure_bounds(random_seed: int, rounding_decimals: int):
    """
    Testing with a inconsistent setup and bounds

    Args:
        random_seed (int): the random seed to use
        rounding_decimals (int): the number of decimals to round to
    """
    evaluation, scores = generate_evaluation(aggregation='mor',
                                            random_state=random_seed,
                                            return_scores=True,
                                            feasible_fold_score_bounds=False,
                                            rounding_decimals=rounding_decimals)
    scores = {'acc': 0.9, 'bacc': 0.1, 'sens': 0.1, 'npv': 0.1, 'ppv': 0.1, 'f1': 0.9}

    result = check_1_dataset_known_folds_mor_scores(dataset=evaluation['dataset'],
                                                folding=evaluation['folding'],
                                                fold_score_bounds=evaluation['fold_score_bounds'],
                                                scores=scores,
                                                eps=10**(-rounding_decimals),
                                                timeout=1)

    assert result['inconsistency'] or result['lp_status'] == 'timeout'

@pytest.mark.skip('dummy')
def test_exception():
    """
    Testing the throwing of an exception
    """

    with pytest.raises(ValueError):
        check_1_dataset_known_folds_mor_scores(evaluation={'aggregation': 'rom'},
                                            scores={},
                                            eps=1e-4)

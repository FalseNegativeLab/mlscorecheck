"""
Testing the checking of scores on multiple datasets using kfold with
score-of-means aggregation on both levels
"""

import pytest

from mlscorecheck.check import check_n_datasets_som_kfold_som_scores
from mlscorecheck.aggregated import (generate_experiment)

@pytest.mark.parametrize('random_seed', list(range(10)))
@pytest.mark.parametrize('rounding_decimals', [3, 4])
def test_consistency(random_seed: int, rounding_decimals: int):
    """
    Testing with a consistent setup

    Args:
        random_seed (int): the random seed to use
        rounding_decimals (int): the number of decimal places to round to
    """

    experiment, scores = generate_experiment(aggregation='som',
                                            evaluation_params={'aggregation': 'som'},
                                            random_state=random_seed,
                                            return_scores=True,
                                            rounding_decimals=rounding_decimals)

    result = check_n_datasets_som_kfold_som_scores(evaluations=experiment['evaluations'],
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
        rounding_decimals (int): the number of decimal places to round to
    """

    experiment, scores = generate_experiment(aggregation='som',
                                            evaluation_params={'aggregation': 'som'},
                                            random_state=random_seed,
                                            return_scores=True,
                                            rounding_decimals=rounding_decimals)

    scores = {'acc': 0.9, 'sens': 0.3, 'spec': 0.5, 'f1': 0.1}

    result = check_n_datasets_som_kfold_som_scores(evaluations=experiment['evaluations'],
                                                    scores=scores,
                                                    eps=10**(-rounding_decimals))

    assert result['inconsistency']

def test_exception():
    """
    Testing the throwing of an exception
    """

    with pytest.raises(ValueError):
        check_n_datasets_som_kfold_som_scores(evaluations=[{'aggregation': 'mos'}],
                                                scores={},
                                                eps=1e-4)

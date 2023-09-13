"""
Testing the checking of scores on 1 dataset using kfold with RoM aggregation
"""

import pytest

from mlscorecheck.check import check_1_dataset_kfold_rom_scores
from mlscorecheck.aggregated import (generate_evaluation)

@pytest.mark.parametrize('random_seed', list(range(10)))
@pytest.mark.parametrize('rounding_decimals', [3, 4])
def test_consistency(random_seed, rounding_decimals):
    """
    Testing with a consistent setup

    Args:
        random_seed (int): the random seed to use
        rounding_decimals (int): the number of decimals to round to
    """
    evaluation, scores = generate_evaluation(aggregation='rom',
                                            random_state=random_seed,
                                            return_scores=True,
                                            rounding_decimals=rounding_decimals)

    result = check_1_dataset_kfold_rom_scores(evaluation=evaluation,
                                                scores=scores,
                                                eps=10**(-rounding_decimals))

    assert not result['inconsistency']

@pytest.mark.parametrize('random_seed', list(range(10)))
@pytest.mark.parametrize('rounding_decimals', [3, 4])
def test_failure(random_seed, rounding_decimals):
    """
    Testing with an inconsistent setup

    Args:
        random_seed (int): the random seed to use
        rounding_decimals (int): the number of decimals to round to
    """
    evaluation, scores = generate_evaluation(aggregation='rom',
                                            random_state=random_seed,
                                            rounding_decimals=rounding_decimals,
                                            return_scores=True)
    scores = {'acc': 0.9, 'sens': 0.3, 'spec': 0.5, 'f1': 0.1}

    result = check_1_dataset_kfold_rom_scores(evaluation=evaluation,
                                                scores=scores,
                                                eps=10**(-rounding_decimals))

    assert result['inconsistency']

def test_exception():
    """
    Testing the throwing of an exception
    """

    with pytest.raises(ValueError):
        check_1_dataset_kfold_rom_scores(evaluation={'aggregation': 'mor'},
                                            scores={},
                                            eps=1e-4)

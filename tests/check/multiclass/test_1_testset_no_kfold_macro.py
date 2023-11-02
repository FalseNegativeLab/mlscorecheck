"""
This module tests the 1 testset no kfold macro averaging consistency test
"""

import pytest

from mlscorecheck.check.multiclass import check_1_testset_no_kfold_macro
from mlscorecheck.individual import (generate_multiclass_dataset,
                                        sample_multiclass_dataset)
from mlscorecheck.scores import calculate_multiclass_scores

@pytest.mark.parametrize('random_seed', list(range(10)))
def test_consistent_configuration(random_seed):
    """
    Testing a consistent configuration

    Args:
        random_seed (int): the random seed to use
    """

    dataset = generate_multiclass_dataset(random_state=random_seed)
    confm = sample_multiclass_dataset(dataset=dataset,
                                        random_state=random_seed)

    scores = calculate_multiclass_scores(confm,
                                            average='macro',
                                            additional_symbols={'beta_positive': 2,
                                                                'beta_negative': 2},
                                            rounding_decimals=4)

    scores = scores | {'beta_positive': 2, 'beta_negative': 2}

    result = check_1_testset_no_kfold_macro(testset=dataset,
                                            scores=scores,
                                            eps=1e-4)

    assert not result['inconsistency']

@pytest.mark.parametrize('random_seed', list(range(10)))
def test_inconsistent_configuration(random_seed):
    """
    Testing a consistent configuration

    Args:
        random_seed (int): the random seed to use
    """

    dataset = generate_multiclass_dataset(random_state=random_seed)
    confm = sample_multiclass_dataset(dataset=dataset,
                                        random_state=random_seed)

    scores = calculate_multiclass_scores(confm,
                                            average='macro',
                                            additional_symbols={'beta_positive': 2,
                                                                'beta_negative': 2},
                                            rounding_decimals=4)

    scores = scores | {'beta_positive': 2, 'beta_negative': 2}
    scores['acc'] = (1.0 + scores['spec']) / 2.0

    result = check_1_testset_no_kfold_macro(testset=dataset,
                                            scores=scores,
                                            eps=1e-4)

    assert result['inconsistency']

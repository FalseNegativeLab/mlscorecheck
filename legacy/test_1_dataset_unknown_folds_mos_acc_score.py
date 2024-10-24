"""
This module tests the accuracy score testing functionality in the kfold MoS case
with unknown folds (only k and the number of repetitions known)
"""

import pytest

import numpy as np

from mlscorecheck.check.binary import check_1_dataset_unknown_folds_mos_acc_score
from mlscorecheck.aggregated import (generate_dataset,
                                        Evaluation, Folding)

@pytest.mark.parametrize('random_seed', range(20))
def test_consistency(random_seed: int):
    """
    Testing with a consistent setup

    Args:
        random_seed (int): the random seed to use
    """
    random_state = np.random.RandomState(random_seed)
    evaluation = Evaluation(dataset=generate_dataset(random_state=random_seed,
                                                        max_p=100,
                                                        max_n=100),
                            folding=Folding(n_folds=random_state.randint(1, 5),
                                            n_repeats=random_state.randint(1, 5),
                                            strategy='stratified_sklearn').to_dict(),
                            aggregation='mos')

    scores = evaluation.sample_figures().calculate_scores(rounding_decimals=4)

    result = check_1_dataset_unknown_folds_mos_acc_score(evaluation.to_dict(),
                                        scores['acc'],
                                        1e-4)
    assert not result['inconsistency']

@pytest.mark.parametrize('random_seed', range(20))
def test_failure(random_seed: int):
    """
    Testing with an inconsistent setup

    Args:
        random_seed (int): the random seed to use
    """
    random_state = np.random.RandomState(random_seed)
    evaluation = Evaluation(dataset=generate_dataset(random_state=random_seed,
                                                        max_p=100,
                                                        max_n=100),
                            folding=Folding(n_folds=random_state.randint(1, 5),
                                            n_repeats=random_state.randint(1, 5),
                                            strategy='stratified_sklearn').to_dict(),
                            aggregation='mos')

    result = check_1_dataset_unknown_folds_mos_acc_score(evaluation.to_dict(),
                                        0.12345678,
                                        1e-8,
                                        numerical_tolerance=1e-9)

    assert result['inconsistency']

def test_exception():
    """
    Testing if the exception is thrown
    """
    with pytest.raises(ValueError):
        check_1_dataset_unknown_folds_mos_acc_score({'folding': {'folds': []}},
                                    0.12345678,
                                    1e-8,
                                    numerical_tolerance=1e-9)

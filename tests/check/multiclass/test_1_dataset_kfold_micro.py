"""
This module tests the 1 dataset and kfold micro averaging consistency test
"""

import pytest

from mlscorecheck.check.multiclass import (check_1_dataset_kfold_micro,
                                            _prepare_1_dataset_kfold_micro)
from mlscorecheck.individual import (generate_multiclass_dataset,
                                        sample_multiclass_dataset,
                                        create_confusion_matrix)
from mlscorecheck.scores import (multiclass_score_map, calculate_multiclass_scores)

def test_preparation():
    dataset = {0: 10, 1: 100, 2: 80}
    folding = {'n_repeats': 2, 'folds': 'dummy'}

    with pytest.raises(ValueError):
        _prepare_1_dataset_kfold_micro(dataset, folding)

    folding = {'folds': [{0: 5, 1: 50, 2: 40},
                            {0: 5, 1: 50, 2: 40}]}

    folds = _prepare_1_dataset_kfold_micro(dataset, folding)

    assert len(folds) > 0

    folding = {}

    folds = _prepare_1_dataset_kfold_micro(dataset, folding)

    assert len(folds) > 0

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
                                            average='micro',
                                            additional_symbols={'beta_positive': 2,
                                                                'beta_negative': 2},
                                            rounding_decimals=4)

    scores = scores | {'beta_positive': 2, 'beta_negative': 2}

    print(dataset)
    print(scores)

    result = check_1_testset_no_kfold_micro(testset=dataset,
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
                                            average='micro',
                                            additional_symbols={'beta_positive': 2,
                                                                'beta_negative': 2},
                                            rounding_decimals=4)

    scores = scores | {'beta_positive': 2, 'beta_negative': 2}
    scores['acc'] = (1.0 + scores['spec']) / 2.0

    result = check_1_testset_no_kfold_micro(testset=dataset,
                                            scores=scores,
                                            eps=1e-4)

    assert result['inconsistency']

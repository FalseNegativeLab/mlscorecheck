"""
This module tests the multiclass scores
"""

import pytest

import numpy as np

from sklearn.metrics import precision_recall_fscore_support

from mlscorecheck.core import safe_call
from mlscorecheck.scores import (multiclass_score_map,
                                    multiclass_score)
from mlscorecheck.individual import (generate_multiclass_dataset,
                                        sample_multiclass_dataset,
                                        create_confusion_matrix)

@pytest.mark.parametrize('score', list(multiclass_score_map.keys()))
@pytest.mark.parametrize('average', ['micro', 'macro', 'weighted'])
@pytest.mark.parametrize('random_seed', list(range(5)))
def test_multiclass_scores(score, average, random_seed):
    """
    Testing a multiclass score

    Args:
        score (str): the score to test
        average (str): the averaging to be used
        random_seed (int): the random seed to be used
    """
    random_state = np.random.RandomState(random_seed)
    dataset = generate_multiclass_dataset(random_state=random_seed)
    matrix = sample_multiclass_dataset(dataset=dataset, random_state=random_seed)

    params = {'confusion_matrix': matrix,
                'beta_positive': 2,
                'beta_negative': 2,
                'average': average}

    score_matrix = safe_call(multiclass_score_map[score], params)

    assert isinstance(score_matrix, float)

    permutation_mask = list(range(len(matrix)))
    random_state.shuffle(permutation_mask)

    shuffled_matrix = matrix[permutation_mask][:, permutation_mask]

    params = {'confusion_matrix': shuffled_matrix,
                'beta_positive': 2,
                'beta_negative': 2,
                'average': average}

    score_shuffled = safe_call(multiclass_score_map[score], params)

    assert abs(score_matrix - score_shuffled) < 1e-8

@pytest.mark.parametrize('average', ['micro', 'macro', 'weighted'])
@pytest.mark.parametrize('random_seed', list(range(5)))
def test_compare_to_sklearn(average, random_seed):
    """
    Comparing sensitivity, positive predictive value and f1 to sklearn

    Args:
        average (str): the averaging to be used
        random_seed (int): the random seed to be used
    """

    random_state = np.random.RandomState(random_seed)
    y_true = random_state.randint(0, 5, size=100)
    y_pred = random_state.randint(0, 5, size=100)

    sample = create_confusion_matrix(y_true, y_pred)

    ppv, sens, f1p, _ = precision_recall_fscore_support(y_true, y_pred, average=average)

    sens_score = multiclass_score_map['sens'](confusion_matrix=sample, average=average)
    ppv_score = multiclass_score_map['ppv'](confusion_matrix=sample, average=average)
    f1_score = multiclass_score_map['f1p'](confusion_matrix=sample, average=average)

    assert abs(sens - sens_score) < 1e-8
    assert abs(ppv - ppv_score) < 1e-8
    assert abs(f1p - f1_score) < 1e-8

def test_exception():
    """
    Testing the exception throwing
    """

    with pytest.raises(ValueError):
        multiclass_score(None, None, 'dummy')

"""
This module tests the rounding of scores
"""

import numpy as np

from mlscorecheck.core import round_scores

def test_round_scores():
    """
    Testing the rounding of scores
    """

    scores = {'acc': 0.123456, 'sens': 0.654321}

    rounded = round_scores(scores, 3)

    assert np.round(scores['acc'], 3) == rounded['acc']
    assert np.round(scores['sens'], 3) == rounded['sens']

    assert round_scores(scores)['acc'] == scores['acc']

    assert round_scores(0.51, 1) == 0.5

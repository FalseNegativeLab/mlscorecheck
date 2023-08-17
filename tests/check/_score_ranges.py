"""
Some helper functions for the testing
"""

import numpy as np

from mlscorecheck.utils import calculate_scores

__all__ = ['calculate_scores_for_folds',
            'score_ranges']

def calculate_scores_for_folds(folds):
    return [calculate_scores(folding, strategy='mor', scores_only=False) for folding in folds]

def score_ranges(folding_scores):
    mins = {}
    maxs = {}
    for folding in folding_scores:
        for key, value in folding.items():
            mins[key] = min(mins.get(key, np.inf), value)
            maxs[key] = max(maxs.get(key, -np.inf), value)

    return {key: (mins[key], maxs[key]) for key in mins}

"""
Some helper functions for the testing
"""

import numpy as np

from mlscorecheck.individual import calculate_scores

__all__ = ['calculate_scores_for_folds',
            'score_ranges',
            'calculate_scores_for_datasets']

def calculate_scores_for_folds(folds):
    return [calculate_scores(folding, scores_only=False) for folding in folds]

def calculate_scores_for_datasets(datasets):
    scores = []
    for dataset in datasets:
        for fold in dataset['folds']:
            scores.append(calculate_scores(fold, scores_only=False))
    return scores

def score_ranges(folding_scores):
    mins = {}
    maxs = {}
    for folding in folding_scores:
        for key, value in folding.items():
            mins[key] = min(mins.get(key, np.inf), value)
            maxs[key] = max(maxs.get(key, -np.inf), value)

    return {key: (mins[key], maxs[key]) for key in mins if key in ['acc', 'sens', 'spec', 'bacc', 'tp', 'tn']}

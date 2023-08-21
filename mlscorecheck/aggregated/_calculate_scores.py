"""
This module implements score calculation for aggregated problems
"""

import copy

from ..individual import (calculate_scores,
                            round_scores)
from ..scores import score_functions_with_solutions

__all__ = ['calculate_scores_list',
            'calculate_scores_dataset',
            'calculate_scores_datasets']

def calculate_scores_list(problems,
                                *,
                                strategy,
                                rounding_decimals=None,
                                populate_original=False):
    """
    Calculates the aggregated scores for a set of problems

    Args:
        problems (list(dict)): the specification of problems
        strategy (str): 'mor'/'rom'/'wmor' specifying the mode of aggregation,
                        'mor' standing for the 'mean of ratios' and
                        'rom' standing for the 'ratio of means' and
                        'wmor' standing for the weighted mean of reatios
        rounding_decimals (None/int): the number of digits to round the scores to
        populate_original (bool): whether to populate the original structure with
                                    partial results

    Returns:
        dict: the scores and the total figures
    """

    if not populate_original:
        problems = copy.deepcopy(problems)

    figures = {'tp': 0, 'tn': 0, 'p': 0, 'n': 0}

    for problem in problems:
        for key in figures:
            figures[key] += problem[key]

    if strategy == 'rom':
        scores = calculate_scores(figures, scores_only=True)
    elif strategy in ('mor', 'wmor'):
        scores = {key: 0.0 for key in score_functions_with_solutions}

        for key in scores:
            total_weight = 0.0
            for problem in problems:
                if strategy == 'mor':
                    scores[key] += problem[key] if problem[key] is not None else 0
                    total_weight += 1
                elif strategy == 'wmor':
                    scores[key] += problem[key] * (problem['p'] + problem['n']) if problem[key] is not None else 0
                    total_weight += (problem['p'] + problem['n'])
            scores[key] = scores[key] / total_weight

    scores = round_scores(scores, rounding_decimals)

    return {**figures, **scores}

def calculate_scores_dataset(dataset,
                                strategy,
                                rounding_decimals=None,
                                populate_original=False,
                                return_populated=False):
    if not populate_original:
        dataset = copy.deepcopy(dataset)

    for fold in dataset['folds']:
        scores = calculate_scores(fold)
        for key in scores:
            fold[key] = scores[key]

    total_scores = calculate_scores_list(dataset['folds'],
                                            strategy=strategy,
                                            populate_original=populate_original)
    for key in total_scores:
        dataset[key] = total_scores[key]

    total_scores = round_scores(total_scores, rounding_decimals)

    return (total_scores, dataset) if return_populated else total_scores

def calculate_scores_datasets(datasets,
                                    *,
                                    strategy,
                                    rounding_decimals=None,
                                    populate_original=False,
                                    return_populated=False
                                    ):
    """
    Calculates scores for multiple datasets

    Args:
        problems (list(dict)): the specification of problems
        strategy (str, str): 2-item iterable of 'mor'/'rom'/'wmor' specifying the mode of
                                aggregation for each level,
                                'mor' standing for the 'mean of ratios' and
                                'rom' standing for the 'ratio of means' and
                                'wmor' standing for the 'weighted mean of ratios'
        rounding_decimals (None/int): the number of digits to round the scores to
        populate_original (bool): whether to populate the original structure with
                                    partial results
        return_populated (bool): if True, returns the problem structure populated with figures

    Returns:
        dict (, list): the scores and optionally the populated problem structure
    """
    if not populate_original:
        datasets = copy.deepcopy(datasets)

    for dataset in datasets:
        calculate_scores_dataset(dataset,
                                    strategy=strategy[1],
                                    populate_original=True)

    scores = calculate_scores_list(datasets,
                                    strategy=strategy[0],
                                    populate_original=populate_original)

    scores = round_scores(scores, rounding_decimals)

    return (scores, datasets) if return_populated else scores

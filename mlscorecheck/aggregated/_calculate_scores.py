"""
This module implements score calculation for aggregated problems
"""

import copy

from ..individual import (calculate_scores,
                            round_scores,
                            score_functions_with_solutions)

__all__ = ['calculate_scores_aggregated',
            'calculate_scores_datasets']

def calculate_scores_aggregated(problems,
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
        dict: the scores and the total figures
    """
    if not populate_original:
        datasets = copy.deepcopy(datasets)

    for problem in datasets:
        for fold in problem['folds']:
            scores = calculate_scores(fold)
            for key in scores:
                fold[key] = scores[key]

        total_scores = calculate_scores_aggregated(problem['folds'],
                                                    strategy=strategy[1],
                                                    populate_original=populate_original)
        for key in total_scores:
            problem[key] = total_scores[key]

    scores = calculate_scores_aggregated(datasets,
                                            strategy=strategy[0],
                                            populate_original=populate_original)

    scores = round_scores(scores, rounding_decimals)

    return (scores, datasets) if return_populated else scores

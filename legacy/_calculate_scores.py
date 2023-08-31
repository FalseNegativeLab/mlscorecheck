"""
This module implements score calculation for aggregated problems
"""

import copy

from ..individual import (calculate_scores,
                            round_scores)
from ..scores import score_functions_with_solutions

from ._folds import _expand_datasets

__all__ = ['calculate_scores_list',
            'calculate_scores_dataset',
            'calculate_scores_datasets']

def calculate_scores_list(problems,
                                *,
                                strategy,
                                rounding_decimals=None,
                                return_populated=False):
    """
    Calculates the aggregated scores for a set of problems

    Args:
        problems (list(dict)): the specification of problems
        strategy (str): 'mor'/'rom' specifying the mode of aggregation,
                        'mor' standing for the 'mean of ratios' and
                        'rom' standing for the 'ratio of means'
        rounding_decimals (None|int): the number of digits to round the scores to
        return_populated (bool): whether to return the populated structure

    Returns:
        dict: the scores and the total figures
    """

    problems = copy.deepcopy(problems)

    figures = {'tp': 0, 'tn': 0, 'p': 0, 'n': 0}

    for problem in problems:
        for key in figures:
            figures[key] += problem[key]

    if strategy == 'rom':
        scores = calculate_scores(figures, scores_only=True)
    elif strategy == 'mor':
        scores = {key: 0.0 for key in score_functions_with_solutions}
        for key in scores:
            total_weight = 0.0
            for problem in problems:
                scores[key] += problem[key] if problem[key] is not None else 0
                total_weight += 1

            scores[key] = scores[key] / total_weight

    scores = round_scores(scores, rounding_decimals)
    scores = {**figures, **scores}

    return (scores, problems) if return_populated else scores

def calculate_scores_dataset(dataset,
                                strategy,
                                rounding_decimals=None,
                                return_populated=False):
    """
    Calculates all scores for a dataset

    dataset (dict): the specification of a dataset
    strategy (str): 'mor'/'rom' specifying the mode
                            of aggregation for each level,
                            'mor' standing for the 'mean of ratios' and
                            'rom' standing for the 'ratio of means'
    rounding_decimals (None|int): the number of digits to round the scores to
    return_populated (bool): if True, returns the dataset structure populated
                                with figures

    Returns:
        dict (, list): the scores and optionally the populated dataset structure
    """
    dataset = copy.deepcopy(dataset)

    dataset = _expand_datasets(dataset)

    for fold in dataset['folds']:
        scores = calculate_scores(fold)
        for key in scores:
            fold[key] = scores[key]

    total_scores, dataset['folds'] = calculate_scores_list(dataset['folds'],
                                                            strategy=strategy,
                                                            return_populated=True)
    for key in total_scores:
        dataset[key] = total_scores[key]

    total_scores = round_scores(total_scores, rounding_decimals)

    return (total_scores, dataset) if return_populated else total_scores

def calculate_scores_datasets(datasets,
                                    *,
                                    strategy,
                                    rounding_decimals=None,
                                    return_populated=False
                                    ):
    """
    Calculates scores for multiple datasets

    Args:
        datasets (list(dict)): the specifications of datasets
        strategy (str, str): 2-item iterable of 'mor'/'rom' specifying the mode
                                of aggregation for each level,
                                'mor' standing for the 'mean of ratios' and
                                'rom' standing for the 'ratio of means'
        rounding_decimals (None|int): the number of digits to round the scores to
        return_populated (bool): if True, returns the dataset structures populated
                                    with figures

    Returns:
        dict (, list): the scores and optionally the populated datasets structures
    """
    datasets = copy.deepcopy(datasets)

    datasets = _expand_datasets(datasets)

    datasets = [calculate_scores_dataset(dataset,
                                            strategy=strategy[1],
                                            return_populated=True)[1] for dataset in datasets]

    scores, datasets = calculate_scores_list(datasets,
                                                strategy=strategy[0],
                                                return_populated=True)

    scores = round_scores(scores, rounding_decimals)

    return (scores, datasets) if return_populated else scores

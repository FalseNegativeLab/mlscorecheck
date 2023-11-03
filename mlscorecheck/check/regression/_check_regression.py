"""
This module implements the testing of regression scores
"""

import numpy as np

from ...individual import Interval
from ...core import NUMERICAL_TOLERANCE, round_scores

__all__ = ['check_1_testset_no_kfold',
            'expand_regression_scores',
            'mean_average_error',
            'mean_squared_error',
            'root_mean_squared_error',
            'r_squared',
            'calculate_regression_scores',
            'generate_regression_problem_and_scores',
            'check_relations']

rules = [{'score0': 'mae', 'score1': 'rmse', 'relation': 'le'}]

def mean_average_error(y_true: np.array, y_pred: np.array):
    return np.mean(np.abs(y_true - y_pred))

def mean_squared_error(y_true: np.array, y_pred: np.array):
    return np.mean((y_true - y_pred)**2)

def root_mean_squared_error(y_true: np.array, y_pred: np.array):
    return np.sqrt(np.mean((y_true - y_pred)**2))

def r_squared(y_true: np.array, y_pred: np.array):
    return 1.0 - np.sum((y_true - y_pred)**2) / (np.var(y_true) * y_true.shape[0])

regression_scores = {'mae': mean_average_error,
                        'mse': mean_squared_error,
                        'rmse': root_mean_squared_error,
                        'r2': r_squared}

def calculate_regression_scores(y_true: np.array, y_pred: np.array, subset=None):
    scores = {}

    for key, function in regression_scores.items():
        if subset is None or key in subset:
            scores[key] = function(y_true, y_pred)

    return scores

def generate_regression_problem_and_scores(random_state=None,
                                            max_n_samples=1000,
                                            subset=None,
                                            rounding_decimals=None):
    if not isinstance(random_state, np.random.RandomState):
        random_state = np.random.RandomState(random_state)

    n_samples = random_state.randint(2, max_n_samples)

    y_true = np.random.random_sample(n_samples)
    y_pred = (np.random.random_sample(n_samples) - 0.5) / 20

    scores = calculate_regression_scores(y_true, y_pred, subset)

    scores = round_scores(scores, rounding_decimals)

    return np.var(y_true), n_samples, scores

def expand_regression_scores(var: float,
                                n_samples: int,
                                scores: dict,
                                eps,
                                numerical_tolerance: float) -> dict:
    """
    Generate scores from the ones available and expand the scores to intervals given the
    numerical uncertainty.

    Args:
        var (float): the variance of the evaluation set
        n_samples (int): the number of samples in the evaluation set
        scores (dict(str,float)): the scores to check ('mae', 'rmse', 'mse', 'r2')
        eps (float,dict(str,float)): the numerical uncertainty of the scores
        numerical_tolerance (float): the numerical tolerance of the test

    Returns:
        dict: the extended set of score intervals
    """
    scores = {key: Interval(value - eps - numerical_tolerance,
                            value + eps + numerical_tolerance) for key, value in scores.items()}

    if 'rmse' in scores and 'mse' not in scores:
        scores['mse'] = scores['rmse'] ** 2

    if 'mse' in scores and 'rmse' not in scores:
        scores['rmse'] = scores['mse'] ** 0.5

    if 'r2' in scores and 'mse' not in scores:
        scores['mse'] = (1 - scores['r2']) * var / n_samples

    if 'r2' in scores and 'rmse' not in scores:
        scores['mse'] = ((1 - scores['r2']) * var / n_samples) ** 0.5

    if 'mse' in scores and 'r2' not in scores:
        scores['r2'] = 1 - scores['mse'] * n_samples / var

    if 'rmse' in scores and 'r2' not in scores:
        scores['r2'] = 1 - scores['rmse'] ** 2 * n_samples / var

    return scores

def check_relations(scores: dict) -> dict:
    """
    Check the relations of the scores

    Args:
        scores (dict(str,Interval)): the score and figure intervals

    Returns:
        dict: a summary of the analysis, with the following entries:

            - ``'inconsistency'`` (bool): whether an inconsistency has been identified
            - ``'details'`` (list(dict)): the details of the analysis, with the following entries
    """
    results = {'details': []}

    for rule in rules:
        if rule['score0'] in scores and rule['score1'] in scores:
            score0 = scores[rule['score0']]
            score1 = scores[rule['score1']]
            if rule['relation'] == 'le':
                value = score0.lower_bound <= score1.upper_bound
            results['details'].append(rule | {'value': value})

    results['inconsistency'] = any(not result['value'] for result in results['details'])

    return results

def check_1_testset_no_kfold(var: float,
                                n_samples: int,
                                scores: dict,
                                eps,
                                numerical_tolerance: float = NUMERICAL_TOLERANCE) -> dict:
    """
    The consistency test for regression scores calculated on a single test set
    with no k-folding

    Args:
        var (float): the variance of the evaluation set
        n_samples (int): the number of samples in the evaluation set
        scores (dict(str,float)): the scores to check ('mae', 'rmse', 'mse', 'r2')
        eps (float,dict(str,float)): the numerical uncertainty of the scores
        numerical_tolerance (float): the numerical tolerance of the test

    Returns:
        dict: a summary of the analysis, with the following entries:

            - ``'inconsistency'`` (bool): whether an inconsistency has been identified
            - ``'details'`` (list(dict)): the details of the analysis, with the following entries
    """
    intervals = expand_regression_scores(var,
                                            n_samples,
                                            scores,
                                            eps,
                                            numerical_tolerance)

    return check_relations(intervals)

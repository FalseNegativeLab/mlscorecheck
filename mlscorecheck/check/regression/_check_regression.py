"""
This module implements the testing of regression scores
"""

import numpy as np

from ...individual import Interval
from ...core import NUMERICAL_TOLERANCE, round_scores, safe_eval

__all__ = [
    "check_1_testset_no_kfold",
    "expand_regression_scores",
    "mean_average_error",
    "mean_squared_error",
    "root_mean_squared_error",
    "r_squared",
    "calculate_regression_scores",
    "generate_regression_problem_and_scores",
    "check_relations",
    "score_formulas",
]

rules = [{"score0": "mae", "score1": "rmse", "relation": "le"}]


def mean_average_error(y_true: np.array, y_pred: np.array) -> float:
    """
    The mean average error (MAE) regression performance score

    Args:
        y_true (np.array): the true labels
        y_pred (np.array): the predicted labels

    Returns:
        float: the MAE score
    """
    return np.mean(np.abs(y_true - y_pred))


def mean_squared_error(y_true: np.array, y_pred: np.array) -> float:
    """
    The mean squared error (MSE) regression performance score

    Args:
        y_true (np.array): the true labels
        y_pred (np.array): the predicted labels

    Returns:
        float: the MSE score
    """
    return np.mean((y_true - y_pred) ** 2)


def root_mean_squared_error(y_true: np.array, y_pred: np.array) -> float:
    """
    The root mean squared error (RMSE) regression performance score

    Args:
        y_true (np.array): the true labels
        y_pred (np.array): the predicted labels

    Returns:
        float: the RMSE score
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))


def r_squared(y_true: np.array, y_pred: np.array) -> float:
    """
    The R squared (r2) regression performance score

    Args:
        y_true (np.array): the true labels
        y_pred (np.array): the predicted labels

    Returns:
        float: the R2 score
    """
    return 1.0 - np.sum((y_true - y_pred) ** 2) / (np.var(y_true) * y_true.shape[0])


regression_scores = {
    "mae": mean_average_error,
    "mse": mean_squared_error,
    "rmse": root_mean_squared_error,
    "r2": r_squared,
}


def calculate_regression_scores(
    y_true: np.array, y_pred: np.array, subset=None
) -> dict:
    """
    Calculate the performance scores for a regression problem

    Args:
        y_true (np.array): the true labels
        y_pred (np.array): the predicted labels
        subset (None|list(str)): the scores to calculate

    Returns:
        dict: the calculated scores
    """
    scores = {}

    for key, function in regression_scores.items():
        if subset is None or key in subset:
            scores[key] = function(y_true, y_pred)

    return scores


def generate_regression_problem_and_scores(
    random_state=None, max_n_samples=1000, subset=None, rounding_decimals=None
) -> (float, int, dict):
    """
    Generate a regression problem and corresponding scores

    Args:
        random_state (None|int|np.random.RandomState): the random state/seed to use
        max_n_samples (int): the maximum number of samples to be used
        subset (None|list(str)): the list of scores to be used
        rounding_decimals (None|int): the number of decimals to round to

    Returns:
        float, int, dict: the variance, the number of samples and the scores
    """
    if not isinstance(random_state, np.random.RandomState):
        random_state = np.random.RandomState(random_state)

    n_samples = random_state.randint(2, max_n_samples)

    y_true = np.random.random_sample(n_samples)
    y_pred = y_true + (np.random.random_sample(n_samples) - 0.5) / 10

    scores = calculate_regression_scores(y_true, y_pred, subset)

    scores = round_scores(scores, rounding_decimals)

    return np.var(y_true), n_samples, scores


score_formulas = {
    "mse": {"rmse": "rmse**2", "r2": "((1 - r2) * (var))"},
    "rmse": {"mse": "mse ** 0.5", "r2": "((1 - r2) * (var)) ** 0.5"},
    "r2": {"mse": "(1 - mse / var)", "rmse": "(1 - rmse**2 / var)"},
}


def expand_regression_scores(
    var: float, n_samples: int, scores: dict, eps, numerical_tolerance: float
) -> dict:
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
    scores = {
        key: Interval(
            value - eps - numerical_tolerance, value + eps + numerical_tolerance
        )
        for key, value in scores.items()
    }

    to_add = {}
    for key, value in score_formulas.items():
        if key not in scores:
            for sol, formula in value.items():
                if sol in scores:
                    to_add[key] = safe_eval(
                        formula, scores | {"var": var, "n_samples": n_samples}
                    )
                    break

    return scores | to_add


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
    results = {"details": []}

    for rule in rules:
        if rule["score0"] in scores and rule["score1"] in scores:
            score0 = scores[rule["score0"]]
            score1 = scores[rule["score1"]]
            if rule["relation"] == "le":
                value = score0.lower_bound <= score1.upper_bound
            results["details"].append(rule | {"value": value})

    results["inconsistency"] = any(not result["value"] for result in results["details"])

    return results


def check_1_testset_no_kfold(
    var: float,
    n_samples: int,
    scores: dict,
    eps,
    numerical_tolerance: float = NUMERICAL_TOLERANCE,
) -> dict:
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

    Examples:
        >>> from mlscorecheck.check.regression import check_1_testset_no_kfold
        >>> var = 0.08316192579267838
        >>> n_samples = 100
        >>> scores =  {'mae': 0.0254, 'r2': 0.9897}
        >>> result = check_1_testset_no_kfold(var=var,
                                                n_samples=n_samples,
                                                scores=scores,
                                                eps=1e-4)
        >>> result['inconsistency']
        # False

        >>> scores['mae'] = 0.03
        >>> result = check_1_testset_no_kfold(var=var,
                                            n_samples=n_samples,
                                            scores=scores,
                                            eps=1e-4)
        >>> result['inconsistency']
        # True
    """
    intervals = expand_regression_scores(
        var, n_samples, scores, eps, numerical_tolerance
    )

    return check_relations(intervals)

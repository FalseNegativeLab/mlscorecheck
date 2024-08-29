"""
This module implements some utilities for the AUC related calculations
"""

import numpy as np

from ..aggregated import determine_fold_configurations

__all__ = [
    "R",
    "F",
    "perturb_solutions",
    "multi_perturb_solutions",
    "check_cvxopt",
    "translate_folding",
    "translate_scores",
    "prepare_intervals",
]


def R(  # pylint: disable=invalid-name
    x: float, k: int, lower: np.array = None, upper: np.array = None
) -> np.array:
    """
    The "representative" function

    1 - R(x, k, lower, upper) = F(R(1 - x, k, 1 - F(upper), 1 - F(lower)))

    holds.

    Args:
        x (float): the desired average
        k (int): the number of dimensions
        lower (np.array|None): the lower bounds
        upper (np.array|None): the upper bounds

    Returns:
        np.array: the representative

    Raises:
        ValueError: if the configuration cannot be satisfied
    """
    if lower is None:
        lower = np.repeat(0.0, k)
    if upper is None:
        upper = np.repeat(1.0, k)

    x = x * len(lower)
    if np.sum(lower) > x or np.sum(upper) < x:
        raise ValueError("infeasible configuration")

    solution = lower.copy().astype(float)
    x = x - np.sum(lower)

    idx = 0
    while x > 0 and idx < len(lower):
        if upper[idx] - lower[idx] < x:
            solution[idx] = upper[idx]
            x -= upper[idx] - lower[idx]
        else:
            solution[idx] = solution[idx] + x
            x = 0.0
        idx += 1

    return np.array(solution).astype(float)


def F(x: np.array) -> np.array:  # pylint: disable=invalid-name
    """
    The flipping operator

    Args:
        x (np.array): the vector to flip

    Returns:
        np.array: the flipped vector
    """
    return x[::-1]


def perturb_solutions(
    values: np.array,
    lower_bounds: np.array,
    upper_bounds: np.array,
    random_state: int = None,
) -> np.array:
    """
    Applies a perturbation to a solution, by keeping the lower and upper bound and the sum

    Args:
        values (np.array): the values to perturb
        lower_bounds (np.array): the lower bounds
        upper_bounds (np.array): the upper bounds
        random_state (int|None): the random seed to use

    Returns:
        np.array: the perturbed values
    """
    random_state = np.random.RandomState(random_state)
    greater = np.where(values > lower_bounds)[0]
    lower = np.where(values < upper_bounds)[0]

    greater = random_state.choice(greater)
    lower = random_state.choice(lower)

    diff = min(
        values[greater] - lower_bounds[greater], upper_bounds[lower] - values[lower]
    )
    step = random_state.random_sample() * diff

    values = values.copy()
    values[greater] -= step
    values[lower] += step

    return values


def multi_perturb_solutions(
    n_perturbations: int,
    values: np.array,
    lower_bounds: np.array,
    upper_bounds: np.array,
    random_state: int = None,
) -> np.array:
    """
    Applies a multiple perturbations to a solution vector,
    keeping its average

    Args:
        n_perturbations (int): the number of perturbations
        values (np.array): the values to perturb
        lower_bounds (np.array): the lower bounds
        upper_bounds (np.array): the upper bounds
        random_state (int|None): the random seed to use

    Returns:
        np.array: the perturbed values
    """
    for _ in range(n_perturbations):
        values = perturb_solutions(values, lower_bounds, upper_bounds, random_state)

    return values


def check_cvxopt(results, message):
    """
    Checking the cvxopt results

    Args:
        results (dict): the output of cvxopt
        message (str): the additional message

    Raises:
        ValueError: when the solution is not optimal
    """
    if results["status"] != "optimal":
        raise ValueError(
            "no optimal solution found for the configuration " + f"({message})"
        )


def translate_folding(folding: dict):
    """
    Translates a folding specification into counts of positives and negatives

    Args:
        folding (dict): a folding specification

    Returns:
        np.array, np.array: the numbers of positives and negatives
        in the folds
    """
    folds = determine_fold_configurations(**folding)

    ps = np.array([fold["p"] for fold in folds])
    ns = np.array([fold["n"] for fold in folds])

    return ps, ns


def translate_scores(scores: dict) -> dict:
    """
    Translates the scores

    Args:
        scores (dict): the dict of scores

    Returns:
        dict: the translated scores

    Raises:
        ValueError: when the provided scores are inconsistent
    """
    scores = {**scores}

    if "sens" in scores and "tpr" in scores and scores["sens"] != scores["tpr"]:
        raise ValueError("differing sens and tpr cannot be specified together")

    if "sens" in scores and "fnr" in scores and scores["sens"] != 1 - scores["fnr"]:
        raise ValueError("differing sens and fnr cannot be specified together")

    if "spec" in scores and "tnr" in scores and scores["spec"] != scores["tnr"]:
        raise ValueError("differing spec and tnr cannot be specified together")

    if "spec" in scores and "fpr" in scores and scores["spec"] != 1 - scores["fpr"]:
        raise ValueError("differing spec and fpr cannot be specified together")

    if "fpr" not in scores:
        if "spec" in scores:
            scores["fpr"] = 1 - scores["spec"]
        if "tnr" in scores:
            scores["fpr"] = 1 - scores["tnr"]

    if "tpr" not in scores:
        if "sens" in scores:
            scores["tpr"] = scores["sens"]
        if "fnr" in scores:
            scores["tpr"] = 1 - scores["fnr"]

    return scores


def prepare_intervals(scores: dict, eps: float):
    """
    Create intervals from the values

    Args:
        scores (dict): the scores to transform to intervals
        eps (float): the estimated numerical uncertainty

    Returns:
        dict: the intervals
    """
    return {
        score: (max(scores[score] - eps, 0), min(scores[score] + eps, 1))
        for score in ["tpr", "fpr", "tnr", "fnr", "sens", "spec", "acc", "auc"]
        if score in scores
    }

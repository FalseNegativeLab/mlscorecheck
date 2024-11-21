"""
This module implements some utilities for the AUC related calculations
"""

import numpy as np

from scipy.optimize import minimize_scalar

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from scipy.optimize import root_scalar

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
    "simplify_roc",
    "roc_value_at",
    "average_roc_curves_to_1",
    "average_n_roc_curves_",
    "average_n_roc_curves",
    "generate_roc_curve",
    "generate_roc_curve_slope",
    "generate_1_roc_curve",
    "sample_triangle",
    "p_norm_fit",
    "p_norm_fit_joint",
    "p_norm_fit_best",
    "auc_estimator",
    "p_norm_fit_auc",
    "max_acc_estimator",
    "integrate_roc_curve",
    "integrate_roc_curves",
    "sample1",
    "sample2"
]


def integrate_roc_curve(fprs, tprs):
    """
    Integrates a ROC curve

    Args:
        fprs (np.array): the fpr values
        tprs (np.array): the tpr values

    Returns:
        float: the integral
    """
    diffs = np.diff(fprs)
    avgs = (tprs[:-1] + tprs[1:]) / 2
    return float(np.sum(diffs * avgs))


def integrate_roc_curves(fprs, tprs):
    """
    Integrates ROC curves

    Args:
        fprs (np.array): the fpr values
        tprs (np.array): the tpr values

    Returns:
        float: the integral
    """
    diffs = np.diff(fprs, axis=1)
    avgs = (tprs[:, :-1] + tprs[:, 1:]) / 2
    return (np.sum(diffs * avgs, axis=1)).astype(float)



def generate_roc_curve(
        tpr: float,
        fpr: float,
        n_nodes: int,
        n_samples: int = 1,
        p: int = None,
        n: int = None,
        random_middle = False,
        random_state = None
):
    
    if not isinstance(random_state, np.random.RandomState):
        random_state = np.random.RandomState(random_state)

    tprs = np.full((n_samples, n_nodes), fill_value=-1, dtype=float)
    fprs = np.full((n_samples, n_nodes), fill_value=-1, dtype=float)

    tprs[:, 0] = 0.0
    tprs[:, -1] = 1.0

    fprs[:, 0] = 0.0
    fprs[:, -1] = 1.0

    if random_middle:
        middle_indices = random_state.choice(list(range(1, n_nodes-1)), n_samples)
    else:
        middle_indices = np.repeat(int(n_nodes/2), n_samples)

    tprs[np.arange(0, n_samples), middle_indices] = tpr
    fprs[np.arange(0, n_samples), middle_indices] = fpr

    first_indices = np.repeat(0, n_samples)
    last_indices = middle_indices
    queue = np.vstack((first_indices, last_indices)).T

    first_indices = middle_indices.copy()
    last_indices = np.repeat(n_nodes - 1, n_samples)
    queue = np.vstack([queue, np.vstack((first_indices, last_indices)).T])

    sample_indices = np.hstack([np.arange(0, n_samples), np.arange(0, n_samples)])

    idx = 0
    while queue.shape[0] > 0:

        first_indices = queue[:, 0]
        last_indices = queue[:, 1]

        fpr_ranges = fprs[sample_indices, last_indices] - fprs[sample_indices, first_indices]
        uniforms = random_state.random_sample(fpr_ranges.shape[0])
        fpr_steps = (uniforms * fpr_ranges)

        tpr_ranges = tprs[sample_indices, last_indices] - tprs[sample_indices, first_indices]
        uniforms = random_state.random_sample(tpr_ranges.shape[0])
        tpr_steps = (uniforms * tpr_ranges)

        index_ranges = last_indices - first_indices - 1
        uniforms = random_state.random_sample(index_ranges.shape[0])
        index_steps = np.floor(index_ranges * uniforms).astype(int) + 1

        fprs[sample_indices, first_indices + index_steps] = fprs[sample_indices, first_indices] + fpr_steps
        tprs[sample_indices, first_indices + index_steps] = tprs[sample_indices, first_indices] + tpr_steps

        lhs_queue = np.vstack([first_indices, first_indices + index_steps]).T
        rhs_queue = np.vstack([first_indices + index_steps, last_indices]).T
        queue = np.vstack([lhs_queue, rhs_queue])
        sample_indices = np.hstack([sample_indices, sample_indices])

        mask = queue[:, 0] < queue[:, 1] - 1
        queue = queue[mask]
        sample_indices = sample_indices[mask]
        first_indices = queue[:, 0]
        last_indices = queue[:, 1]

        idx += 1
    
    if p is not None:
        tprs = np.round(tprs * p) / p
    if n is not None:
        fprs = np.round(fprs * n) / n
        
    return fprs, tprs


def intersection(fpr0, tpr0, fpr1, tpr1, s0, s1):
    fpr = (fpr0 * s0 - fpr1 * s1 - tpr0 + tpr1)/(s0 - s1)
    tpr = (fpr - fpr0) * s0  + tpr0
    return fpr, tpr


def sample_triangle(fpr0, tpr0, fpr1, tpr1, fpr2, tpr2, random_state):
    r1 = np.sqrt(random_state.random_sample())
    r2 = np.sqrt(random_state.random_sample())

    fpr_new = (1 - r1) * fpr0 + r1 * (1 - r2**2) * fpr1 + r2**2 * r1 *  fpr2
    tpr_new = (1 - r1) * tpr0 + r1 * (1 - r2**2) * tpr1 + r2**2 * r1 *  tpr2

    return fpr_new, tpr_new


def generate_1_roc_curve(
    tpr: float,
    fpr: float,
    n_nodes: int,
    p: int = None,
    n: int = None,
    random_middle = False,
    random_state = None
):
    if not isinstance(random_state, np.random.RandomState):
        random_state = np.random.RandomState(random_state)

    tprs = np.full(n_nodes, fill_value=-1, dtype=float)
    fprs = np.full(n_nodes, fill_value=-1, dtype=float)
    slope_left = np.full(n_nodes, fill_value=-1, dtype=float)
    slope_right = np.full(n_nodes, fill_value=-1, dtype=float)

    tprs[0] = 0.0
    tprs[-1] = 1.0

    fprs[0] = 0.0
    fprs[-1] = 1.0

    if random_middle:
        middle_index = random_state.choice(list(range(1, n_nodes-1)))
    else:
        middle_index = int(n_nodes/2)

    tprs[middle_index] = tpr
    fprs[middle_index] = fpr

    slope_right[0] = (tprs[middle_index] - tprs[0]) / (fprs[middle_index] - fprs[0])
    slope_left[-1] = (tprs[-1] - tprs[middle_index]) / (fprs[-1] - fprs[middle_index])
    slope_right[middle_index] = slope_left[-1]
    slope_left[middle_index] = slope_right[0]
    slope_right[-1] = 0.0
    slope_left[0] = 1000

    first_indices = np.array([0, middle_index])
    last_indices = np.array([middle_index, n_nodes-1])
    queue = np.vstack((first_indices, last_indices)).T

    idx = 0
    while queue.shape[0] > 0:
        sample = np.random.choice(len(queue))
        sample_mask = np.arange(len(queue)) == sample

        segment = queue[sample]
        first_idx, last_idx = segment

        if last_idx - first_idx <= 1:
            queue = queue[~sample_mask]
            continue

        fpr_left = fprs[first_idx]
        fpr_right = fprs[last_idx]
        tpr_left = tprs[first_idx]
        tpr_right = tprs[last_idx]
        s_left = slope_left[first_idx]
        s_right = slope_right[last_idx]

        fpr1, tpr1 = intersection(fpr_left, tpr_left, fpr_right, tpr_right, s_left, s_right)

        index_new = random_state.choice(list(range(first_idx+1, last_idx)))

        fpr2, tpr2 = sample_triangle(fpr_left, tpr_left, fpr_right, tpr_right, fpr1, tpr1, random_state)

        fprs[index_new] = fpr2
        tprs[index_new] = tpr2

        slope_left[index_new] = (tprs[index_new] - tprs[first_idx]) / (fprs[index_new] - fprs[first_idx])
        slope_right[first_idx] = slope_left[index_new]
        slope_right[index_new] = (tprs[last_idx] - tprs[index_new]) / (fprs[last_idx] - fprs[index_new])
        slope_left[last_idx] = slope_right[index_new]

        lhs_item = np.hstack([first_idx, index_new])
        rhs_item = np.hstack([index_new, last_idx])

        queue = np.vstack([lhs_item, rhs_item, queue[~sample_mask]])

        idx += 1
    
    if p is not None:
        tprs = np.round(tprs * p) / p
    if n is not None:
        fprs = np.round(fprs * n) / n
        
    return fprs, tprs


def generate_roc_curve_slope(
        tpr: float,
        fpr: float,
        n_nodes: int,
        n_samples: int = 1,
        p: int = None,
        n: int = None,
        random_middle = False,
        random_state = None
):
    
    if not isinstance(random_state, np.random.RandomState):
        random_state = np.random.RandomState(random_state)

    tprs = np.full((n_samples, n_nodes), fill_value=-1, dtype=float)
    fprs = np.full((n_samples, n_nodes), fill_value=-1, dtype=float)
    slope_left = np.full((n_samples, n_nodes), fill_value=-1, dtype=float)
    slope_right = np.full((n_samples, n_nodes), fill_value=-1, dtype=float)

    tprs[:, 0] = 0.0
    tprs[:, -1] = 1.0

    fprs[:, 0] = 0.0
    fprs[:, -1] = 1.0

    if random_middle:
        middle_indices = random_state.choice(list(range(1, n_nodes-1)), n_samples)
    else:
        middle_indices = np.repeat(int(n_nodes/2), n_samples)

    tprs[np.arange(0, n_samples), middle_indices] = tpr
    fprs[np.arange(0, n_samples), middle_indices] = fpr

    slope_right[np.arange(n_samples), 0] = (tprs[np.arange(n_samples), middle_indices] - tprs[np.arange(n_samples), 0]) / (fprs[np.arange(n_samples), middle_indices] - fprs[np.arange(n_samples), 0])
    slope_left[np.arange(n_samples), -1] = (tprs[np.arange(n_samples), -1] - tprs[np.arange(n_samples), middle_indices]) / (fprs[np.arange(n_samples), -1] - fprs[np.arange(n_samples), middle_indices])
    slope_right[np.arange(n_samples), middle_indices] = slope_left[np.arange(n_samples), -1]
    slope_left[np.arange(n_samples), middle_indices] = slope_right[np.arange(n_samples), 0]
    slope_right[np.arange(n_samples), -1] = 0.0
    slope_left[np.arange(n_samples), 0] = np.inf

    first_indices = np.repeat(0, n_samples)
    last_indices = middle_indices
    queue = np.vstack((first_indices, last_indices)).T

    first_indices = middle_indices.copy()
    last_indices = np.repeat(n_nodes - 1, n_samples)
    queue_all = np.vstack([queue, np.vstack((first_indices, last_indices)).T])

    sample_indices_all = np.hstack([np.arange(0, n_samples), np.arange(0, n_samples)])

    idx = 0
    while queue_all.shape[0] > 0:
        sample = np.random.choice(len(queue_all))
        sample_mask = np.arange(len(queue_all)) == sample

        queue = queue_all[sample_mask]
        sample_indices = sample_indices_all[sample_mask]

        first_indices = queue[:, 0]
        last_indices = queue[:, 1]

        fpr_left = fprs[sample_indices, first_indices]
        fpr_right = fprs[sample_indices, last_indices]



        fpr_ranges = fpr_right - fpr_left
        uniforms = random_state.random_sample(fpr_ranges.shape[0])
        fpr_steps = (uniforms * fpr_ranges)
        fpr_new = fpr_left + fpr_steps

        index_ranges = last_indices - first_indices - 1
        uniforms = random_state.random_sample(index_ranges.shape[0])
        index_steps = np.floor(index_ranges * uniforms).astype(int) + 1
        index_new = first_indices + index_steps

        fprs[sample_indices, index_new] = fpr_new
        tpr_right = tprs[sample_indices, last_indices]
        tpr_left = tprs[sample_indices, first_indices]

        #print(last_indices)
        #print(slope_left[sample_indices, last_indices])
        #print(slope_right[sample_indices, first_indices])

        tpr_lower = tpr_left + (fpr_new - fpr_left) * slope_right[sample_indices, first_indices]
        tpr_upper = np.minimum((tpr_right + (fpr_new - fpr_right) * slope_right[sample_indices, last_indices]), 
                               (tpr_left + (fpr_new - fpr_left) * slope_left[sample_indices, first_indices]))

        #print(np.sum(tpr_lower > tpr_upper))

        failed_mask = tpr_lower > tpr_upper

        tpr_ranges = tpr_upper - tpr_lower
        #print(tpr_lower, tpr_upper)

        if np.sum(failed_mask) > 0:
            continue

        
        uniforms = random_state.random_sample(tpr_ranges.shape[0])
        tpr_steps = (uniforms * tpr_ranges)
        
        tpr_new = tpr_lower + tpr_steps

        tprs[sample_indices, index_new] = tpr_new

        slope_left[sample_indices, index_new] = (tprs[sample_indices, index_new] - tprs[sample_indices, first_indices]) / (fprs[sample_indices, index_new] - fprs[sample_indices, first_indices])
        slope_right[sample_indices, first_indices] = slope_left[sample_indices, index_new]
        slope_right[sample_indices, index_new] = (tprs[sample_indices, last_indices] - tprs[sample_indices, index_new]) / (fprs[sample_indices, last_indices] - fprs[sample_indices, index_new])
        slope_left[sample_indices, last_indices] = slope_right[sample_indices, index_new]

        lhs_queue = np.vstack([first_indices, first_indices + index_steps]).T[~failed_mask]
        rhs_queue = np.vstack([first_indices + index_steps, last_indices]).T[~failed_mask]

        queue_all = np.vstack([queue[failed_mask], lhs_queue, rhs_queue, queue_all[~sample_mask]])
        sample_indices_all = np.hstack([sample_indices[failed_mask], sample_indices[~failed_mask], sample_indices[~failed_mask], sample_indices_all[~sample_mask]])

        mask = queue_all[:, 0] < queue_all[:, 1] - 1
        queue_all = queue_all[mask]
        sample_indices_all = sample_indices_all[mask]

        idx += 1
    
    if p is not None:
        tprs = np.round(tprs * p) / p
    if n is not None:
        fprs = np.round(fprs * n) / n
        
    return fprs, tprs


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


def simplify_roc(fprs, tprs, ths=None):
    """
    Simplifies a ROC curve

    Args:
        fprs (np.array): the array of false positive rates
        tprs (np.array): the array of true positive rates
        ths (np.array/None): the array of thresholds
    
    Returns:
        np.array, np.array, np.array: the arrays of simplified false positive
                                        rates, true positive rates and 
                                        thresholds
    """
    fprs_simp = [fprs[0]]
    tprs_simp = [tprs[0]]

    if ths is not None:
        ths_simp = [ths[0]]

    for idx in range(1, len(fprs)-1):
        if fprs[idx] == fprs[idx+1] and fprs[idx-1] == fprs[idx]:
            continue
        if tprs[idx] == tprs[idx+1] and tprs[idx-1] == tprs[idx]:
            continue

        if tprs[idx] != tprs[idx+1] and tprs[idx] != tprs[idx-1]:
            if np.abs((fprs[idx-1] - fprs[idx])/(tprs[idx-1] - tprs[idx]) - (fprs[idx] - fprs[idx+1])/(tprs[idx] - tprs[idx+1])) < 1e-6:
                continue
        
        if fprs[idx] != fprs[idx+1] and fprs[idx] != fprs[idx-1]:
            if np.abs((tprs[idx-1] - tprs[idx])/(fprs[idx-1] - fprs[idx]) - (tprs[idx] - tprs[idx+1])/(fprs[idx] - fprs[idx+1])) < 1e-6:
                continue

        fprs_simp.append(fprs[idx])
        tprs_simp.append(tprs[idx])

        if ths is not None:
            ths_simp.append(ths[idx])
    
    fprs_simp.append(fprs[-1])
    tprs_simp.append(tprs[-1])

    if ths is not None:
        ths_simp.append(ths[-1])

    if ths is not None:
        return np.array(fprs_simp), np.array(tprs_simp), np.array(ths_simp)
    
    return np.array(fprs_simp), np.array(tprs_simp)


def roc_value_at(fpr, fpr_curve, tpr_curve):
    """
    Evaluates a ROC curve at a parcitular fpr value

    Args:
        fpr (float): the point to evaluate at
        fpr_curve (np.array): the sequence of fpr values
        tpr_curve (np.array): the sequence of tpr values
    
    Returns:
        float: the tpr value of the curve at fpr
    """
    values = []
    for idx in range(len(fpr_curve)):
        if fpr_curve[idx] == fpr:
            values.append(tpr_curve[idx])
        elif idx < len(fpr_curve) and fpr > fpr_curve[idx] and fpr < fpr_curve[idx+1]:
            diff_left = fpr - fpr_curve[idx]
            diff_right = fpr_curve[idx+1] - fpr

            diff_sum = diff_left + diff_right

            diff_left, diff_right = diff_left/diff_sum, diff_right/diff_sum

            values.append(tpr_curve[idx] * diff_right + tpr_curve[idx+1]*diff_left)
            break

    return np.array(values)


def average_roc_curves_to_1(curve0, curves):
    """
    Averages ROC curves to one curve

    Args:
        curve0 (tuple(np.array, np.array)): the curve to average to
        curves (list(tuple(np.array, np.array))): the curves to average to curve0
    
    Returns:
        np.array, np.array: the average curve
    """
    fprs0, tprs0 = curve0

    fprs_new = []
    tprs_new = []
    for fpr, tpr in zip(fprs0, tprs0):
        tpr_x = np.array([tpr, tpr])
        for fprs, tprs in curves:
            tpr_js = roc_value_at(fpr, fprs, tprs)
            tpr_x = tpr_x + tpr_js
        
        fprs_new.append(fpr)
        tprs_new.append(tpr_x[0]/(1 + len(curves)))

        if tpr_x[0] != tpr_x[1]:
            fprs_new.append(fpr)
            tprs_new.append(tpr_x[1]/(1 + len(curves)))
    
    return np.array(fprs_new), np.array(tprs_new)


def average_n_roc_curves_(curves):
    """
    Determines the average of n ROC curves

    Args:
        curves (list(tuple(np.array, np.array))): the curves to average
    
    Returns:
        np.array, np.array: the average curve
    """
    fprs_new = []
    tprs_new = []

    for idx in range(len(curves)):
        curves_b = curves[0:idx] + curves[(idx+1):]
        fprs_a, tprs_a = average_roc_curves_to_1(curves[idx], curves_b)
        fprs_new.append(fprs_a)
        tprs_new.append(tprs_a)
    
    fprs5 = np.hstack(fprs_new)
    tprs5 = np.hstack(tprs_new)

    rates = np.vstack([fprs5, tprs5]).T.copy()

    sorting = np.argsort(rates[:, 0])
    rates = rates[sorting]
    sorting = np.argsort(rates[:, 1])
    rates = rates[sorting]
    rates = np.unique(rates, axis=0)

    fprs5 = rates[:, 0]
    tprs5 = rates[:, 1]

    assert np.all(np.diff(fprs5) >= -1e-10), np.min(np.diff(fprs5))
    assert np.all(np.diff(tprs5) >= -1e-10), np.min(np.diff(fprs5))

    fprs5, tprs5 = simplify_roc(fprs5, tprs5)

    return fprs5, tprs5


def average_n_roc_curves(curves, random_state=None):
    """
    Determines the average of n ROC curves

    Args:
        curves (list(tuple(np.array, np.array))): the curves to average
        random_state (None/int/np.random.RandomState): the random state
            governing the selection of the average curve when the averaging
            horizontally and vertically leads to curves with the same
            number of nodes
    
    Returns:
        np.array, np.array: the average curve
    """
    if not isinstance(random_state, np.random.RandomState):
        random_state = np.random.RandomState(random_state)

    fprs0, tprs0 = average_n_roc_curves_(curves)

    flipped_curves = []
    for fprs, tprs in curves:
        flipped_curves.append(((1 - tprs)[::-1], (1 - fprs)[::-1]))

    fprs1, tprs1 = average_n_roc_curves_(flipped_curves)

    fprs1, tprs1 = (1 - tprs1)[::-1], (1 - fprs1)[::-1]

    if len(fprs0) == len(fprs1):
        if random_state.randint(2) == 0:
            return fprs0, tprs0
        return fprs1, tprs1

    if len(fprs0) < len(fprs1):
        return fprs0, tprs0

    return fprs1, tprs1


def p_norm_fit(x, y, bracket=(-5, 3), mode='implicit'):
    p = np.logspace(bracket[0], bracket[1], 2000)
    if mode == 'implicit':
        return p[np.argmin(np.mean(np.abs(1 - x**p[:, None] - y**p[:, None])**1, axis=1))]
    elif mode == 'explicit':
        return p[np.argmin(np.mean(np.abs(y - (1 - x**p[:, None])**(1/p[:, None]))**1, axis=1))]


def p_norm_fit_joint(x, y0, y1, bracket=(-5, 0, 3), p=None, n=None, max_acc=None):
    p0 = np.logspace(bracket[0], bracket[1], 500)
    p1 = np.logspace(bracket[1], bracket[2], 500)
    #p0 = np.logspace(-1, bracket[1], 4)
    #p1 = np.logspace(1, bracket[2], 5)

    err0 = np.mean(np.abs(1 - x**p0[:, None] - y0**p0[:, None])**1, axis=1)
    err1 = np.mean(np.abs(1 - x**(p1[:, None]) - y1**(p1[:, None]))**1, axis=1)

    err = err0[:, None] + err1

    if max_acc is not None:
        #z = np.linspace(0, 1, 6)
        z = np.linspace(0, 1, 100)
        tmp = (((1 - (1 - z**p0[:, None])**(1/p0[:, None])) * n)[:, None] + ((1 - z**p1[:, None])**(1/p1[:, None]) * p)) / (p + n)
        #print(tmp)
        max_accs = np.max(tmp, axis=2)
        mask = max_accs > max_acc
        #print(max_accs)
        #print(np.sum(mask), np.prod(mask.shape))
        err[mask] = np.inf

    min0, min1 = np.unravel_index(np.argmin(err), err.shape)

    return p0[min0], p1[min1]


def p_norm_fit_best(x, y, bracket=(-5, 3), mode='implicit', p=None, n=None, max_acc=None):
    exp = np.logspace(bracket[0], bracket[1], 2000)
    err = np.mean(np.abs(1 - x**exp[:, None] - y**exp[:, None])**1, axis=1)
    if max_acc is not None:
        z = np.linspace(0, 1, min(100, n if n is not None else 100))
        fprs = (z)[:, None]
        tprs = ((1 - z[:, None]**exp)**(1/exp))
        tmp = (fprs * n + tprs * p) / (p + n)
        max_accs = np.max(tmp, axis=0)
        mask = max_accs > max_acc
        err[mask] = np.inf
    
    return exp[np.argmin(err)]


def auc_estimator(fpr, tpr, p, n, mode='separate', return_details=False, integral=200, best=False, rasterize=False):
    if fpr < 1e-6 and tpr > 1 - 1e-6 and not best:
        return 1.0, -1, -1
    if fpr < 1e-6 and tpr < 1e-6 and not best:
        return 0.5, -1, -1
    if fpr > 1 - 1e-6 and tpr > 1 - 1e-6 and not best:
        return 0.5, -1, -1
    
    fpr = min(max(fpr, 1/n), 1 - 1/n)
    tpr = min(max(tpr, 1/p), 1 - 1/p)

    fprs = np.array([0.0, fpr, 1.0])
    tprs = np.array([0.0, tpr, 1.0])
    fracs = 1.0 - (p*tprs + n*fprs)/(p + n)

    if mode == 'separate':
        p_fpr = p_norm_fit(fracs, fprs, bracket=(-5, 0))
        p_tpr = p_norm_fit(fracs, tprs, bracket=(0, 2))

        #print(p_fpr, p_tpr)

        fracs = 1.0 - np.linspace(0, 1, integral)
        x = (1.0 - fracs**p_fpr)**(1/p_fpr)
        y = (1.0 - fracs**p_tpr)**(1/p_tpr)
    elif mode == 'joint':
        if best:
            max_acc = (n*(1 - fpr) + p*tpr) / (p + n)
        else:
            max_acc = None
        p_fpr, p_tpr = p_norm_fit_joint(fracs, 
                                        fprs, 
                                        tprs, 
                                        bracket=(-5, 0, 3),
                                        p=p,
                                        n=n,
                                        max_acc=max_acc)

        if not rasterize:
            fracs = 1.0 - np.linspace(0, 1, integral)
            x = (1.0 - fracs**p_fpr)**(1/p_fpr)
            y = (1.0 - fracs**p_tpr)**(1/p_tpr)
        else:
            fracs = 1.0 - np.linspace(0, 1, p + n)
            x = np.round((1.0 - fracs**p_fpr)**(1/p_fpr)*n)/n
            y = np.round((1.0 - fracs**p_tpr)**(1/p_tpr)*p)/p
    elif mode == 'roc':
        p_both = p_norm_fit(1 - fprs, tprs, bracket=(0, 2))
        p_fpr, p_tpr = p_both, p_both
        #print(p_both)
        if not rasterize:
            x = np.linspace(0, 1, integral)
            y = (1.0 - (1 - x)**p_both)**(1/p_both)
        else:
            x = np.linspace(0, 1, n)
            y = np.round((1.0 - x**p_both)**(1/p_both)*p)/p
    
    elif mode == 'roc2':
        if best:
            max_acc = (n*(1 - fpr) + p*tpr) / (p + n)
        else:
            max_acc = None

        p_both = p_norm_fit_best(1.0 - fprs, 
                                    tprs, 
                                    bracket=(0, 2),
                                    p=p,
                                    n=n,
                                    max_acc=max_acc)

        p_fpr, p_tpr = p_both, p_both

        if not rasterize:
            x = np.linspace(0, 1, integral)
            y = (1.0 - (1 - x)**p_both)**(1/p_both)
        else:
            x = np.linspace(0, 1, n)
            y = np.round((1.0 - x**p_both)**(1/p_both)*p)/p

    if best and mode != 'joint' and mode != 'roc2':
        best_acc = ((1 - fpr)*n + tpr*p) / (p + n)
        accs = ((1 - x)*n + y * p) / (p + n)
        mask = accs > best_acc
        x_change = x[mask]
        y_change = y[mask]

        y_change = (best_acc * (p + n) - (1 - x_change)*n)/p
        y[mask] = y_change

    if not return_details:
        return integrate_roc_curve(x, y), float(p_fpr), float(p_tpr)
    else:
        return (integrate_roc_curve(x, y), x, y)

def auc_error(auc, p):
    x = np.linspace(0, 1, 4000)
    tprs = (1 - (1 - x)**(1/p))**p
    auc0 = integrate_roc_curve(x, tprs[::-1])
    return (auc0 - auc)

def p_norm_fit_auc(auc, bracket=(1e-20, 1)):

    if auc_error(auc, bracket[0]) < 0 and auc_error(auc, bracket[1]) < 0:
        if auc_error(auc, bracket[0]) > auc_error(auc, bracket[1]):
            return bracket[0]
        else:
            return bracket[1]
    
    if auc_error(auc, bracket[0]) > 0 and auc_error(auc, bracket[1]) > 0:
        if auc_error(auc, bracket[0]) > auc_error(auc, bracket[1]):
            return bracket[1]
        else:
            return bracket[0]

    res = root_scalar(
        lambda p: auc_error(auc, p), 
        bracket=bracket
    )
    
    return float(res['root'])

def max_acc_estimator(auc, p, n):
    if auc >= 1 - 1e-4:
        return 1.0
    #print(auc)
    exp = p_norm_fit_auc(auc)
    x = np.linspace(0, 1, 100)
    tprs = (1 - (1 - x)**(1/exp))**exp
    return np.max(((1 - x)*n + tprs*p)/(p + n))

def sample0_min_max(fpr1, tpr1, fpr2, tpr2):
    active = np.repeat(True, len(fpr1))
    fpr_result = np.repeat(-1.0, len(fpr1))
    tpr_result = np.repeat(-1.0, len(fpr1))
    n_active = len(fpr1)

    fpr_result[active] = (fpr2[active] - fpr1[active]) * np.random.random_sample(n_active) + fpr1[active]
    tpr_result[active] = (tpr2[active] - tpr1[active]) * np.random.random_sample(n_active) + tpr1[active]
    #tpr_result[active] = (tpr2[active] - tpr1[active]) * 0.9 + tpr1[active]

    return fpr_result, tpr_result

def sample0_rmin_max(fpr1, tpr1, fpr2, tpr2):
    active = np.repeat(True, len(fpr1))
    fpr_result = np.repeat(-1.0, len(fpr1))
    tpr_result = np.repeat(-1.0, len(fpr1))
    n_active = len(fpr1)

    while n_active > 0:
        
        fpr_result[active] = (fpr2[active] - fpr1[active]) * np.random.random_sample(n_active) + fpr1[active]
        tpr_result[active] = (tpr2[active] - tpr1[active]) * np.random.random_sample(n_active) + tpr1[active]

        lower_bounds = np.max(np.vstack([tpr1, fpr_result]).T, axis=1)

        active = active & (tpr_result < lower_bounds)

        n_active = np.sum(active)

    return fpr_result, tpr_result

def sample0_rmin_maxa(fpr1, tpr1, fpr2, tpr2, max_acc, p, n):
    active = np.repeat(True, len(fpr1))
    fpr_result = np.repeat(-1.0, len(fpr1))
    tpr_result = np.repeat(-1.0, len(fpr1))
    n_active = len(fpr1)

    while n_active > 0:
        
        fpr_result[active] = (fpr2[active] - fpr1[active]) * np.random.random_sample(n_active) + fpr1[active]
        tpr_result[active] = (tpr2[active] - tpr1[active]) * np.random.random_sample(n_active) + tpr1[active]
        #tpr_result[active] = (tpr2[active] - tpr1[active]) * 0.5 + tpr1[active]

        maxa_bounds = (max_acc * (p + n) - (1 - fpr_result) * n) / p

        upper_bounds = np.min(np.vstack([tpr2, maxa_bounds]).T, axis=1)
        lower_bounds = np.max(np.vstack([tpr1, fpr_result]).T, axis=1)

        active = active & ((tpr_result < lower_bounds) | (tpr_result > upper_bounds))

        n_active = np.sum(active)

    return fpr_result, tpr_result

def sample1(fpr0, tpr0, n_samples, n_nodes, p=None, n=None, max_acc=None, mode='min-max'):
    fpr0s = np.repeat(fpr0, n_samples)
    tpr0s = np.repeat(tpr0, n_samples)
    zeros = np.repeat(0.0, n_samples)
    ones = np.repeat(1.0, n_samples)

    curves_fpr = np.zeros((n_samples, n_nodes))
    curves_tpr = np.zeros((n_samples, n_nodes))

    curves_fpr[:, 0] = zeros
    curves_tpr[:, 0] = zeros
    curves_fpr[:, 1] = ones
    curves_tpr[:, 1] = ones

    curves_fpr[:, 2] = fpr0s
    curves_tpr[:, 2] = tpr0s

    pool = [(0, 2), (2, 1)]

    for idx in range(n_nodes - 3):
        left, right = pool[0]
        pool = pool[1:]
        if mode == 'min-max':
            fprs_new, tprs_new = sample0_min_max(curves_fpr[:, left], curves_tpr[:, left], curves_fpr[:, right], curves_tpr[:, right])
        elif mode == 'rmin-max':
            fprs_new, tprs_new = sample0_rmin_max(curves_fpr[:, left], curves_tpr[:, left], curves_fpr[:, right], curves_tpr[:, right])
        elif mode == 'rmin-maxa':
            fprs_new, tprs_new = sample0_rmin_maxa(curves_fpr[:, left], curves_tpr[:, left], curves_fpr[:, right], curves_tpr[:, right], max_acc, p, n)
        curves_fpr[:, idx+3] = fprs_new
        curves_tpr[:, idx+3] = tprs_new
        pool = pool + [(left, idx+3), (idx+3, right)]
    
    sorting = np.argsort(curves_fpr, axis=1)
    curves_fpr = curves_fpr[np.arange(n_samples)[:, None], sorting]
    curves_tpr = curves_tpr[np.arange(n_samples)[:, None], sorting]

    if n is not None:
        curves_fpr = np.round(curves_fpr * n) / n

    if p is not None:
        curves_tpr = np.round(curves_tpr * p) / p
    
    return curves_fpr, curves_tpr

def sample2(fpr0, tpr0, n_samples, n_nodes, p=None, n=None, max_acc=None, mode='min-max', raw=False):
    fprs, tprs = sample1(fpr0, tpr0, n_samples, n_nodes, p, n, max_acc, mode)
    aucs = integrate_roc_curves(fprs, tprs)
    n_nodes = n_nodes - np.sum((fprs[:, :-1] == fprs[:, 1:]) & (tprs[:, :-1] == tprs[:, 1:]), axis=1)
    if not raw:
        return np.mean(aucs)
    else:
        return aucs, n_nodes
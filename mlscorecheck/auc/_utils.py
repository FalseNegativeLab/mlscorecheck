"""
This module implements some utilities for the AUC related calculations
"""

import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

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
    "exponential_fitting",
    "exponential_fitting2"
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


def exponential_fitting(row, label, frac_label):

    values = row[label].copy()
    counts = row[frac_label].copy()

    mask = values > 1e-6
    values_nz = values[mask]
    counts_nz = counts[mask]

    ln_values = np.log(values_nz)
    ln_counts = np.log(counts_nz).reshape(-1, 1)

    linreg_a = LinearRegression(fit_intercept=False, positive=True)
    pred_values = linreg_a\
        .fit(ln_counts, ln_values)\
        .predict(ln_counts)

    if len(values) > 3:
        r2_a = r2_score(ln_values, pred_values)
    else:
        r2_a = 1.0

    values = (1 - values)
    counts = (1 - counts)

    mask = values > 1e-6
    values_nz = values[mask]
    counts_nz = counts[mask]

    ln_values = np.log(values_nz)
    ln_counts = np.log(counts_nz).reshape(-1, 1)

    linreg_b = LinearRegression(fit_intercept=False, positive=True)
    pred_values = linreg_b\
        .fit(ln_counts, ln_values)\
        .predict(ln_counts)

    if len(values) > 3:
        r2_b = r2_score(ln_values, pred_values)
    else:
        r2_b = 1.0

    #print(r2_a, linreg_a.coef_[0], 0)
    #print(r2_b, linreg_b.coef_[0], 1)

    if r2_a > r2_b:
        return (r2_a, linreg_a.coef_[0], 0)
    return (r2_b, linreg_b.coef_[0], 1)

    if label == 'fprs':
        return (r2_b, linreg_b.coef_[0], 1)
    return (r2_a, linreg_a.coef_[0], 0)

def exponential_fitting2(row, label, frac_label):
    values = row[label].copy()
    counts = row[frac_label].copy()

    if label == 'tprs':
        r2, coef, _ = exponential_fitting2_(values, counts)
        return r2, 1/coef, -3
    else:
        r2, coef, _ = exponential_fitting2_(values, 1 - counts)
        return r2, coef, -3

    r2a, coefa, _ = exponential_fitting2_(values, counts)
    r2b, coefb, _ = exponential_fitting2_(1 - values, 1 - counts)

    print(r2a, coefa, r2b, coefb)

    if r2a > r2b:
        return r2a, coefa, -2
    else:
        return r2b, 1.0/coefb, -3

def exponential_fitting2_(values, counts):

    """if len(values) <= 3:
        return (1.0, 1.0, -1)"""


    mask = (values > 1e-6) & (counts > 1e-6)
    values_nz = values[mask]
    counts_nz = counts[mask]

    ln_values = np.log(values_nz)
    ln_counts = np.log(counts_nz)

    """values2 = (1 - values)
    counts2 = (1 - counts)

    mask2 = (values2 > 1e-6) & (values2 < 1)
    values2_nz = values2[mask2]
    counts2_nz = counts2[mask2]"""

    """ln_values2 = 1/np.log(values2_nz)
    ln_counts2 = 1/np.log(counts2_nz)"""

    ln_x = np.hstack([ln_counts]).reshape(-1, 1)
    ln_y = np.hstack([ln_values])
    
    if len(ln_x) <= 1:
        return (1.0, 1.0, -1)
    
    linreg_a = LinearRegression(fit_intercept=False, positive=True)
    pred_values = linreg_a\
        .fit(ln_x, ln_y)\
        .predict(ln_x)

    if len(values) >= 3:
        r2_a = r2_score(ln_y, pred_values)
    else:
        r2_a = 1.0
    
    return (r2_a, linreg_a.coef_[0], -1)

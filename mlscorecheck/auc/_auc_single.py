"""
This module implements all AUC related functionalities
"""

import numpy as np

from ._utils import translate_scores, prepare_intervals

__all__ = [
    "augment_intervals",
    "auc_from",
    "auc_lower_from",
    "auc_upper_from",
    "integrate_roc_curve",
    "roc_min",
    "roc_max",
    "roc_rmin",
    "roc_rmin_grid",
    "roc_rmin_grid_correction",
    "roc_maxa",
    "roc_onmin",
    "auc_min",
    "auc_max",
    "auc_rmin",
    "auc_rmin_grid",
    "auc_maxa",
    "auc_amin",
    "auc_armin",
    "auc_amax",
    "auc_onmin",
    "auc_onmin_grad",
    "auc_maxa_grad",
    "auc_min_grad",
    "auc_max_grad",
    "auc_rmin_grad",
    "check_lower_applicability",
    "check_upper_applicability",
]


def augment_intervals(intervals: dict, p: int, n: int):
    """
    Augment the intervals based on the relationship between tpr, fpr and acc

    Args:
        intervals (dict): the intervals of scores
        p (int): the number of positive samples
        n (int): the number of negative samples

    Returns:
        dict: the intervals augmented
    """
    intervals = {**intervals}

    if "tpr" not in intervals and ("acc" in intervals and "fpr" in intervals):
        lower = max(
            ((intervals["acc"][0]) * (p + n) - ((1 - intervals["fpr"][0]) * n)) / p, 0
        )
        upper = min(
            ((intervals["acc"][1]) * (p + n) - ((1 - intervals["fpr"][1]) * n)) / p, 1
        )
        intervals["tpr"] = (lower, upper)
    if "fpr" not in intervals and ("acc" in intervals and "tpr" in intervals):
        lower = max(
            ((intervals["acc"][0]) * (p + n) - (intervals["tpr"][1] * p)) / n, 0
        )
        upper = min(
            ((intervals["acc"][1]) * (p + n) - (intervals["tpr"][0] * p)) / n, 1
        )
        intervals["fpr"] = (1 - upper, 1 - lower)
    if "acc" not in intervals and ("fpr" in intervals and "tpr" in intervals):
        lower = max(
            (intervals["tpr"][0] * p + (1 - intervals["fpr"][1]) * n) / (p + n), 0
        )
        upper = min(
            (intervals["tpr"][1] * p + (1 - intervals["fpr"][0]) * n) / (p + n), 1
        )
        intervals["acc"] = (lower, upper)

    return intervals


def integrate_roc_curve(fprs, tprs):
    """
    Integrates ROC curves

    Args:
        fprs (np.array): the fpr values
        tprs (np.array): the tpr values

    Returns:
        float: the integral
    """
    diffs = np.diff(fprs)
    avgs = (tprs[:-1] + tprs[1:]) / 2
    return float(np.sum(diffs * avgs))


def roc_min(fpr, tpr):
    """
    The minimum ROC curve at fpr, tpr

    Args:
        fpr (float): the false positive rate
        tpr (float): the true positive rate

    Returns:
        np.array, np.array: the fpr and tpr values
    """
    return (np.array([0, fpr, fpr, 1, 1]), np.array([0, 0, tpr, tpr, 1]))


def roc_max(fpr, tpr):
    """
    The maximum ROC curve at fpr, tpr

    Args:
        fpr (float): the false positive rate
        tpr (float): the true positive rate

    Returns:
        np.array, np.array: the fpr and tpr values
    """

    return (np.array([0, 0, fpr, fpr, 1]), np.array([0, tpr, tpr, 1, 1]))


def roc_rmin(fpr, tpr):
    """
    The regulated minimum ROC curve at fpr, tpr

    Args:
        fpr (float): the false positive rate
        tpr (float): the true positive rate

    Returns:
        np.array, np.array: the fpr and tpr values

    Raises:
        ValueError: when tpr < fpr
    """

    if tpr < fpr:
        raise ValueError("the regulated minimum curve does not exist when tpr < fpr")

    return (np.array([0, fpr, fpr, tpr, 1]), np.array([0, fpr, tpr, tpr, 1]))


def roc_rmin_grid(fpr, tpr, p, n):
    """
    The regulated minimum ROC curve at fpr, tpr on the grid

    Args:
        fpr (float): the false positive rate
        tpr (float): the true positive rate
        p (int): the number of positive samples
        n: (int): the number of negative samples

    Returns:
        np.array, np.array: the fpr and tpr values

    Raises:
        ValueError: when tpr < fpr
    """

    if tpr < fpr:
        raise ValueError("the regulated minimum curve does not exist when tpr < fpr")

    return (
        np.array([0, fpr, fpr, np.floor(tpr * n) / n, 1]),
        np.array([0, np.ceil(fpr * p) / p, tpr, tpr, 1]),
    )


def roc_rmin_grid_correction(fpr, tpr, p, n):
    """
    The grid correction for regulated minimum curves

    Args:
        fpr (float): the false positive rate
        tpr (float): the true positive rate
        p (int): the number of positive samples
        n: (int): the number of negative samples

    Returns:
        float: the fpr and tpr values
    """

    eps = 0

    a = np.ceil(fpr * p - eps) / p - fpr
    b = np.floor(tpr * n + eps) / n - tpr

    return float(fpr * a / 2 + (1 - tpr) / 2 * (-b))


def roc_maxa(acc, p, n):
    """
    The maximuma accuracy ROC curve with acc accuracy

    Args:
        acc (float): the accuracy
        p (int): the number of positive samples
        n: (int): the number of negative samples

    Returns:
        np.array, np.array: the fpr and tpr values

    Raises:
        ValueError: when acc < max(p, n) / (p + n)
    """

    if acc < max(p, n) / (p + n):
        raise ValueError(
            "the maximum accuracy curve does not exist when acc < max(p,n)/(p + n)"
        )

    tpr_a = (acc * (p + n) - n) / p
    fpr_b = 1 - (acc * (p + n) - p) / n

    return (np.array([0, 0, fpr_b, 1]), np.array([0, tpr_a, 1, 1]))


def roc_onmin(fpr, tpr):
    """
    The one node ROC curve

    Args:
        fpr (float): the false positive rate
        tpr (float): the true positive rate

    Returns:
        np.array, np.array: the fpr and tpr values
    """

    return (np.array([0, fpr, 1]), np.array([0, tpr, 1]))


def auc_min(fpr, tpr):
    """
    The area under the minimum curve at fpr, tpr

    Args:
        fpr (float): upper bound on false positive rate
        tpr (float): lower bound on true positive rate

    Returns:
        float: the area
    """

    return float(tpr * (1 - fpr))


def auc_min_grad(fpr, tpr):
    """
    The gradient of the minimum AUC

    Args:
        fpr (float): upper bound on false positive rate
        tpr (float): lower bound on true positive rate

    Returns:
        float: the gradient magnitude
    """

    return np.sqrt((1 - fpr)**2 + (-tpr)**2)


def auc_rmin(fpr, tpr):
    """
    The area under the regulated minimum curve at fpr, tpr

    Args:
        fpr (float): lower bound on false positive rate
        tpr (float): upper bound on true positive rate

    Returns:
        float: the area

    Raises:
        ValueError: when tpr < fpr
    """

    if tpr < fpr:
        raise ValueError(
            'sens >= 1 - spec does not hold for "\
                        "the regulated minimum curve'
        )
    return float(0.5 + (tpr - fpr) ** 2 / 2.0)


def auc_rmin_grad(fpr, tpr):
    """
    The gradient of the minimum AUC

    Args:
        fpr (float): upper bound on false positive rate
        tpr (float): lower bound on true positive rate

    Returns:
        float: the gradient magnitude
    """

    return np.sqrt((tpr-fpr)**2 + (fpr-tpr)**2)


def auc_rmin_grid(fpr, tpr, p, n):
    """
    The area under the regulated minimum curve at fpr, tpr, with grid
    correction

    Args:
        fpr (float): lower bound on false positive rate
        tpr (float): upper bound on true positive rate

    Returns:
        float: the area

    Raises:
        ValueError: when tpr < fpr
    """

    if tpr < fpr:
        raise ValueError(
            'sens >= 1 - spec does not hold for "\
                        "the regulated minimum curve'
        )
    return float(0.5 + (tpr - fpr) ** 2 / 2.0) + roc_rmin_grid_correction(
        fpr, tpr, p, n
    )


def auc_max(fpr, tpr):
    """
    The area under the maximum curve at fpr, tpr

    Args:
        fpr (float): lower bound on false positive rate
        tpr (float): upper bound on true positive rate

    Returns:
        float: the area
    """

    return float(1 - (1 - tpr) * fpr)


def auc_max_grad(fpr, tpr):
    """
    The gradient of the maximum AUC

    Args:
        fpr (float): upper bound on false positive rate
        tpr (float): lower bound on true positive rate

    Returns:
        float: the gradient magnitude
    """

    return np.sqrt(fpr**2 + (tpr - 1)**2)
    #return max(fpr**2, (tpr - 1)**2)


def auc_maxa(acc, p, n):
    """
    The area under the maximum accuracy curve at acc

    Args:
        acc (float): upper bound on accuracy
        p (int): the number of positive test samples
        n (int): the number of negative test samples

    Returns:
        float: the area

    Raises:
        ValueError: when acc < max(p, n) / (p + n)
    """

    if acc < max(p, n) / (p + n):
        raise ValueError("accuracy too small")

    return float(1 - ((1 - acc) * (p + n)) ** 2 / (2 * n * p))


def auc_maxa_grad(acc, p, n):
    """
    The gradient magnitude of the amax estimation

    Args:
        acc (float): the accuracy
        p (int): the number of positive samples
        n (int): the number of negative samples

    Returns:
        float: the gradient magnitude 
    """

    #d_sens = (1 - acc)*(p + n)/n
    #d_spec = (1 - acc)*(p + n)/p

    #return np.sqrt(d_sens**2 + d_spec**2)
    return - (2*acc - 2)*(n + p)**2/(2*n*p)


def auc_amin(acc, p, n):
    """
    The smallest area under the minimum curve at acc

    Args:
        acc (float): lower bound on accuracy
        p (int): the number of positive test samples
        n (int): the number of negative test samples

    Returns:
        float: the area
    """
    if acc < max(p, n) / (p + n):
        return 0.0

    return float(acc - (1 - acc) * max(p, n) / min(p, n))


def auc_amax(acc, p, n):
    """
    The greatest area under the maximum curve at acc

    Args:
        acc (float): upper bound on accuracy
        p (int): the number of positive test samples
        n (int): the number of negative test samples

    Returns:
        float: the area
    """

    if acc > min(p, n) / (p + n):
        return 1.0

    return float(acc * (1 + max(p, n) / min(p, n)))


def auc_armin(acc, p, n):
    """
    The smallest area under the regulated minimum curve at acc

    Args:
        acc (float): lower bound on accuracy
        p (int): the number of positive test samples
        n (int): the number of negative test samples

    Returns:
        float: the area

    Raises:
        ValueError: when acc < min(p, n) / (p + n)
    """
    if acc < min(p, n) / (p + n):
        raise ValueError("accuracy too small")
    if min(p, n) / (p + n) <= acc <= max(p, n) / (p + n):
        return 0.5

    return float(auc_amin(acc, p, n) ** 2 / 2 + 0.5)


def auc_onmin(fpr, tpr):
    """
    The area under the one-node ROC curve

    Args:
        fpr (float): lower bound on false positive rate
        tpr (float): upper bound on true positive rate

    Returns:
        float: the area
    """

    return (tpr + 1 - fpr) / 2.0


def auc_onmin_grad(fpr, tpr):
    """
    The gradient magnitude of the onmin estimation

    Args:
        acc (float): the accuracy
        p (int): the number of positive samples
        n (int): the number of negative samples

    Returns:
        float: the gradient magnitude 
    """

    return np.sqrt(2*0.5**2)
    #return 0.5


def check_lower_applicability(intervals: dict, lower: str, p: int, n: int):
    """
    Checks the applicability of the methods

    Args:
        intervals (dict): the score intervals
        lower (str): the lower bound method
        p (int): the number of positive samples
        n (int): the number of negative samples

    Raises:
        ValueError: when the methods are not applicable with the
                    specified scores
    """
    if lower in ["min", "rmin", "grmin", "onmin"] and (
        "fpr" not in intervals or "tpr" not in intervals
    ):
        raise ValueError("fpr, tpr or their complements must be specified")
    if lower in ["grmin", "amin", "armin"] and (p is None or n is None):
        raise ValueError("p and n must be specified")
    if lower in ["amin", "armin"] and ("acc" not in intervals):
        raise ValueError("acc must be specified")


def check_upper_applicability(intervals: dict, upper: str, p: int, n: int):
    """
    Checks the applicability of the methods

    Args:
        intervals (dict): the score intervals
        upper (str): the upper bound method
        p (int): the number of positive samples
        n (int): the number of negative samples

    Raises:
        ValueError: when the methods are not applicable with the
                    specified scores
    """
    if upper in ["max"] and ("fpr" not in intervals or "tpr" not in intervals):
        raise ValueError("fpr, tpr or their complements must be specified")
    if upper in ["amax", "maxa"] and (p is None or n is None):
        raise ValueError("p and n must be specified")
    if upper in ["amax", "maxa"] and "acc" not in intervals:
        raise ValueError("acc must be specified")


def auc_lower_from(
    *, scores: dict, eps: float, p: int = None, n: int = None, lower: str = "min"
):
    """
    This function applies the lower bound estimation schemes to estimate
    AUC from scores

    Args:
        scores (dict): the reported scores
        eps (float): the numerical uncertainty
        p (int): the number of positive samples
        n (int): the number of negative samples
        lower (str): ('min'/'rmin'/'grmin'/'amin'/'armin') - the type of
                        estimation for the lower bound

    Returns:
        float: the lower bound for the AUC

    Raises:
        ValueError: when the parameters are not suitable for the estimation methods
        or the scores are inconsistent
    """

    scores = translate_scores(scores)
    intervals = prepare_intervals(scores, eps)

    if p is not None and n is not None:
        intervals = augment_intervals(intervals, p, n)

    check_lower_applicability(intervals, lower, p, n)

    if lower == "min":
        lower0 = auc_min(intervals["fpr"][1], intervals["tpr"][0])
    elif lower == "rmin":
        lower0 = auc_rmin(intervals["fpr"][0], intervals["tpr"][1])
    elif lower == "onmin":
        lower0 = auc_onmin(intervals["fpr"][0], intervals["tpr"][0])
    elif lower == "grmin":
        lower0 = auc_rmin_grid(intervals["fpr"][0], intervals["tpr"][1], p, n)
    elif lower == "amin":
        lower0 = auc_amin(intervals["acc"][0], p, n)
    elif lower == "armin":
        lower0 = auc_armin(intervals["acc"][0], p, n)
    else:
        raise ValueError(f"unsupported lower bound {lower}")

    return lower0


def auc_upper_from(
    *, scores: dict, eps: float, p: int = None, n: int = None, upper: str = "max"
):
    """
    This function applies the lower bound estimation schemes to estimate
    AUC from scores

    Args:
        scores (dict): the reported scores
        eps (float): the numerical uncertainty
        p (int): the number of positive samples
        n (int): the number of negative samples
        upper (str): ('max'/'maxa'/'amax') - the type of estimation for
                        the upper bound

    Returns:
        float: the upper bound for the AUC

    Raises:
        ValueError: when the parameters are not suitable for the estimation methods
        or the scores are inconsistent
    """

    scores = translate_scores(scores)
    intervals = prepare_intervals(scores, eps)

    if p is not None and n is not None:
        intervals = augment_intervals(intervals, p, n)

    check_upper_applicability(intervals, upper, p, n)

    if upper == "max":
        upper0 = auc_max(intervals["fpr"][0], intervals["tpr"][1])
    elif upper == "amax":
        upper0 = auc_amax(intervals["acc"][1], p, n)
    elif upper == "maxa":
        upper0 = auc_maxa(intervals["acc"][1], p, n)
    else:
        raise ValueError(f"unsupported upper bound {upper}")

    return upper0


def auc_from(
    *,
    scores: dict,
    eps: float,
    p: int = None,
    n: int = None,
    lower: str = "min",
    upper: str = "max",
    gradient_correction: bool = False
) -> tuple:
    """
    This function applies the estimation schemes to estimate AUC from scores

    Args:
        scores (dict): the reported scores
        eps (float): the numerical uncertainty
        p (int): the number of positive samples
        n (int): the number of negative samples
        lower (str): ('min'/'rmin'/'grmin'/'amin'/'armin'/'onmin') - the 
                        type of estimation for the lower bound
        upper (str): ('max'/'maxa'/'amax') - the type of estimation for
                        the upper bound
        gradient_correction (bool): whether to use gradient correction

    Returns:
        tuple(float, float): the interval for the AUC

    Raises:
        ValueError: when the parameters are not suitable for the estimation methods
        or the scores are inconsistent
    """

    lower0 = auc_lower_from(scores=scores, eps=eps, p=p, n=n, lower=lower)
    lower_weight = 1.0

    upper0 = auc_upper_from(scores=scores, eps=eps, p=p, n=n, upper=upper)
    upper_weight = 1.0

    return (lower0, upper0)

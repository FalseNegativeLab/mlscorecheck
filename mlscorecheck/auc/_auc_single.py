"""
This module implements all AUC related functionalities
"""

import numpy as np

from ._utils import translate_scores, prepare_intervals

__all__ = [
    "augment_intervals",
    "auc_from",
    "integrate_roc_curve",
    "roc_min",
    "roc_max",
    "roc_rmin",
    "roc_rmin_grid",
    "roc_rmin_grid_correction",
    "roc_maxa",
    "auc_min",
    "auc_max",
    "auc_rmin",
    "auc_rmin_grid",
    "auc_maxa",
    "auc_amin",
    "auc_armin",
    "auc_amax",
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
        fpr (float): the false positive rate
        tpr (float): the true positive rate
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


def check_applicability(intervals: dict, lower: str, upper: str, p: int, n: int):
    """
    Checks the applicability of the methods

    Args:
        intervals (dict): the score intervals
        lower (str): the lower bound method
        upper (str): the upper bound method
        p (int): the number of positive samples
        n (int): the number of negative samples

    Raises:
        ValueError: when the methods are not applicable with the
                    specified scores
    """
    if lower in ["min", "rmin", "grmin"] or upper in ["max"]:
        if "fpr" not in intervals or "tpr" not in intervals:
            raise ValueError("fpr, tpr or their complements must be specified")
    if lower in ["grmin", "amin", "armin"] or upper in ["amax", "maxa"]:
        if p is None or n is None:
            raise ValueError("p and n must be specified")
    if lower in ["amin", "armin"] or upper in ["amax", "maxa"]:
        if "acc" not in intervals:
            raise ValueError("acc must be specified")


def auc_from(
    *,
    scores: dict,
    eps: float,
    p: int = None,
    n: int = None,
    lower: str = "min",
    upper: str = "max",
) -> tuple:
    """
    This function applies the estimation schemes to estimate AUC from scores

    Args:
        scores (dict): the reported scores
        eps (float): the numerical uncertainty
        p (int): the number of positive samples
        n (int): the number of negative samples
        lower (str): ('min'/'rmin'/'grmin'/'amin'/'armin') - the type of
                        estimation for the lower bound
        upper (str): ('max'/'maxa'/'amax') - the type of estimation for
                        the upper bound

    Returns:
        tuple(float, float): the interval for the AUC

    Raises:
        ValueError: when the parameters are not suitable for the estimation methods
        or the scores are inconsistent
    """

    scores = translate_scores(scores)
    intervals = prepare_intervals(scores, eps)

    if p is not None and n is not None:
        intervals = augment_intervals(intervals, p, n)

    check_applicability(intervals, lower, upper, p, n)

    if lower == "min":
        lower0 = auc_min(intervals["fpr"][1], intervals["tpr"][0])
    elif lower == "rmin":
        lower0 = auc_rmin(intervals["fpr"][0], intervals["tpr"][1])
    elif lower == "grmin":
        lower0 = auc_rmin_grid(intervals["fpr"][0], intervals["tpr"][1], p, n)
    elif lower == "amin":
        lower0 = auc_amin(intervals["acc"][0], p, n)
    elif lower == "armin":
        lower0 = auc_armin(intervals["acc"][0], p, n)
    else:
        raise ValueError(f"unsupported lower bound {lower}")

    if upper == "max":
        upper0 = auc_max(intervals["fpr"][0], intervals["tpr"][1])
    elif upper == "amax":
        upper0 = auc_amax(intervals["acc"][1], p, n)
    elif upper == "maxa":
        upper0 = auc_maxa(intervals["acc"][1], p, n)
    else:
        raise ValueError(f"unsupported upper bound {upper}")

    return (lower0, upper0)

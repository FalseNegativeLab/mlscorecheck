"""
This module implements all AUC based accuracy
estimation related functionalities
"""

import numpy as np

from ._utils import prepare_intervals

__all__ = [
    "acc_from",
    "max_acc_from",
    "acc_lower_from",
    "max_acc_lower_from",
    "acc_upper_from",
    "max_acc_upper_from",
    "acc_min",
    "acc_rmin",
    "acc_max",
    "acc_max_grad",
    "acc_rmax",
    "acc_rmax_grad",
    "acc_onmax",
    "acc_onmax_grad",
    "macc_min",
    "macc_min_grad",
]


def acc_min(auc, p, n):
    """
    The minimum accuracy given an AUC

    Args:
        auc (float): lower bound on AUC
        p (int): the number of positive test samples
        n (int): the number of negative test samples

    Returns:
        float: the accuracy
    """
    return auc * min(p, n) / (p + n)


def acc_rmin(auc, p, n):
    """
    The minimum accuracy given an AUC, assuming the curve does not
    go below the random classification line

    Args:
        auc (float): the lower bound on AUC
        p (int): the number of positive test samples
        n (int): the number of negative test samples

    Returns:
        float: the accuracy

    Raises:
        ValueError: when auc < 0.5
    """
    if auc < 0.5:
        raise ValueError("the AUC is too small")

    return min(p, n) / (p + n)


def acc_max(auc, p, n):
    """
    The maximum accuracy given an AUC

    Args:
        auc (float): upper bound on AUC
        p (int): the number of positive test samples
        n (int): the number of negative test samples

    Returns:
        float: the accuracy
    """
    return (auc * min(p, n) + max(p, n)) / (p + n)


def acc_max_grad(auc, p, n):
    """
    The gradient of maximum accuracy given an AUC

    Args:
        auc (float): upper bound on AUC
        p (int): the number of positive test samples
        n (int): the number of negative test samples

    Returns:
        float: the accuracy
    """
    return min(p, n) / (p + n)


def acc_rmax(auc, p, n):
    """
    The maximum accuracy on a regulated minimum curve given an AUC

    Args:
        auc (float): upper bound on AUC
        p (int): the number of positive test samples
        n (int): the number of negative test samples

    Returns:
        float: the accuracy

    Raises:
        ValueError: when auc < 0.5
    """
    if auc < 0.5:
        raise ValueError("auc too small")
    return (max(p, n) + min(p, n) * np.sqrt(2 * (auc - 0.5))) / (p + n)


def acc_rmax_grad(auc, p, n):
    """
    The gradient of regulated maximum accuracy given an AUC

    Args:
        auc (float): upper bound on AUC
        p (int): the number of positive test samples
        n (int): the number of negative test samples

    Returns:
        float: the accuracy
    """
    return np.sqrt(2) * min(p, n) / 2 / (np.sqrt(auc - 0.5) * (p + n))


def acc_onmax(auc, p, n):
    """
    The maximum accuracy on a one node curve given an AUC

    Args:
        auc (float): upper bound on AUC
        p (int): the number of positive test samples
        n (int): the number of negative test samples

    Returns:
        float: the accuracy

    Raises:
        ValueError: when auc < 0.5
    """

    if auc < 0.5:
        raise ValueError("auc too small for acc_onmax")

    return (2 * auc * min(p, n) + max(p, n) - min(p, n)) / (p + n)


def acc_onmax_grad(auc, p, n):
    """
    The gradient of one node maximum accuracy given an AUC

    Args:
        auc (float): upper bound on AUC
        p (int): the number of positive test samples
        n (int): the number of negative test samples

    Returns:
        float: the accuracy
    """
    return 2 * min(p, n) / (p + n)


def macc_min(auc, p, n):
    """
    The minimum of the maximum accuracy

    Args:
        auc (float): lower bound on AUC
        p (int): the number of positive test samples
        n (int): the number of negative test samples

    Returns:
        float: the accuracy
    """
    if auc >= 1 - min(p, n) / (2 * max(p, n)):
        return 1 - (np.sqrt(2 * p * n - 2 * auc * p * n)) / (p + n)

    return max(p, n) / (p + n)


def macc_min_grad(auc, p, n):
    """
    The gradient of the minimum maximum accuracy

    Args:
        fpr (float): upper bound on false positive rate
        tpr (float): lower bound on true positive rate

    Returns:
        float: the gradient magnitude
    """

    if auc >= 1 - min(p, n) / (2 * max(p, n)):
        return n * p / ((n + p) * np.sqrt(-2 * auc * n * p + 2 * n * p))

    return 0.0


def acc_lower_from(*, scores: dict, eps: float, p: int, n: int, lower: str = "min"):
    """
    This function applies the lower bound estimation schemes to estimate
    acc from scores

    Args:
        scores (dict): the reported scores
        eps (float): the numerical uncertainty
        p (int): the number of positive samples
        n (int): the number of negative samples
        lower (str): 'min'/'rmin'

    Returns:
        float: the lower bound for the accuracy

    Raises:
        ValueError: when the parameters are not suitable for the estimation methods
        or the scores are inconsistent
    """

    intervals = prepare_intervals(scores, eps)

    if "auc" not in intervals:
        raise ValueError("auc must be specified")

    if lower == "min":
        lower0 = acc_min(intervals["auc"][0], p, n)
    elif lower == "rmin":
        lower0 = acc_rmin(intervals["auc"][0], p, n)
    else:
        raise ValueError(f"unsupported lower bound {lower}")

    return lower0


def acc_upper_from(*, scores: dict, eps: float, p: int, n: int, upper: str = "max"):
    """
    This function applies the lower bound estimation schemes to estimate
    acc from scores

    Args:
        scores (dict): the reported scores
        eps (float): the numerical uncertainty
        p (int): the number of positive samples
        n (int): the number of negative samples
        upper (str): 'max'/'rmax'/'onmax' - the type of upper bound

    Returns:
        float: the upper bound for the accuracy

    Raises:
        ValueError: when the parameters are not suitable for the estimation methods
        or the scores are inconsistent
    """

    intervals = prepare_intervals(scores, eps)

    if "auc" not in intervals:
        raise ValueError("auc must be specified")

    if upper == "max":
        upper0 = acc_max(intervals["auc"][1], p, n)
    elif upper == "rmax":
        upper0 = acc_rmax(intervals["auc"][1], p, n)
    elif upper == "onmax":
        upper0 = acc_onmax(intervals["auc"][1], p, n)
    else:
        raise ValueError(f"unsupported upper bound {upper}")

    return upper0


def acc_from(
    *, scores: dict, eps: float, p: int, n: int, lower: str = "min", upper: str = "max"
) -> tuple:
    """
    This function applies the estimation schemes to estimate acc from scores

    Args:
        scores (dict): the reported scores
        eps (float): the numerical uncertainty
        p (int): the number of positive samples
        n (int): the number of negative samples
        lower (str): 'min'/'rmin'/'onmax'
        upper (str): 'max'/'rmax' - the type of upper bound

    Returns:
        tuple(float, float): the interval for the accuracy

    Raises:
        ValueError: when the parameters are not suitable for the estimation methods
        or the scores are inconsistent
    """

    lower0 = acc_lower_from(scores=scores, eps=eps, p=p, n=n, lower=lower)
    upper0 = acc_upper_from(scores=scores, eps=eps, p=p, n=n, upper=upper)

    return (lower0, upper0)


def max_acc_lower_from(*, scores: dict, eps: float, p: int, n: int, lower: str = "min"):
    """
    This function applies the estimation schemes to estimate maximum accuracy
    from scores

    Args:
        scores (dict): the reported scores
        eps (float): the numerical uncertainty
        p (int): the number of positive samples
        n (int): the number of negative samples
        lower (str): 'min'

    Returns:
        float: the lower bound for the maximum accuracy

    Raises:
        ValueError: when the parameters are not suitable for the estimation methods
        or the scores are inconsistent
    """

    intervals = prepare_intervals(scores, eps)

    if "auc" not in intervals:
        raise ValueError("auc must be specified")

    if lower == "min":
        lower0 = macc_min(intervals["auc"][0], p, n)
    else:
        raise ValueError(f"unsupported lower bound {lower}")

    return lower0


def max_acc_upper_from(*, scores: dict, eps: float, p: int, n: int, upper: str = "min"):
    """
    This function applies the estimation schemes to estimate maximum accuracy
    from scores

    Args:
        scores (dict): the reported scores
        eps (float): the numerical uncertainty
        p (int): the number of positive samples
        n (int): the number of negative samples
        upper (str): 'max'/'rmax'/'onmax' - the type of upper bound

    Returns:
        float: the upper bound for the maximum accuracy

    Raises:
        ValueError: when the parameters are not suitable for the estimation methods
        or the scores are inconsistent
    """

    intervals = prepare_intervals(scores, eps)

    if "auc" not in intervals:
        raise ValueError("auc must be specified")

    if upper == "max":
        upper0 = acc_max(intervals["auc"][1], p, n)
    elif upper == "rmax":
        upper0 = acc_rmax(intervals["auc"][1], p, n)
    elif upper == "onmax":
        upper0 = acc_onmax(intervals["auc"][1], p, n)
    else:
        raise ValueError(f"unsupported upper bound {upper}")

    return upper0


def max_acc_from(
    *, scores: dict, eps: float, p: int, n: int, lower: str = "min", upper: str = "max"
) -> tuple:
    """
    This function applies the estimation schemes to estimate maximum accuracy
    from scores

    Args:
        scores (dict): the reported scores
        eps (float): the numerical uncertainty
        p (int): the number of positive samples
        n (int): the number of negative samples
        lower (str): 'min'
        upper (str): 'max'/'rmax'/'onmax' - the type of upper bound

    Returns:
        tuple(float, float): the interval for the maximum accuracy

    Raises:
        ValueError: when the parameters are not suitable for the estimation methods
        or the scores are inconsistent
    """

    lower0 = max_acc_lower_from(scores=scores, eps=eps, p=p, n=n, lower=lower)

    upper0 = max_acc_upper_from(scores=scores, eps=eps, p=p, n=n, upper=upper)

    return (lower0, upper0)

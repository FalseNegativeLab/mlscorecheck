"""
This module implements all AUC related functionalities
"""

import numpy as np

from scipy.stats import beta
from scipy.stats import norm as gaussian

from ._utils import translate_scores, prepare_intervals, integrate_roc_curve, integrate_roc_curves

__all__ = [
    "augment_intervals",
    "auc_from",
    "auc_lower_from",
    "auc_upper_from",
    "integrate_roc_curve",
    "integrate_roc_curves",
    "roc_min",
    "roc_max",
    "roc_rmin",
    "roc_rmin_grid",
    "roc_rmin_grid_correction",
    "roc_maxa",
    "roc_onmin",
    "roc_maxa2",
    "auc_min",
    "auc_max",
    "auc_rmin",
    "auc_rmin_grid",
    "auc_maxa",
    "auc_maxa2",
    "auc_amin",
    "auc_armin",
    "auc_amax",
    "auc_onmin",
    "auc_onmin_grad",
    "auc_maxa_grad",
    "auc_min_grad",
    "auc_max_grad",
    "auc_rmin_grad",
    "auc_onmin_profile",
    "auc_maxa_profile",
    "auc_min_profile",
    "auc_max_profile",
    "auc_rmin_profile",
    "check_lower_applicability",
    "check_upper_applicability",
]


def expected_value(a, b, start, end, n):
    aucs = np.linspace(start, end, n)
    aucs = (aucs[1:] + aucs[:-1])/2
    dx = (end - start)/n
    norm = beta.cdf(end, a, b) - beta.cdf(start, a, b)
    pdfs = beta.pdf(aucs, a, b)
    pdfs = pdfs / norm
    return np.sum(aucs * pdfs)*dx


def rline_intersect(sens, spec):
    a = (1 - sens)/(1 - spec)
    b = sens - a*spec
    se0 = (a + b)/(1 + a)
    sp0 = 1 - se0
    return se0, sp0

def rcirc_intersect(sens, spec):
    a = (1 - sens)/(1 - spec)
    b = sens - a*spec
    se0 = (2*b + np.sqrt(4*b**2 - 4*(1 + a**2)*(b**2 - a**2)))/(2*(1 + a**2))
    sp0 = np.sqrt(1 - se0**2)
    return se0, sp0


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


def roc_min(fpr, tpr):
    """
    The minimum ROC curve at fpr, tpr

    Args:
        fpr (float): the false positive rate
        tpr (float): the true positive rate

    Returns:
        np.array, np.array: the fpr and tpr values
    """
    if fpr != 0 and tpr != 1:
        return (np.array([0, fpr, fpr, 1, 1]), np.array([0, 0, tpr, tpr, 1]))
    elif fpr == 0:
        return (np.array([0, fpr, 1, 1]), np.array([0, tpr, tpr, 1]))
    elif tpr == 1:
        return (np.array([0, fpr, fpr, 1]), np.array([0, 0, tpr, tpr]))


def roc_max(fpr, tpr):
    """
    The maximum ROC curve at fpr, tpr

    Args:
        fpr (float): the false positive rate
        tpr (float): the true positive rate

    Returns:
        np.array, np.array: the fpr and tpr values
    """

    if fpr != 1 and tpr != 0:
        return (np.array([0, 0, fpr, fpr, 1]), np.array([0, tpr, tpr, 1, 1]))
    elif fpr == 1:
        return (np.array([0, 0, fpr, fpr]), np.array([0, tpr, tpr, 1]))
    elif tpr == 0:
        return (np.array([0, fpr, fpr, 1]), np.array([tpr, tpr, 1, 1]))


def roc_rmax(fpr, tpr):
    d = max(tpr - fpr, 0)
    return (np.array([0, 0, max(fpr - d, 0), fpr, fpr, max(1 - 2*d, 0), 1]),
            np.array([0, min(2*d, 1), tpr, tpr, min(tpr + d, 1), 1, 1]))


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
        #raise ValueError("the regulated minimum curve does not exist when tpr < fpr")
        return (None, None)

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
        #raise ValueError(
        #    "the maximum accuracy curve does not exist when acc < max(p,n)/(p + n)"
        #)
        return (None, None)

    tpr_a = (acc * (p + n) - n) / p
    fpr_b = 1 - (acc * (p + n) - p) / n

    if fpr_b != 1 and tpr_a != 0:
        return (np.array([0, 0, fpr_b, 1]), np.array([0, tpr_a, 1, 1]))
    elif fpr_b == 1:
        return (np.array([0, 0, 0.5, fpr_b]), np.array([0, tpr_a, tpr_a + (1 - tpr_a)/2, 1]))
    elif tpr_a == 0:
        return (np.array([0, fpr_b/2, fpr_b, 1]), np.array([0, 0.5, 1, 1]))
    

def roc_maxa2(acc, p, n):
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
        #raise ValueError(
        #    "the maximum accuracy curve does not exist when acc < max(p,n)/(p + n)"
        #)
        return (None, None)

    tpr_a = (acc * (p + n) - n) / p
    fpr_b = 1 - (acc * (p + n) - p) / n

    if fpr_b != 1 and tpr_a != 0:
        fprs, tprs = (np.array([0, 0, fpr_b, 1]), np.array([0, tpr_a, 1, 1]))
    elif fpr_b == 1:
        fprs, tprs = (np.array([0, 0, 0.5, fpr_b]), np.array([0, tpr_a, tpr_a + (1 - tpr_a)/2, 1]))
    elif tpr_a == 0:
        fprs, tprs = (np.array([0, fpr_b/2, fpr_b, 1]), np.array([0, 0.5, 1, 1]))

    return tprs, fprs

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


def auc_min_profile(fpr, tpr):
    """
    The profile length of the minimum AUC

    Args:
        fpr (float): upper bound on false positive rate
        tpr (float): lower bound on true positive rate

    Returns:
        float: the profile lenght
    """

    return 1 + tpr


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
    The gradient of the regulated minimum AUC curve

    Args:
        fpr (float): upper bound on false positive rate
        tpr (float): lower bound on true positive rate

    Returns:
        float: the gradient magnitude
    """

    return np.sqrt((tpr-fpr)**2 + (fpr-tpr)**2)


def auc_rmin_profile(fpr, tpr):
    """
    The profile length of the regulated minimum AUC curve

    Args:
        fpr (float): upper bound on false positive rate
        tpr (float): lower bound on true positive rate

    Returns:
        float: the profile length
    """

    fprs, tprs = roc_rmin(fpr, tpr)
    total = 0.0
    for idx in range(len(fprs) - 1):
        total += np.sqrt((fprs[idx] - fprs[idx+1])**2 + (tprs[idx] - tprs[idx+1])**2)
    return float(total)


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


def auc_max_profile(fpr, tpr):
    """
    The profile length of the maximum AUC curve

    Args:
        fpr (float): upper bound on false positive rate
        tpr (float): lower bound on true positive rate

    Returns:
        float: the profile lenght
    """

    return 1 + (1 - tpr)


def auc_rmax(fpr, tpr):
    return integrate_roc_curve(*roc_rmax(fpr, tpr))


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

def auc_maxa2(acc, p, n):
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

    return 1.0 - float(1 - ((1 - acc) * (p + n)) ** 2 / (2 * n * p))


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

    if acc < max(p, n) / (p + n):
        raise ValueError("accuracy too small")

    return - (2*acc - 2)*(n + p)**2/(2*n*p)

def auc_maxa_grad2(fpr, tpr, p, n):
    """
    The gradient magnitude of the amax estimation

    Args:
        acc (float): the accuracy
        p (int): the number of positive samples
        n (int): the number of negative samples

    Returns:
        float: the gradient magnitude 
    """

    acc = ((1 - fpr)*n + tpr*p)/(p + n)

    if acc < max(p, n) / (p + n):
        raise ValueError("accuracy too small")

    dtpr = (fpr*n - p*tpr + p)/n
    dfpr = tpr - 1 - fpr*n/p

    return np.sqrt(dtpr**2 + dfpr**2)


def auc_maxa_profile(acc, p, n):
    """
    The profile length of the amax estimation

    Args:
        acc (float): the accuracy
        p (int): the number of positive samples
        n (int): the number of negative samples

    Returns:
        float: the profile length
    """

    fprs, tprs = roc_maxa(acc, p, n)
    return float(np.sqrt((fprs[1] - fprs[2])**2 + (tprs[1] - tprs[2])**2))

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
    

def auc_onmin_profile(fpr, tpr):
    """
    The profile length of the onmin estimation

    Args:
        acc (float): the accuracy
        p (int): the number of positive samples
        n (int): the number of negative samples

    Returns:
        float: the profile length
    """

    fprs, tprs = roc_onmin(fpr, tpr)
    segment_a = np.sqrt((fprs[0] - fprs[1])**2 + ([tprs[0] - tprs[1]])**2)
    segment_b = np.sqrt((fprs[1] - fprs[2])**2 + ([tprs[1] - tprs[2]])**2)

    return float(segment_a + segment_b)

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
    if upper in ["max","rmax"] and ("fpr" not in intervals or "tpr" not in intervals):
        raise ValueError("fpr, tpr or their complements must be specified")
    if upper in ["amax", "maxa"] and (p is None or n is None):
        raise ValueError("p and n must be specified")
    if upper in ["amax", "maxa"] and "acc" not in intervals:
        raise ValueError("acc must be specified")


def auc_lower_from(
    *, 
    scores: dict, 
    eps: float, 
    p: int = None, 
    n: int = None, 
    lower: str = "min",
    correction: str = None
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

    corr = 1.0

    if lower == "min":
        lower0 = auc_min(intervals["fpr"][1], intervals["tpr"][0])
        if correction == 'gradient':
            corr = auc_min_grad(intervals["fpr"][1], intervals["tpr"][0])
        elif correction == 'profile':
            corr = auc_min_profile(intervals["fpr"][1], intervals["tpr"][0])
    elif lower == "rmin":
        lower0 = auc_rmin(intervals["fpr"][0], intervals["tpr"][1])
        if correction == 'gradient':
            corr = auc_rmin_grad(intervals["fpr"][1], intervals["tpr"][0])
        elif correction == 'profile':
            corr = auc_rmin_profile(intervals["fpr"][1], intervals["tpr"][0])
    elif lower == "onmin":
        lower0 = auc_onmin(intervals["fpr"][0], intervals["tpr"][0])
        if correction == 'gradient':
            corr = auc_onmin_grad(intervals["fpr"][1], intervals["tpr"][0])
        elif correction == 'profile':
            corr = auc_onmin_profile(intervals["fpr"][1], intervals["tpr"][0])
    elif lower == "grmin":
        lower0 = auc_rmin_grid(intervals["fpr"][0], intervals["tpr"][1], p, n)
    elif lower == "amin":
        lower0 = auc_amin(intervals["acc"][0], p, n)
    elif lower == "armin":
        lower0 = auc_armin(intervals["acc"][0], p, n)
    else:
        raise ValueError(f"unsupported lower bound {lower}")

    return lower0, corr


def auc_upper_from(
    *, scores: dict, eps: float, p: int = None, n: int = None, upper: str = "max", correction: str = None
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

    corr = 1.0

    if upper == "max":
        upper0 = auc_max(intervals["fpr"][0], intervals["tpr"][1])
        if correction == 'gradient':
            corr = auc_max_grad(intervals["fpr"][0], intervals["tpr"][1])
        elif correction == 'profile':
            corr = auc_max_profile(intervals["fpr"][0], intervals["tpr"][1])
    elif upper == "rmax":
        upper0 = auc_rmax(intervals["fpr"][0], intervals["tpr"][1])
    elif upper == "amax":
        upper0 = auc_amax(intervals["acc"][1], p, n)
    elif upper == "maxa":
        upper0 = auc_maxa(intervals["acc"][1], p, n)
        if correction == 'gradient':
            corr = auc_maxa_grad(intervals["acc"][1], p, n)
        elif correction == 'profile':
            corr = auc_maxa_profile(intervals["acc"][1], p, n)
    else:
        raise ValueError(f"unsupported upper bound {upper}")

    return upper0, corr


def auc_from(
    *,
    scores: dict,
    eps: float,
    p: int = None,
    n: int = None,
    lower: str = "min",
    upper: str = "max",
    correction: str = None
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
        correction (str): None/'gradient'/'profile'

    Returns:
        tuple(float, float): the interval for the AUC

    Raises:
        ValueError: when the parameters are not suitable for the estimation methods
        or the scores are inconsistent
    """

    try:

        lower0, grad_lower = auc_lower_from(scores=scores, eps=eps, p=p, n=n, lower=lower, correction='gradient')
        upper0, grad_upper = auc_upper_from(scores=scores, eps=eps, p=p, n=n, upper=upper, correction='gradient')

        corr_upper = 1
        corr_lower = 1

        corr_sum = corr_lower + corr_upper

        corr_lower = corr_lower / corr_sum
        corr_upper = corr_upper / corr_sum

        midpoint = lower0 * corr_lower + upper0 * corr_upper

        return (midpoint, midpoint)
    except:
        return np.nan, np.nan

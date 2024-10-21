"""
This module implements all AUC related functionalities
"""

import numpy as np

from scipy.stats import beta
from scipy.stats import norm as gaussian

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
        lower0_min, grad_lower_min = auc_lower_from(scores=scores, eps=eps, p=p, n=n, lower='min', correction='gradient')
        onmin, grad_onmin = auc_lower_from(scores=scores, eps=eps, p=p, n=n, lower='onmin', correction='gradient')

        upper0, grad_upper = auc_upper_from(scores=scores, eps=eps, p=p, n=n, upper=upper, correction='gradient')
        

        vector = np.array([1.0 - scores['spec'], scores['sens']])
        direction = np.array([1.0, 1.0]) / np.sqrt(2.0)
        inner = np.inner(vector, direction)
        intersection = inner * direction
        diff = vector - intersection
        length = np.linalg.norm(diff)
        length_sign = 1 if scores['sens'] > 1 - scores['spec'] else -1

        dist_05 = np.sqrt((scores['sens'] - 0.5)**2 + (scores['spec'] - 0.5)**2)
        dist_0 = (np.sqrt((scores['sens'] - 0)**2 + (scores['spec'] - 0)**2))
        dist_1 = np.sqrt((scores['sens'] - 1)**2 + (scores['spec'] - 1)**2)

        #dist_05 = (np.abs(scores['sens'] - 0.5) + np.abs(scores['spec'] - 0.5))
        #dist_1 = (np.abs(scores['sens'] - 1) + np.abs(scores['spec'] - 1))
        #dist_wall = min(1-scores['sens'],  1-scores['spec']) + 0.01

        dist_05_norm = dist_05 / (np.sqrt(2)/2)
        dist_random = length
        dist_random_norm = dist_random / (np.sqrt(2)/2)

        #corr_lower = 1.0/(dist_random_norm + 0.0001)
        #corr_upper = 1.0/(dist_1 + 0.0001)

        midpoint = lower0 * 0.5 + upper0 * 0.5

        exponent = 1.0

        corr_lower = dist_1 + 0.01
        #corr_upper = 0.5

        corr_upper = dist_05 + 0.0 + 0.1
        #corr_lower = 1 - corr_upper + 0.1

        #corr_upper = (grad_lower + 1)**exponent
        #corr_lower = (grad_upper + 1)**exponent

        corr_upper = 0.01
        corr_lower = 0.01

        #corr_upper = (1 + (midpoint - 0.75))**0
        #corr_lower = (1 - (midpoint - 0.75))**0

        #corr_upper = dist_0
        #corr_lower = 1.0 - corr_upper

        #corr_lower = dist_1 + 0.1
        #corr_upper = dist_random + 0.1

        #corr_upper = 1 - corr_lower + 0.1
        #corr_lower = 0.5 + 0.01
        #corr_lower = 1.0 - corr_upper + 0.0

        exponent = 1.0

        #corr_upper = (grad_lower)**exponent
        #corr_lower = (grad_upper)**exponent

        # arbitrary
        #corr_lower = (dist_1)**exponent
        #corr_upper = (dist_random)**exponent
        #corr_lower = 1.0 - corr_upper
        #corr_upper = 1 - corr_lower

        #corr_lower = 1/(dist_random_norm + 0.01)
        #corr_upper = 3
        #corr_upper = 1/(dist_1 + 0.01)
        #corr_lower = 1

        #corr_upper = 1/(dist_1 + 0.01)
        #corr_lower = 1/((1 - dist_1) + 0.01)

        #dist_random_norm = 

        #midpoint = (upper0 + ((1 - dist_random_norm)*lower0_min + (dist_random_norm)*lower0))/2

        midpoint = (lower0 + upper0)/2

        #corr_lower = 0.01
        #corr_upper = 0.01

        beta0 = 20
        alpha_lower = lower0 * beta0
        alpha_upper = upper0 * beta0
        alpha_mid = midpoint * beta0

        #midpoint = expected_value(alpha_mid, beta0 - alpha_mid, lower0, upper0, 10000)

        #lower0_new = expected_value(alpha_lower, beta0 - alpha_lower, lower0, upper0, 10000)
        #upper0_new = expected_value(alpha_upper, beta0 - alpha_upper, lower0, upper0, 10000)

        #midpoint = np.mean([expected_value(tmp, beta0 - tmp, lower0, upper0, 1000) for tmp in np.linspace(alpha_lower, alpha_upper, 2)])

        #print(lower0, upper0, corr_lower, corr_upper, beta0, alpha_lower, alpha_upper, lower0_new, upper0_new)

        #lower0 = lower0_new
        #upper0 = upper0_new


        """points = np.linspace(alpha_lower, alpha_upper, 50)
        perc5 = []
        perc95 = []
        for point in points:
            perc5.append(beta.ppf([0.01], point, beta0 - point)[0])
            perc95.append(beta.ppf([0.99], point, beta0 - point)[0])
        perc5 = np.array(perc5)
        perc95 = np.array(perc95)

        idx = np.argmin(np.abs(lower0 - perc5)**2 + np.abs(upper0 - perc95)**2)

        midpoint = points[idx] / beta0"""


        #corr_lower = (upper0 - onmin)**2
        #corr_upper = (onmin - lower0)**2

        #corr_lower = dist_1
        if length_sign == -1:
            dist_random = 0.0

        dist01 = np.sqrt((scores['sens'] - 0)**2 + (scores['spec'] - 1)**2)
        dist10 = np.sqrt((scores['sens'] - 1)**2 + (scores['spec'] - 0)**2)

        dist_corner = 1 - min(dist01, dist10)

        area = min(1 - (scores['sens'] + scores['spec'])/2, (scores['sens'] + scores['spec'])/2 - 0.5)
        #area = min(dist_random, np.sqrt(2)/2 - dist_random)

        #midpoint = 0.5 + (dist_random/(np.sqrt(2)/2))*0.5 + dist_random**2*dist_corner

        #midpoint = 0.5 + (dist_random/(np.sqrt(2)/2))*0.5 + 1/min(p, n)

        
        se0, sp0 = rline_intersect(scores['sens'], scores['spec'])
        se1, sp1 = rcirc_intersect(scores['sens'], scores['spec'])

        dist_circ = np.sqrt((scores['sens'] - se1)**2 + (scores['spec'] - sp1)**2)
        dist_rline = np.sqrt((scores['sens'] - se0)**2 + (scores['spec'] - sp0)**2)

        if (scores['sens'] < 0.001 and scores['spec'] > 0.999) or (scores['spec'] < 0.001 and scores['sens'] > 0.999):
            return (0.5, 0.5)
        
        at = 0.75
        
        if scores['sens']**2 + scores['spec']**2 < 1:
            dist_rline = np.sqrt((scores['sens'] - se0)**2 + (scores['spec'] - sp0)**2)
            dist_circ = np.sqrt((scores['sens'] - se1)**2 + (scores['spec'] - sp1)**2)
            ratio = dist_rline / (dist_rline + dist_circ)
            midpoint = (at - 0.5)*ratio + 0.5
            circ_sign = -1
        else:
            dist_1 = np.sqrt((scores['sens'] - 1)**2 + (scores['spec'] - 1)**2)
            dist_circ = np.sqrt((scores['sens'] - se1)**2 + (scores['spec'] - sp1)**2)
            ratio = dist_circ / (dist_1 + dist_circ)
            midpoint = (1 - at)*ratio + at
            circ_sign = 1
        

        """dist_rline = np.sqrt((scores['sens'] - se0)**2 + (scores['spec'] - sp0)**2)
        dist_1 = np.sqrt((scores['sens'] - 1)**2 + (scores['spec'] - 1)**2)

        ratio = dist_rline / (dist_rline + dist_1)
        midpoint = 0.5 + ratio * 0.5"""

        #corr_lower = np.sqrt(2) - (dist_0)
        #corr_upper = (dist_0)


        #corr_lower = 1
        #corr_upper = (scores['sens']) + (scores['spec'])

        lower_extremity = 1
        upper_extremity = 1
        if lower == 'min' and upper == 'max':
            #lower_extremity = 2/4
            #upper_extremity = (1 - scores['spec'] + 1 - scores['sens'])/2
            upper_extremity = auc_max_grad(1 - scores['spec'], scores['sens'])**0.5
            lower_extremity = auc_min_grad(1 - scores['spec'], scores['sens'])**0.5
        
        if lower == 'rmin' and upper == 'max':
            #lower_extremity = (scores['sens'] - (1 - scores['spec']))*2/4
            #upper_extremity = (1 - scores['spec'] + 1 - scores['sens'])/2
            upper_extremity = auc_max_grad(1 - scores['spec'], scores['sens'])**0.5
            lower_extremity = auc_rmin_grad(1 - scores['spec'], scores['sens'])**0.5
        
        if lower == 'rmin' and upper == 'maxa':

            #lower_extremity = (scores['sens'] - (1 - scores['spec']))*2/2 + 0.25

            #fprs, tprs = roc_maxa(scores['acc'], p, n)

            #upper_extremity = np.abs(fprs[2] - fprs[1] - (tprs[2] - tprs[1])) + 0.25
            upper_extremity = auc_maxa_grad2(1 - scores['spec'], scores['sens'], p, n)**0.5
            lower_extremity = auc_rmin_grad(1 - scores['spec'], scores['sens'])**0.5


        #lower_extremity = 1 - dist_1
        #upper_extremity = dist_1


        #upper_extremity = gaussian.pdf(lower0**0.5, 0, 0.65)
        #lower_extremity = gaussian.pdf((1 - upper0)**0.5, 0, 0.65)

        p1 = p/(p + n)
        p0 = n/(p + n)

        prob = np.sqrt(p1*p0)

        upper_extremity = prob**(scores['sens']) * prob**(scores['spec'])
        lower_extremity = prob**(1 - scores['sens']) * prob**(1 - scores['spec'])

        corr_upper = lower_extremity
        corr_lower = upper_extremity

        corr_upper = 1
        corr_lower = 1

        #corr_lower = 2
        #corr_upper = auc_rmin_profile(1 - scores['spec'], scores['sens'])
        #lower0 = 0.5
        #upper0 = 1.0

        #corr_upper = ((1 - scores['spec']) + (1 - scores['spec'])*scores['sens'] + (1 - scores['sens'])*scores['spec'] + (1 - scores['sens']))
        #corr_upper = lower0
        corr_upper = ((lower0))**0.2
        corr_lower = ((1 - upper0))**0.2

        corr_upper = 1
        corr_lower = 1

        corr_sum = corr_lower + corr_upper

        

        corr_lower = corr_lower / corr_sum
        corr_upper = corr_upper / corr_sum

        #print(corr_lower, corr_upper, alpha_lower, alpha_upper)

        #norm = np.sqrt(scores['sens']**2 + (scores['spec']**2))/np.sqrt(2)
        #midpoint = (dist_random/(np.sqrt(2)) + 0.5)

        #midpoint = dist_circ / (np.sqrt(2) - 1) * 0.25 * circ_sign + 0.75
        tmp = (auc_min(1 - scores['spec'], scores['sens']) + auc_max(1 - scores['spec'], scores['sens']))/2.0
        #scaler = (tmp - 0.5)*2
        #midpoint = tmp*norm + (1 - tmp)*tmp
        #midpoint = norm
        #midpoint = min(tmp*1.035, 1.0)
        #midpoint = ((tmp + norm)/2.0)**0.8
        #midpoint = dist_random/(np.sqrt(2)) + 0.5
        #midpoint = (((dist_rline)*1 + (dist_1)*0.5)/(dist_rline + dist_1))

        midpoint = lower0 * corr_lower + upper0 * corr_upper
        #midpoint = np.sqrt(scores['sens']**2 + (scores['spec']**2))/(np.sqrt(2))
        #midpoint = (lower0 + upper0)/2
        #tmp = max(dist_0 - np.sqrt(2)/2, 0) / (np.sqrt(2) - np.sqrt(2)/2)
        #midpoint = 0.5 + tmp**1.2*0.5

        

        return (midpoint, midpoint)
    except:
        return np.nan, np.nan

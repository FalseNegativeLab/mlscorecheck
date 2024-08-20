"""
This module implements all AUC related functionalities
"""

import numpy as np

__all__ = [
    "prepare_intervals_for_auc_estimation",
    "auc_from_sens_spec",
    "acc_from_auc",
    "translate",
    "auc_min",
    "auc_max",
    "auc_cmin",
    "auc_amax",
    "auc_amin",
    "auc_acmin",
    "acc_min",
    "acc_max",
    "acc_cmax"
]


def translate(scores: dict) -> dict:
    """
    Translates the scores

    Args:
        scores (dict): the dict of scores

    Returns:
        dict: the translated scores
    """
    scores = {**scores}
    if "tpr" in scores:
        if not "sens" in scores:
            scores["sens"] = scores["tpr"]
        else:
            raise ValueError("tpr and sens cannot be specified together")
    if "tnr" in scores:
        if not "spec" in scores:
            scores["spec"] = scores["tnr"]
        else:
            raise ValueError("tnr and spec cannot be specified together")
    if "fpr" in scores:
        if not "spec" in scores:
            scores["spec"] = 1 - scores["fpr"]
        else:
            raise ValueError("fpr and spec cannot be specified together")
    return scores


def prepare_intervals_for_auc_estimation(
    scores: dict, eps: float, p: int, n: int
) -> dict:
    """
    Prepares all intervals

    Args:
        scores (dict): the dictionary of reported scores
        eps (float): the numerical uncertainty
        p (int): the number of positive samples
        n (int): the number of negative samples

    Returns:
        dict: the intervals
    """

    results = {
        score: (max(scores[score] - eps, 0), min(scores[score] + eps, 1))
        for score in ["acc", "sens", "spec"]
        if score in scores
    }

    if "sens" not in results:
        lower = max(((results["acc"][0]) * (p + n) - (results["spec"][1] * n)) / p, 0)
        upper = min(((results["acc"][1]) * (p + n) - (results["spec"][0] * n)) / p, 1)
        results["sens"] = (lower, upper)
    if "spec" not in results:
        lower = max(((results["acc"][0]) * (p + n) - (results["sens"][1] * p)) / n, 0)
        upper = min(((results["acc"][1]) * (p + n) - (results["sens"][0] * p)) / n, 1)
        results["spec"] = (lower, upper)
    if "acc" not in results:
        lower = max((results["sens"][0] * p + results["spec"][0] * n) / (p + n), 0)
        upper = min((results["sens"][1] * p + results["spec"][1] * n) / (p + n), 1)
        results["acc"] = (lower, upper)

    return results


def auc_from_sens_spec_wrapper(
    *, 
    scores: dict, 
    eps: float, 
    p: int, 
    n: int, 
    lower: str = "min", 
    upper: str = "max"):
    try:
        return auc_from_sens_spec(
            scores=scores,
            eps=eps,
            p=p,
            n=n,
            lower=lower,
            upper=upper,
            raise_errors=True
        )
    except:
        return None


def auc_min(sens, spec):
    """
    The area under the minimum curve at sens, spec

    Args:
        sens (float): lower bound on sensitivity
        spec (float): lower bound on specificity
    
    Returns:
        float: the area
    """
    
    return sens * spec

def auc_cmin(sens, spec):
    """
    The area under the corrected minimum curve at sens, spec

    Args:
        sens (float): lower bound on sensitivity
        spec (float): lower bound on specificity
    
    Returns:
        float: the area
    """
    
    if sens < 1 - spec:
            raise ValueError(
                'sens >= 1 - spec does not hold for "\
                            "the corrected minimum curve'
            )
    return 0.5 + (1 - (sens + spec)) ** 2 / 2.0

def auc_max(sens, spec):
    """
    The area under the maximum curve at sens, spec

    Args:
        sens (float): upper bound on sensitivity
        spec (float): upper bound on specificity
    
    Returns:
        float: the area
    """
    
    return 1 - (1 - sens) * (1 - spec)

def auc_amax(acc, p, n):
    """
    The area under the maximum accuracy curve at acc

    Args:
        acc (float): upper bound on accuracy
        p (int): the number of positive test samples
        n (int): the number of negative test samples
    
    Returns:
        float: the area
    """

    if not acc >= max(p, n) / (p + n):
        raise ValueError("accuracy too small")
    return 1 - ((1 - acc) * (p + n)) ** 2 / (2 * n * p)

def auc_amin(acc, p, n):
    """
    The smallest area under the minimum curve curve at acc

    Args:
        acc (float): lower bound on accuracy
        p (int): the number of positive test samples
        n (int): the number of negative test samples
    
    Returns:
        float: the area
    """
    if acc < max(p, n)/(p + n):
        return 0.0
    
    return acc - (1 - acc) * max(p, n)/min(p, n)

def auc_acmin(acc, p, n):
    """
    The smallest area under the corrected minimum curve curve at acc

    Args:
        acc (float): lower bound on accuracy
        p (int): the number of positive test samples
        n (int): the number of negative test samples
    
    Returns:
        float: the area
    """
    if acc < max(p, n)/(p + n):
        raise ValueError("accuracy too small")
    
    return auc_amin(acc, p, n)**2/2 + 0.5

def acc_min(auc, p, n):
    """
    The minimum implied maximum accuracy given an AUC

    Args:
        auc (float): lower bound on AUC
        p (int): the number of positive test samples
        n (int): the number of negative test samples
    
    Returns:
        float: the area
    """
    if auc < 1 - min(p, n)/(2*max(p, n)):
        return 1 - (np.sqrt(2 * p * n - 2 * auc * p * n)) / (p + n)
    else:
        return max(p, n)/(p + n)

def acc_max(auc, p, n):
    """
    The maximum implied accuracy given an AUC

    Args:
        auc (float): upper bound on AUC
        p (int): the number of positive test samples
        n (int): the number of negative test samples
    
    Returns:
        float: the area
    """
    return (auc * min(p, n) + max(p, n)) / (p + n)

def acc_cmax(auc, p, n):
    """
    The maximum implied accuracy on a corrected minimum curve given an AUC

    Args:
        auc (float): upper bound on AUC
        p (int): the number of positive test samples
        n (int): the number of negative test samples
    
    Returns:
        float: the area
    """
    return (max(p, n) + min(p, n) * np.sqrt(2 * (auc - 0.5))) / (p + n)

def auc_from_sens_spec(
    *, 
    scores: dict, 
    eps: float, 
    p: int, 
    n: int, 
    lower: str = "min", 
    upper: str = "max",
    raise_errors: bool = False
) -> tuple:
    """
    This module applies the estimation scheme A to estimate AUC from scores

    Args:
        scores (dict): the reported scores
        eps (float): the numerical uncertainty
        p (int): the number of positive samples
        n (int): the number of negative samples
        lower (str): ('min'/'cmin') - the type of estimation for the lower bound
        upper (str): ('max'/'amax') - the type of estimation for the upper bound

    Returns:
        tuple(float, float): the interval for the AUC
    """

    if not raise_errors:
        return auc_from_sens_spec_wrapper(
            scores=scores,
            eps=eps,
            p=p,
            n=n,
            lower=lower,
            upper=upper
        )

    scores = translate(scores)

    if ("sens" in scores) + ("spec" in scores) + ("acc" in scores) < 2:
        raise ValueError("Not enough scores specified for the estimation")

    intervals = prepare_intervals_for_auc_estimation(scores, eps, p, n)

    if lower == "min":
        lower0 = intervals["sens"][0] * intervals["spec"][0]
    elif lower == "cmin":
        if intervals["sens"][0] < 1 - intervals["spec"][0]:
            raise ValueError(
                'sens >= 1 - spec does not hold for "\
                            "the corrected minimum curve'
            )
        lower0 = 0.5 + (1 - (intervals["sens"][0] + intervals["spec"][0])) ** 2 / 2.0
    else:
        raise ValueError("Unsupported lower bound")

    if upper == "max":
        upper0 = 1 - (1 - intervals["sens"][1]) * (1 - intervals["spec"][1])
    elif upper == "amax":
        if not intervals["acc"][0] >= max(p, n) / (p + n):
            raise ValueError("accuracy too small")
        upper0 = 1 - ((1 - intervals["acc"][1]) * (p + n)) ** 2 / (2 * n * p)
    else:
        raise ValueError("Unsupported upper bound")

    return (float(lower0), float(upper0))


def acc_from_auc_wrapper(
    *, 
    scores: dict, 
    eps: float, 
    p: int, 
    n: int, 
    upper: str = "max"
):
    try:
        return acc_from_auc(
            scores=scores,
            eps=eps,
            p=p,
            n=n,
            upper=upper,
            raise_errors=True
        )
    except:
        return None
    

def acc_from_auc(
    *, 
    scores: dict, 
    eps: float, 
    p: int, 
    n: int, 
    upper: str = "max",
    raise_errors: bool = False
) -> tuple:
    """
    This module applies the estimation scheme A to estimate AUC from scores

    Args:
        scores (dict): the reported scores
        eps (float): the numerical uncertainty
        p (int): the number of positive samples
        n (int): the number of negative samples
        upper (str): 'max'/'cmax' - the type of upper bound

    Returns:
        tuple(float, float): the interval for the maximum accuracy
    """

    if not raise_errors:
        return acc_from_auc_wrapper(
            scores=scores,
            eps=eps,
            p=p,
            n=n,
            upper=upper
        )

    scores = translate(scores)

    auc = (max(scores["auc"] - eps, 0), min(scores["auc"] + eps, 1))

    #if np.sqrt((1 - auc[0])*2*p*n) > min(p, n):
    if auc[0] < 1 - min(p, n)/(2*max(p, n)):
        lower = 1 - (np.sqrt(2 * p * n - 2 * auc[0] * p * n)) / (p + n)
    else:
        lower = max(p, n)/(p + n)
        #raise ValueError("AUC too small")
    
    if upper == "max":
        upper = (auc[1] * min(p, n) + max(p, n)) / (p + n)
    else:
        upper = (max(p, n) + min(p, n) * np.sqrt(2 * (auc[1] - 0.5))) / (p + n)

    return (float(lower), float(upper))


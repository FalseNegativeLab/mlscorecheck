"""
This module implements all AUC related functionalities
"""

import numpy as np
from sklearn.metrics import roc_auc_score

__all__ = [
    "prepare_intervals_for_auc_estimation",
    "auc_from_sens_spec",
    "acc_from_auc",
    "auc_from_sens_spec_kfold",
    #'generate_sens_spec_acc_problem',
    #'generate_kfold_sens_spec_acc_problem',
    "generate_average",
    "generate_kfold_sens_spec_fix_problem",
    "R",
    "translate",
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


def generate_average(avg_value, n_items, lower_bound=None, random_state=None):
    random_state = (
        np.random.RandomState(random_state)
        if not isinstance(random_state, np.random.RandomState)
        else random_state
    )

    if lower_bound is not None:
        if avg_value < lower_bound:
            raise ValueError("The average value cannot be less than the lower bound")

    values = np.repeat(avg_value, n_items)

    indices = list(range(n_items))

    for _ in range(n_items * 10):
        a, b = random_state.choice(indices, 2, replace=False)
        if random_state.randint(2) == 0:
            dist = min(values[a], 1 - values[a], values[b], 1 - values[b])
            d = random_state.random() * dist

            if lower_bound is not None and values[b] - d < lower_bound:
                d = values[b] - lower_bound
            values[a] += d
            values[b] -= d
        else:
            mean = (values[a] + values[b]) / 2
            values[a] = (values[a] + mean) / 2
            values[b] = (values[b] + mean) / 2

    return values.astype(float)


"""
def generate_sens_spec_acc_problem(
        *,
        n_swaps : float = 0.3,
        max_acc : bool = False,
        p : int = None,
        n : int = None,
        random_state = None
    ) -> dict:

    random_state = (np.random.RandomState(random_state) 
                    if not isinstance(random_state, np.random.RandomState) 
                    else random_state)

    if p is None or n is None:
        size = random_state.randint(100, 1000)

        labels = np.sort(random_state.randint(2, size=size))

        p = int(np.sum(labels))
        n = size - p

        preds = np.sort(random_state.rand(size))
    else:
        size = p + n
        labels = np.hstack([np.repeat(0, n), np.repeat(1, p)])
        preds = np.sort(random_state.rand(size))

    for _ in range(int(size*n_swaps)):
        idx, jdx = random_state.randint(size, size=2)
        preds[idx], preds[jdx] = preds[jdx], preds[idx]

    if not max_acc:
        th = 0.5

        tp = np.sum((preds >= th) & (labels == 1))
        tn = np.sum((preds < th) & (labels == 0))

        result = {
            'acc': float((tp + tn) / (p + n)),
            'sens': float(tp / p),
            'spec': float(tn / n),
            'p': p,
            'n': n,
            'auc': float(roc_auc_score(labels, preds)),
            'labels': labels,
            'preds': preds,
            'th': 0.5
        }
    else:
        tp = p
        tn = 0
        sorting = np.argsort(preds)
        labels = labels[sorting]
        preds = preds[sorting]

        max_tp = tp
        max_tn = 0
        max_th = 0

        for (label, pred) in zip(labels, preds):
            if label == 0:
                tn += 1
            if label == 1:
                tp -= 1
            if tp + tn > max_tp + max_tn:
                max_tp = tp
                max_tn = tn
                max_th = pred
        
        result = {
            'acc': float((max_tp + max_tn) / (p + n)),
            'sens': float(max_tp / p),
            'spec': float(max_tn / n),
            'p': p,
            'n': n,
            'auc': float(roc_auc_score(labels, preds)),
            'labels': labels,
            'preds': preds,
            'th': float(max_th)
        }

    return result
"""

"""
def generate_kfold_sens_spec_acc_problem(
        n_folds : int | None = None,
        n_swaps : float = 0.3,
        max_acc : bool = False,
        random_state = None
    ) -> dict:
    
    random_state = (np.random.RandomState(random_state) 
                    if not isinstance(random_state, np.random.RandomState) 
                    else random_state)

    n_folds = n_folds if n_folds is not None else random_state.randint(2, 20)

    if not max_acc:
        all_results = [generate_sens_spec_acc_problem(
                            n_swaps=n_swaps, 
                            max_acc=False, 
                            random_state=random_state) 
                       for _ in range(n_folds)]
    else:
        p = np.random.randint(100, 1000)
        n = np.random.randint(100, 1000)
        p = (p // n_folds) * n_folds
        n = (n // n_folds) * n_folds

        all_results = [generate_sens_spec_acc_problem(
                            n_swaps=n_swaps, 
                            max_acc=True,
                            p=int(p/n_folds),
                            n=int(n/n_folds),
                            random_state=random_state) 
                       for _ in range(n_folds)]
    results = {
        'acc': float(np.mean([res['acc'] for res in all_results])),
        'sens': float(np.mean([res['sens'] for res in all_results])),
        'spec': float(np.mean([res['spec'] for res in all_results])),
        'auc': float(np.mean([res['auc'] for res in all_results])),
        'p': int(np.sum([res['p'] for res in all_results])),
        'n': int(np.sum([res['n'] for res in all_results])),
        'k': n_folds,
        'details': all_results
    }

    return results
"""


def generate_kfold_sens_spec_fix_problem(
    *, sens, spec, k, sens_lower_bound=None, spec_lower_bound=None, random_state=None
):
    return {
        "sens": generate_average(sens, k, sens_lower_bound, random_state),
        "spec": generate_average(spec, k, spec_lower_bound, random_state),
    }


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


def auc_from_sens_spec(
    *, scores: dict, eps: float, p: int, n: int, lower: str = "min", upper: str = "max"
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

    if ("sens" in scores) + ("spec" in scores) + ("acc" in scores) < 2:
        raise ValueError("Not enough scores specified for the estimation")

    scores = translate(scores)

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


def R(x: float, k: int) -> list:
    result = []
    x = x * k
    while x >= 1:
        result.append(1)
        x = x - 1
    result.append(x)
    while len(result) < k:
        result.append(0)

    return result


def auc_from_sens_spec_kfold(
    *,
    scores: dict,
    eps: float,
    p: int,
    n: int,
    lower: str = "min",
    upper: str = "max",
    k: int = None
) -> tuple:
    """
    This module applies the estimation scheme A to estimate AUC from scores
    in k-fold evaluation

    Args:
        scores (dict): the reported scores average scores
        eps (float): the numerical uncertainty
        p (int): the number of positive samples
        n (int): the number of negative samples
        lower (str): ('min'/'cmin') - the type of estimation for the lower bound
        upper (str): ('max'/'amax') - the type of estimation for the upper bound
        k (int): the number of folds (if any)

    Returns:
        tuple(float, float): the interval for the average AUC
    """

    if ("sens" in scores) + ("spec" in scores) + ("acc" in scores) < 2:
        raise ValueError("Not enough scores specified for the estimation")

    if p is None or n is None:
        raise ValueError("For k-fold estimation p and n are needed")
    if p % k != 0 or n % k != 0:
        raise ValueError("For k-fold, p and n must be divisible by k")

    scores = translate(scores)

    intervals = prepare_intervals_for_auc_estimation(scores, eps, p, n)

    if lower == "min":
        RL_avg_sens = R(intervals["sens"][0], k)
        R1mU_avg_spec = R(intervals["spec"][1], k)

        lower0 = np.mean([a * b for a, b in zip(RL_avg_sens, R1mU_avg_spec[::-1])])
    elif lower == "cmin":
        if intervals["sens"][0] < 1 - intervals["spec"][0]:
            raise ValueError(
                'sens >= 1 - spec does not hold for "\
                            "the corrected minimum curve'
            )
        lower0 = 0.5 + (1 - intervals["spec"][0] - intervals["sens"][0]) ** 2 / 2.0
    else:
        raise ValueError("Unsupported lower bound")

    if upper == "max":
        R1mU_avg_sens = R(intervals["sens"][1], k)
        RL_avg_spec = R(intervals["spec"][0], k)

        upper0 = 1 - np.mean(
            [(1 - a) * (1 - b) for a, b in zip(R1mU_avg_sens, RL_avg_spec[::-1])]
        )
    elif upper == "amax":
        if not intervals["acc"][0] >= max(p, n) / (p + n):
            raise ValueError("accuracy too small")

        upper0 = 1 - ((1 - intervals["acc"][1]) * (p + n)) ** 2 / (2 * n * p)
    else:
        raise ValueError("Unsupported upper bound")

    return (float(lower0), float(upper0))


def acc_from_auc(
    *, scores: dict, eps: float, p: int, n: int, upper: str = "max"
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

    scores = translate(scores)

    auc = (max(scores["auc"] - eps, 0), min(scores["auc"] + eps, 1))

    if not auc[0] >= 1 - min(p, n) / (2 * max(p, n)):
        raise ValueError("AUC too small")

    lower = 1 - (2 * np.sqrt(p * n - auc[0] * p * n)) / (p + n)
    if upper == "max":
        upper = (auc[1] * max(p, n) + min(p, n)) / (p + n)
    else:
        upper = (max(p, n) + min(p, n) * np.sqrt(2 * (auc[1] - 0.5))) / (p + n)

    return (float(lower), float(upper))

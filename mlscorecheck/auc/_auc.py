"""
This module implements all AUC related functionalities
"""

import numpy as np

__all__ = [
    'prepare_intervals_for_auc_estimation',
    'auc_from',
    'acc_from'
]

def prepare_intervals_for_auc_estimation(scores: dict, eps: float, p: int, n: int) -> dict:
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

    results =  {score: (max(scores[score] - eps, 0), min(scores[score] + eps, 1))
            for score in ['acc', 'sens', 'spec'] if score in scores}

    if 'sens' not in results:
        lower = max(((results['acc'][0]) * (p + n) - (results['spec'][1] * n)) / p, 0)
        upper = min(((results['acc'][1]) * (p + n) - (results['spec'][0] * n)) / p, 1)
        results['sens'] = (lower, upper)
    if 'spec' not in results:
        lower = max(((results['acc'][0]) * (p + n) - (results['sens'][1] * p)) / n, 0)
        upper = min(((results['acc'][1]) * (p + n) - (results['sens'][0] * p)) / n, 1)
        results['spec'] = (lower, upper)
    if 'acc' not in results:
        lower = max((results['sens'][0] * p + results['spec'][0] * n) / (p + n), 0)
        upper = min((results['sens'][1] * p + results['spec'][1] * n) / (p + n), 1)
        results['acc'] = (lower, upper)

    return results

def auc_from(*, scores: dict, eps: float, p: int, n: int, lower: str = 'min', upper: str = 'max'):
    """
    This module applies the estimation scheme A to estimate AUC from scores

    Args:
        scores (dict): the reported scores
        eps (float): the numerical uncertainty
        p (int): the number of positive samples
        n (int): the number of negative samples
        method (str): ('A'/'B') - the type of estimation
    """

    if ('sens' in scores) + ('spec' in scores) + ('acc' in scores) < 2:
        raise ValueError('Not enough scores specified for the estimation')

    intervals = prepare_intervals_for_auc_estimation(scores, eps, p, n)

    if lower == 'min':
        lower0 = intervals['sens'][0] * intervals['spec'][0]
    elif lower == 'cmin':
        tmp = [
            abs(1 - intervals['sens'][0] + intervals['spec'][0]),
            abs(1 - intervals['sens'][0] + intervals['spec'][1]),
            abs(1 - intervals['sens'][1] + intervals['spec'][0]),
            abs(1 - intervals['sens'][1] + intervals['spec'][1])
            ]
        min_idx = np.argmin(tmp)
        if min_idx == 0:
            sens_idx, spec_idx = 0, 0
        elif min_idx == 1:
            sens_idx, spec_idx = 0, 1
        elif min_idx == 2:
            sens_idx, spec_idx = 1, 0
        elif min_idx == 3:
            sens_idx, spec_idx = 1, 1
        lower0 = 0.5 + (1 - (intervals['sens'][sens_idx] + intervals['spec'][spec_idx]))**2 / 2.0
    else:
        raise ValueError('Unsupported lower bound')

    if upper == 'max':
        upper0 = 1 - (1 - intervals['sens'][1]) *(1 - intervals['spec'][1])
    elif upper == 'max-acc':
        if not (intervals['acc'][0] >= max(p, n)/(p + n)):
            raise ValueError('accuracy too small')
        upper0 = 1 - ((1 - intervals['acc'][1]) * (p + n))**2 / (2*n*p)
    else:
        raise ValueError('Unsupported upper bound')

    return (lower0, upper0)

def acc_from(*, scores: dict, eps: float, p: int, n: int):
    """
    This module applies the estimation scheme A to estimate AUC from scores

    Args:
        scores (dict): the reported scores
        eps (float): the numerical uncertainty
        p (int): the number of positive samples
        n (int): the number of negative samples
        method (str): ('A'/'B') - the type of estimation
    """

    auc = (max(scores['auc'] - eps, 0), min(scores['auc'] + eps, 1))

    if not (auc[0] >= 1 - min(p, n)/(2*max(p, n))):
        raise ValueError('AUC too small')

    lower = 1 - (2*np.sqrt(p*n - auc[0] * p * n)) / (p + n)
    upper = (auc[1] * max(p, n) + min(p, n)) / (p + n)

    return (lower, upper)

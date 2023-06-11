"""
This module implements a random problem generator
"""

import numpy as np

__all__ = ['generate_problem',
            'generate_problem_tp0']

def generate_problem(*, max_p=1000, max_n=1000, random_seed=None):
    """
    Generates a random problem

    Args:
        max_p (int): the maximum number of positives
        max_n (int): the maximum number of negatives
        random_seed (int, optional): the random seed to use

    Returns:
        dict: the problem
    """
    random_state = np.random.RandomState(random_seed)

    p = random_state.randint(2, max_p+1)
    n = random_state.randint(2, max_n+1)
    tp = random_state.randint(1, p)
    tn = random_state.randint(1, n)
    fp = n - tn
    fn = p - tp

    return {'p': p, 'n': n, 'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn}

def generate_problem_tp0(*, max_p=1000, max_n=1000, random_seed=None):
    """
    Generates a random problem

    Args:
        max_p (int): the maximum number of positives
        max_n (int): the maximum number of negatives
        random_seed (int, optional): the random seed to use

    Returns:
        dict: the problem
    """
    random_state = np.random.RandomState(random_seed)

    p = random_state.randint(2, max_p+1)
    n = random_state.randint(2, max_n+1)
    tp = 0
    tn = random_state.randint(1, n)
    fp = n - tn
    fn = p - tp

    return {'p': p, 'n': n, 'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn}

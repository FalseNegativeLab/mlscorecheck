"""
This module implements a random problem generator
"""

import numpy as np

__all__ = ['generate_problems',
            'generate_1_problem']

def generate_1_problem(*,
                        max_p=1000,
                        max_n=1000,
                        zeros=None,
                        add_complements=False,
                        random_state=None):
    """
    Generates a random problem

    Args:
        max_p (int): the maximum number of positives
        max_n (int): the maximum number of negatives
        zeros (None/list): the list of items to set to zero
        add_complements (bool): whether to add the complements
        random_state (int, optional): the random seed to use

    Returns:
        dict: the problem
    """
    if random_state is None or isinstance(random_state, int):
        random_state = np.random.RandomState(random_state)

    if zeros is None:
        zeros = []

    p = random_state.randint(2, max_p+1)
    n = random_state.randint(2, max_n+1)

    if 'tp' in zeros:
        tp = 0
    elif 'fn' in zeros:
        tp = p
    else:
        tp = random_state.randint(1, p)

    if 'tn' in zeros:
        tn = 0
    elif 'fp' in zeros:
        tn = n
    else:
        tn = random_state.randint(1, n)

    result = {'p': p, 'n': n, 'tp': tp, 'tn': tn}

    if add_complements:
        result['fn'] = p - tp
        result['fp'] = n - tn

    return result, {'p': result['p'], 'n': result['n']}

def generate_problems(*,
                        n_problems=1,
                        max_p=1000,
                        max_n=1000,
                        zeros=None,
                        add_complements=False,
                        random_state=None):
    """
    Generates a random problem

    Args:
        max_p (int): the maximum number of positives
        max_n (int): the maximum number of negatives
        zeros (None/list): the list of items to set to zero
        add_complements (bool): whether to add the complements
        random_state (int, optional): the random seed to use

    Returns:
        dict,dict/list(dict),list(dict): the evaluation and the problem or
                                            a list of evaluations and corresponding problems
    """

    if random_state is None or isinstance(random_state, int):
        random_state = np.random.RandomState(random_state)

    evaluations, problems = [], []
    for _ in range(n_problems):
        evaluation, problem = generate_1_problem(max_p=max_p,
                                    max_n=max_n,
                                    zeros=zeros,
                                    add_complements=add_complements,
                                    random_state=random_state)
        evaluations.append(evaluation)
        problems.append(problem)

    return (evaluations[0], problems[0]) if n_problems == 1 else (evaluations, problems)

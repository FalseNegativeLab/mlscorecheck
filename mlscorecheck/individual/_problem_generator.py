"""
This module implements a random problem generator
"""

import numpy as np

from ._calculate_scores import calculate_scores

__all__ = ['generate_problems',
            'generate_1_problem',
            'generate_problem_and_scores']

def generate_problem_and_scores(*,
                                max_p: int = 1000,
                                max_n: int = 1000,
                                zeros: list = None,
                                add_complements: bool = None,
                                score_subset: list = None,
                                rounding_decimals: int = None,
                                random_state=None) -> (dict, dict):
    """
    Generates a random problem and random but feasible scores

    Args:
        max_p (int): the maximum number of positives
        max_n (int): the maximum number of negatives
        zeros (None|list): the list of items to set to zero
        add_complements (bool): whether to add the complements
        score_subset (list|None): the subset of scores to compute
        rounding_decimals (int): the number of decimals to round to
        random_state (int, optional): the random seed to use

    Returns:
        dict, dict: the problem and the scores
    """
    evaluation, problem = generate_1_problem(max_p=max_p,
                                                max_n=max_n,
                                                zeros=zeros,
                                                add_complements=add_complements,
                                                random_state=random_state)
    evaluation['beta_negative'] = 2
    evaluation['beta_positive'] = 2
    scores = calculate_scores(evaluation, rounding_decimals=rounding_decimals)
    if score_subset is not None:
        scores = {key: value for key, value in scores.items() if key in score_subset}
    return problem, scores

def generate_1_problem(*,
                        max_p: int = 1000,
                        max_n: int = 1000,
                        zeros: list = None,
                        add_complements: bool = False,
                        random_state=None) -> dict:
    """
    Generates a random problem

    Args:
        max_p (int): the maximum number of positives
        max_n (int): the maximum number of negatives
        zeros (None|list): the list of items to set to zero
        add_complements (bool): whether to add the complements
        random_state (int, optional): the random seed to use

    Returns:
        dict: the problem
    """
    if random_state is None or isinstance(random_state, int):
        random_state = np.random.RandomState(random_state)

    if zeros is None:
        zeros = []

    p = int(random_state.randint(2, max_p+1))
    n = int(random_state.randint(2, max_n+1))

    if 'tp' in zeros:
        tp = 0
    elif 'fn' in zeros:
        tp = p
    else:
        tp = int(random_state.randint(1, p))

    if 'tn' in zeros:
        tn = 0
    elif 'fp' in zeros:
        tn = n
    else:
        tn = int(random_state.randint(1, n))

    result = {'p': p, 'n': n, 'tp': tp, 'tn': tn}

    if add_complements:
        result['fn'] = p - tp
        result['fp'] = n - tn

    return result, {'p': result['p'], 'n': result['n']}

def generate_problems(*,
                        n_problems: int = 1,
                        max_p: int = 1000,
                        max_n: int = 1000,
                        zeros: list = None,
                        add_complements: bool = False,
                        random_state=None) -> (dict, dict):
    """
    Generates a random problem

    Args:
        max_p (int): the maximum number of positives
        max_n (int): the maximum number of negatives
        zeros (None|list): the list of items to set to zero
        add_complements (bool): whether to add the complements
        random_state (int, optional): the random seed to use

    Returns:
        dict,dict|list(dict),list(dict): the evaluation and the problem or
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

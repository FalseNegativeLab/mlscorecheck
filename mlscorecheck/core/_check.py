"""
This module implements the main check functionality
"""

import itertools

from ._solutions import (load_solutions)
from ._score_loaders import (score_functions,
                            score_functions_standardized, score_function_aliases,
                            score_function_complementers)
from ._interval import Interval
from ._safe_eval import safe_eval, safe_call
from ._logger import logger

__all__ = ['check',
            'check_2v1',
            'create_intervals',
            'create_problems_2',
            'evaluate_1_solution',
            'check_zero_division',
            'check_negative_base',
            'check_empty_interval',
            'check_intersection']

solutions = load_solutions()
supported_scores = {key[0] for key in solutions}.union({key[1] for key in solutions})
aliases = score_function_aliases()
complementers = score_function_complementers()
functions = score_functions()
functions_standardized = score_functions_standardized()

def create_intervals(scores, eps):
    if not isinstance(eps, dict):
        eps = {key: eps for key in scores}

    intervals = {key: Interval(val - eps[key], val + eps[key]) for key, val in scores.items()}

    aliased = {}
    for key, val in intervals.items():
        if key in aliases:
            aliased[aliases[key]] = val
        else:
            aliased[key] = val

    complemented = {}
    for key, val in aliased.items():
        if key in complementers:
            complemented[complementers[key]] = 1.0 - val
        else:
            complemented[key] = val

    return complemented

def create_problems_2(scores):
    bases = list(itertools.combinations(scores, 2))
    problems = []
    for base0, base1 in bases:
        problems.extend((base0, base1, score)
                        for score in scores
                        if base0 != score and base1 != score)

    return problems

def check_zero_division(result):
    if result.get('message', None) == 'zero division':
        return {'consistency': True,
                'explanation': 'zero division indicates an underdetermined system'}
    return None

def check_negative_base(result):
    if result.get('message', None) == 'negative base':
        return {'consistency': True,
                'explanation': 'negative base indicates a non-suitable formula'}
    return None

def check_empty_interval(interval, name):
    if interval.is_empty():
        return {'consistency': False,
                'explanation': f'the interval for {name} does not contain integers'}
    return None

def check_intersection(target, reconstructed):
    if target.intersection(reconstructed).is_empty():
        return {'consistency': False,
                'explanation': f'the target score interval ({target}) and '\
                                f'the reconstructed intervals ({reconstructed}) '\
                                'do not intersect',
                'target_interval_reconstructed': reconstructed}
    return {'consistency': True,
            'explanation': f'the target score interval ({target}) and '\
                                f'the reconstructed intervals ({reconstructed}) '\
                                'do not intersect',
            'target_interval_reconstructed': reconstructed}


def evaluate_1_solution(target_interval, result, p, n, score_function):

    if tmp := check_zero_division(result):
        return tmp

    if tmp := check_negative_base(result):
        return tmp

    tp = result['tp'].shrink_to_integers().intersection(Interval(0, p))
    tn = result['tn'].shrink_to_integers().intersection(Interval(0, n))

    if tmp := check_empty_interval(tp, 'tp'):
        return tmp

    if tmp := check_empty_interval(tn, 'tn'):
        return tmp

    score = safe_call(score_function, {**result, 'p': p, 'n': n})

    return check_intersection(target_interval, score)

def check_2v1(intervals,
                problem,
                p, n):
    """
    Check one particular problem
    """
    # extracting the problem
    score0, score1, target = problem

    logger.info(f'checking {score0} and {score1} against {target}')

    # querying the solution and the target function
    solution = solutions[tuple(sorted([score0, score1]))]
    score_function = functions_standardized[target]

    # evaluating the solution
    results = solution.evaluate({**intervals,
                                 **{'p': p, 'n': n}})

    output = []

    # iterating and evaluating all sub-solutions
    for result in results:
        res = {'score_0': score0,
                'score_0_interval': intervals[score0],
                'score_1': score1,
                'score_1_interval': intervals[score1],
                'target_score': target,
                'target_interval': intervals[target],
                'solution': result}

        evaluation = evaluate_1_solution(intervals[target], result, p, n, score_function)

        output.append({**res, **evaluation})

    # constructing the final output
    # there can be multiple solutions to a problem, if one of them is consistent,
    # the triplet is considered consistent
    return {'details': output,
            'consistency': any(tmp['consistency'] for tmp in output)}

def check(scores, p, n, eps):
    """
    The main check functionality

    Args:
        scores (dict): the scores to check
        p (int): the number of positives
        n (int): the number of negatives
        eps (float/dict): the numerical uncertainty of the scores
    """

    intervals = create_intervals(scores, eps)

    problems = create_problems_2(list(scores.keys()))

    results = [check_2v1(intervals, problem, p, n) for problem in problems]

    succeeded = []
    failed = []

    for result in results:
        if result['consistency']:
            succeeded.append(result)
        else:
            failed.append(result)

    return {
        'succeeded': succeeded,
        'failed': failed,
        'overall_consistency': not len(failed)
        }

"""
This module implements the main check functionality
"""

import itertools

import numpy as np

from ._solutions import (load_solutions)
from ..scores import (score_functions_without_complements,
                            score_functions_standardized_without_complements,
                            score_function_aliases,
                            score_function_complements,
                            score_specifications)
from ._interval import Interval, IntervalUnion, sqrt
from ..core import safe_eval, safe_call, logger

__all__ = ['check',
            'check_2v1',
            'create_intervals',
            'create_problems_2',
            'evaluate_1_solution',
            'check_zero_division',
            'check_negative_base',
            'check_empty_interval',
            'check_intersection']

score_descriptors = score_specifications
solutions = load_solutions()
supported_scores = {key[0] for key in solutions}.union({key[1] for key in solutions})
aliases = score_function_aliases
complementers = score_function_complements
functions = score_functions_without_complements
functions_standardized = score_functions_standardized_without_complements

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

    for key, val in complemented.items():
        if key in score_descriptors:
            lower_bound = score_descriptors[key].get('lower_bound', -np.inf)
            upper_bound = score_descriptors[key].get('upper_bound', np.inf)

            complemented[key] = complemented[key].intersection(Interval(lower_bound, upper_bound))

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
                'target_interval_reconstructed': reconstructed.to_tuple()}
    return {'consistency': True,
            'explanation': f'the target score interval ({target}) and '\
                                f'the reconstructed intervals ({reconstructed}) '\
                                'do intersect',
            'target_interval_reconstructed': reconstructed.to_tuple()}


def evaluate_1_solution(target_interval, result, p, n, score_function):

    if tmp := check_zero_division(result):
        return tmp

    if tmp := check_negative_base(result):
        return tmp

    if not isinstance(result['tp'], (Interval, IntervalUnion)):
        result['tp'] = Interval(result['tp'], result['tp'])
    if not isinstance(result['tn'], (Interval, IntervalUnion)):
        result['tn'] = Interval(result['tn'], result['tn'])

    tp = result['tp'].shrink_to_integers().intersection(Interval(0, p))
    tn = result['tn'].shrink_to_integers().intersection(Interval(0, n))

    if tmp := check_empty_interval(tp, 'tp'):
        return tmp

    if tmp := check_empty_interval(tn, 'tn'):
        return tmp

    score = safe_call(score_function, {**result, 'p': p, 'n': n, 'sqrt': sqrt})

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
                'score_0_interval': intervals[score0].to_tuple(),
                'score_1': score1,
                'score_1_interval': intervals[score1].to_tuple(),
                'target_score': target,
                'target_interval': intervals[target].to_tuple(),
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
    scores = {key: value for key, value in scores.items() if key in supported_scores}

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

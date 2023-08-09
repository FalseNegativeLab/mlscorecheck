"""
This module implements the main check functionality
"""

import itertools

from ._solutions import (load_solutions)
from ._score_loaders import (score_functions,
                            score_functions_standardized, score_function_aliases,
                            score_function_complementers)
from ._interval import Interval
from ._logger import logger

__all__ = ['check',
            'check_2v1',
            'create_intervals',
            'create_problems_2']

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
        for score in scores:
            if base0 != score and base1 != score:
                problems.append((base0, base1, score))

    return problems

def check_2v1(intervals,
                problem,
                p, n):
    """
    Check one particular problem
    """
    score0, score1, target = problem

    logger.info(f'checking {score0} and {score1} against {target}')

    solution = solutions[tuple(sorted([score0, score1]))]
    results = solution.evaluate({**intervals, **{'p': p, 'n': n}})

    score_function = functions_standardized[target]

    args = list(score_function.__code__.co_varnames[:score_function.__code__.co_kwonlyargcount])

    output = []

    for result in results:
        res = {'base_score_0': score0,
                'base_score_0_interval': intervals[score0],
                'base_score_1': score1,
                'base_score_1_interval': intervals[score1],
                'target_score': target,
                'target_interval': intervals[target]}

        if result['results'].get('message', None) is not None:
            if result['results']['message'] == 'zero division error':
                res['solution'] = result
                res['explanation'] = 'zero division indicates an underdetermined system'
                res['consistency'] = True
                output.append(res)
                continue
            elif result['results']['message'] == 'negative base':
                pass
                # TODO

        tp = result['results']['tp']
        tn = result['results']['tn']

        tp = tp.shrink_to_integers()
        tn = tn.shrink_to_integers()

        params = {**result['results']}

        params['p'] = p
        params['n'] = n

        score = score_function(**{arg: params[arg] for arg in args})

        consistency = True
        if score.intersection(intervals[target]).is_empty():
            consistency = False

        res['solution'] = result
        res['consistency'] = consistency
        res['target_interval_reconstructed'] = (score.lower_bound, score.upper_bound)
        res['explanation'] = 'the intervals do intersect'

        output.append(res)

    final = {}
    final['details'] = output
    final['consistency'] = any(tmp['consistency'] for tmp in output)
    return final


def check(scores, p, n, eps):
    """
    The main check functinality

    Args:
        scores (dict): the scores to check
        p (int): the number of positives
        n (int): the number of negatives
        eps (float/dict): the numerical uncertainty of the scores
    """

    intervals = create_intervals(scores, eps)

    problems = create_problems_2(list(scores.keys()))

    print(intervals)

    results = [check_2v1(intervals, problem, p, n) for problem in problems]

    succeeded = []
    failed = []

    for result in results:
        if result['consistency']:
            succeeded.append(result)
        else:
            failed.append(result)

    return {'succeeded': succeeded,
                'failed': failed,
                'overall_consistency': len(failed) == 0}

"""
This module implements the main check functionality for individual scores
"""

import itertools

import numpy as np

from ._solutions import solution_specifications
from ..scores import (score_functions_without_complements,
                            score_functions_standardized_without_complements,
                            score_function_aliases,
                            score_function_complements,
                            score_specifications,
                            score_functions_standardized_all)
from ._interval import Interval, IntervalUnion, sqrt
from ..core import safe_call, logger, check_uncertainty_and_tolerance, NUMERICAL_TOLERANCE

__all__ = ['check_individual_scores',
            'check_2v1',
            'create_intervals',
            'create_problems_2',
            'evaluate_1_solution',
            'check_zero_division',
            'check_negative_base',
            'check_empty_interval',
            'check_intersection',
            'determine_edge_cases',
            'resolve_aliases_and_complements']

solutions = solution_specifications
score_descriptors = score_specifications
supported_scores = {key[0] for key in solutions}.union({key[1] for key in solutions})
aliases = score_function_aliases
complementers = score_function_complements
functions = score_functions_without_complements
functions_standardized = score_functions_standardized_without_complements

def resolve_aliases_and_complements(scores):
    """
    Standardizing the scores by resolving aliases and complements

    Args:
        scores (dict(str,float)): the dictionary of scores

    Returns:
        dict(str,float): the resolved scores
    """
    aliased = {}
    for key, val in scores.items():
        if key in score_function_aliases:
            aliased[score_function_aliases[key]] = val
        else:
            aliased[key] = val

    complemented = {}
    for key, val in aliased.items():
        if key in score_function_complements:
            complemented[score_function_complements[key]] = 1.0 - val
        else:
            complemented[key] = val

    return complemented

def determine_edge_cases(score, p, n):
    """
    Determining the edge cases of a score

    Args:
        scores (dict(str,float)): the dictionary of scores
        p (int): the number of positives
        n (int): the number of negatives

    Returns:
        list(float): the list of edge case values the score can take
    """
    edge_cases = set()

    tp_cases = [{'tp': 0, 'fn': p}, {'tp': p, 'fn': 0}]
    tn_cases = [{'tn': 0, 'fp': n}, {'tn': n, 'fp': 0}]

    nans = score_specifications[score].get('nans')

    for arg0 in tp_cases:
        for arg1 in tn_cases:
            params = {**arg0, **arg1, 'p': p, 'n': n}
            edge_cases.add(safe_call(score_functions_standardized_all[score],
                                        {**params, 'sqrt': sqrt},
                                        nans))

    return list(edge_cases)

def create_intervals(scores, eps, numerical_tolerance=NUMERICAL_TOLERANCE):
    """
    Turns the scores into intervals using the uncertainty specifications,
    the interval for a score will be (score - eps, score + eps).
    The score set is also standardized by replacing the aliases and replacing
    complements with the corresponding scores from the base set.

    Args:
        scores (dict): the scores to be turned into intervals
        eps (float|dict): the numerical uncertainty
        numerical_tolerance (float): in practice, beyond the numerical uncertainty of
                                    the scores, some further tolerance is applied. This is
                                    orders of magnitude smaller than the uncertainty of the
                                    scores. It does ensure that the specificity of the test
                                    is 1, it might slightly decrease the sensitivity.

    Returns:
        dict: the score intervals
    """

    # turning the uncertainty into a score specific dictionary if it isnt that
    if not isinstance(eps, dict):
        eps = {key: eps for key in scores}

    # creating the intervals
    intervals = {key: Interval(val - eps[key] - numerical_tolerance,
                                val + eps[key] + numerical_tolerance)
                    for key, val in scores.items()}

    # trimming the intervals into the domains of the scores
    # for example, to prevent acc - eps < 0 implying a negative subinterval for
    # accuracy
    for key in intervals:
        if key in score_descriptors:
            lower_bound = score_descriptors[key].get('lower_bound', -np.inf)
            upper_bound = score_descriptors[key].get('upper_bound', np.inf)

            intervals[key] = intervals[key].intersection(Interval(lower_bound, upper_bound))

    return intervals

def create_problems_2(scores):
    """
    Given a set of scores, this function generates all test case specifications.
    A test case specification consists of two base scores and a third score they
    are checked against.

    Args:
        list(str): the list of scores specified

    Returns:
        list(tuple): all possible triplets of the form (base0, base1, target)
    """
    bases = list(itertools.combinations(scores, 2))
    problems = []
    for base0, base1 in bases:
        problems.extend((base0, base1, score)
                        for score in scores
                        if score not in {base0, base1})

    return problems

def check_zero_division(result):
    """
    Check if zero division occured in a particular case

    Args:
        result (dict): the dictionary of results

    Returns:
        None|dict: None if there is no zero division, some explanation otherwise
    """
    if result.get('message', None) == 'zero division':
        return {'inconsistency': False,
                'explanation': 'zero division indicates an underdetermined system'}
    return None

def check_negative_base(result):
    """
    Check if negative base occured in a particular case

    Args:
        result (dict): the dictionary of results

    Returns:
        None|dict: None if there is no negative base, some explanation otherwise
    """
    if result.get('message', None) == 'negative base':
        return {'inconsistency': True,
                'explanation': 'negative base indicates a non-suitable formula'}
    return None

def check_empty_interval(interval, name):
    """
    Check if the interval is empty

    Args:
        interval (Interval|IntervalUnion): the interval
        name (str): name of the variable the interval is determined for

    Returns:
        None|dict: None if the interval is not empty, some explanation otherwise
    """
    if interval.is_empty():
        return {'inconsistency': True,
                'explanation': f'the interval for {name} does not contain integers'}
    return None

def check_intersection(target, reconstructed):
    """
    Checks the intersection of the target score and the reconstructed interval

    Args:
        target (Interval|IntervalUnion): the interval
        reconstructed (Interval|IntervalUnion): the reconstructed interval

    Returns:
        dict: a dictionary containing the consistency decision and the explanation
    """

    if target.intersection(reconstructed).is_empty():
        return {'inconsistency': True,
                'explanation': f'the target score interval ({target}) and '\
                                f'the reconstructed intervals ({reconstructed}) '\
                                'do not intersect',
                'target_interval_reconstructed': reconstructed.to_tuple()}
    return {'inconsistency': False,
            'explanation': f'the target score interval ({target}) and '\
                                f'the reconstructed intervals ({reconstructed}) '\
                                'do intersect',
            'target_interval_reconstructed': reconstructed.to_tuple()}

def evaluate_1_solution(target_interval, result, p, n, score_function):
    """
    Carry out the evaluation for 1 particular solution

    Args:
        target_interval (Interval|IntervalUnion): the interval of the target score
        result (dict): the result of the evaluation
        p (int): the number of positives
        n (int): the number of negatives
        score_function (callable): the score function to be called

    Returns:
        dict: the dictionary of the result of the evaluation
    """

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

def check_2v1(scores,
                eps,
                problem,
                p,
                n,
                *,
                numerical_tolerance=NUMERICAL_TOLERANCE):
    """
    Check one particular problem

    Args:
        scores (dict(str,float)): the scores
        eps (float|dict(str,float)): the numerical uncertainty(ies)
        problem (tuple(str,str,str)): the problem specification in the form
                                        (base_score0, base_score1, target_score)
        p (int): the number of positives
        n (int): the number of negatives
        numerical_tolerance (float): in practice, beyond the numerical uncertainty of
                                    the scores, some further tolerance is applied. This is
                                    orders of magnitude smaller than the uncertainty of the
                                    scores. It does ensure that the specificity of the test
                                    is 1, it might slightly decrease the sensitivity.

    Returns:
        dict: the result of the evaluation
    """
    # extracting the problem
    score0, score1, target = problem

    logger.info('checking %s and %s against %s', score0, score1, target)

    intervals = create_intervals({key: scores[key] for key in scores
                                    if key in problem},
                                    eps,
                                    numerical_tolerance=numerical_tolerance)

    # evaluating the solution
    results = solutions[tuple(sorted([score0, score1]))].evaluate({**intervals,
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

        evaluation = evaluate_1_solution(intervals[target],
                                            result,
                                            p,
                                            n,
                                            functions_standardized[target])

        output.append({**res, **evaluation})

    # constructing the final output
    # there can be multiple solutions to a problem, if one of them is consistent,
    # the triplet is considered consistent
    return {'details': output,
            'edge_scores': list({key for key in [score0, score1]
                            if scores[key] in determine_edge_cases(key, p, n)}),
            'underdetermined': all(tmp.get('message') == 'zero division'
                                    for tmp in output),
            'inconsistency': all(tmp['inconsistency'] for tmp in output)}

def check_individual_scores(scores, p, n, eps, numerical_tolerance=1e-6):
    """
    The main check functionality

    Args:
        scores (dict): the scores to check
        p (int): the number of positives
        n (int): the number of negatives
        eps (float|dict): the numerical uncertainty of the scores
        numerical_tolerance (float): in practice, beyond the numerical uncertainty of
                                    the scores, some further tolerance is applied. This is
                                    orders of magnitude smaller than the uncertainty of the
                                    scores. It does ensure that the specificity of the test
                                    is 1, it might slightly decrease the sensitivity.

    Returns:
        dict: the result of the check. The 'consistency' flag contains the
        overall decision, the 'succeeded' and 'failed' lists contain
        the details of the individual tests
    """
    check_uncertainty_and_tolerance(eps, numerical_tolerance)

    scores = resolve_aliases_and_complements(scores)

    scores = {key: value for key, value in scores.items() if key in supported_scores}

    problems = create_problems_2(list(scores.keys()))

    results = [check_2v1(scores, eps, problem, p, n,
                            numerical_tolerance=numerical_tolerance)
                for problem in problems]

    succeeded = []
    failed = []
    edge_scores = set()

    for result in results:
        edge_scores = edge_scores.union(set(result['edge_scores']))
        if result['inconsistency']:
            failed.append(result)
        else:
            succeeded.append(result)

    return {
        'tests_succeeded': succeeded,
        'tests_failed': failed,
        'underdetermined': len(failed) == 0 and all(tmp['underdetermined'] for tmp in succeeded),
        'edge_scores': list(edge_scores),
        'inconsistency': len(failed) > 0
        }

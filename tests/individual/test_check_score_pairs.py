"""
Testing the check functionality
"""

import pytest

from mlscorecheck.core import safe_call, NUMERICAL_TOLERANCE
from mlscorecheck.individual import (check_2v1,
                                create_intervals,
                                create_problems_2,
                                load_solutions,
                                check_negative_base,
                                evaluate_1_solution,
                                Interval,
                                sqrt,
                                generate_1_problem,
                                determine_edge_cases,
                                resolve_aliases_and_complements)
from mlscorecheck.scores import (score_functions_with_solutions,
                                    score_specifications)

scores = score_specifications
functions = score_functions_with_solutions
solutions = load_solutions()

random_seeds = [5]

def test_resolve_aliases_and_complements():
    """
    Resolve the score aliases and complemnets
    """

    assert resolve_aliases_and_complements({'err': 0.4, 'f1': 0.8, 'sens': 0.3})\
            == {'acc': 0.6, 'f1p': 0.8, 'sens': 0.3}

def test_check_negative_base():
    """
    Testing the check_negative_base function
    """
    tmp = check_negative_base({'message': 'negative base'})
    assert tmp['inconsistency']
    assert check_negative_base({'results': {}}) is None

def test_evaluate_1_solution_negative_base():
    """
    Testing the evaluate_1_solution function for negative base
    """
    tmp = evaluate_1_solution(None, {'message': 'negative base'}, None, None, None)
    assert tmp['explanation'].startswith('negative base')

def test_evaluate_1_scalar():
    """
    Testing the evaluate_1_solution with scalar tp and tn
    """
    tmp = evaluate_1_solution(Interval(0, 1),
                                {'tp': 5, 'tn': 10, 'fn': 5, 'fp': 10},
                                p=10,
                                n=20,
                                score_function=functions['acc'])
    assert not tmp['inconsistency']

def test_create_intervals():
    """
    Testing the create intervals function
    """
    scores_ = resolve_aliases_and_complements({'acc': 0.6, 'tpr': 0.4, 'fnr': 0.6})
    intervals = create_intervals(scores_, eps=1e-4)
    assert abs(intervals['acc'].lower_bound - 0.5999) <= 1.1 * NUMERICAL_TOLERANCE
    assert abs(intervals['acc'].upper_bound - 0.6001) <= 1.1 * NUMERICAL_TOLERANCE
    assert abs(intervals['sens'].lower_bound - 0.3999) <= 1.1 * NUMERICAL_TOLERANCE
    assert abs(intervals['sens'].upper_bound - 0.4001) <= 1.1 * NUMERICAL_TOLERANCE

    scores_ = resolve_aliases_and_complements({'fnr': 0.6})
    intervals = create_intervals(scores_, eps=1e-4)
    assert abs(intervals['sens'].lower_bound - 0.3999) <= 1.1 * NUMERICAL_TOLERANCE
    assert abs(intervals['sens'].upper_bound - 0.4001) <= 1.1 * NUMERICAL_TOLERANCE

    intervals = create_intervals({'kappa': 0.5}, eps=1e-4)
    assert abs(intervals['kappa'].lower_bound - 0.4999) <= 1.1 * NUMERICAL_TOLERANCE
    assert abs(intervals['kappa'].upper_bound - 0.5001) <= 1.1 * NUMERICAL_TOLERANCE

def test_create_problems_2():
    """
    Testing the create problems function
    """

    problems = create_problems_2(['acc', 'sens', 'spec', 'ppv'])

    assert len(problems) == 6*2

@pytest.mark.parametrize("problem", create_problems_2(list(functions.keys())))
@pytest.mark.parametrize("random_state", random_seeds)
@pytest.mark.parametrize("zeros", [[], ['tp'], ['tn'], ['fp'], ['fn'],
                                    ['tp', 'tn'], ['tp', 'fp'], ['fp', 'fn'], ['tn', 'fn']])
def test_check_2v1(problem, random_state, zeros):
    """
    Testing the check_2v1 function
    """

    # generating a random setup
    evaluation, _ = generate_1_problem(random_state=random_state,
                                        add_complements=True,
                                        zeros=zeros)
    evaluation['sqrt'] = sqrt
    evaluation['beta_plus'] = 2
    evaluation['beta_minus'] = 2

    # extracting the problem
    base0 = problem[0]
    base1 = problem[1]
    target = problem[2]

    if tuple(sorted([base0, base1])) not in solutions:
        return

    # determining the scores
    scores_dict = {base0: safe_call(functions[base0], evaluation, scores[base0].get('nans')),
                base1: safe_call(functions[base1], evaluation, scores[base1].get('nans')),
                target: safe_call(functions[target], evaluation, scores[target].get('nans'))}

    scores_dict['beta_plus'] = 2
    scores_dict['beta_minus'] = 2

    print('SCORES', scores_dict)

    # if all scores are Nones, skip the test
    if scores_dict[base0] is None or scores_dict[base1] is None or scores_dict[target] is None:
        return

    # checking the problem
    result = check_2v1(scores_dict, 1e-8, problem, evaluation['p'], evaluation['n'])

    print(result)

    assert result['underdetermined'] or not result['inconsistency']

@pytest.mark.parametrize("problem", create_problems_2(list(functions.keys())))
@pytest.mark.parametrize("zeros", [[], ['tp'], ['tn'], ['fp'], ['fn'],
                                    ['tp', 'tn'], ['tp', 'fp'], ['fp', 'fn'], ['tn', 'fn']])
@pytest.mark.parametrize("random_state", random_seeds)
def test_check_false_2v1(problem, zeros, random_state):
    """
    Testing the check_2v1 function
    """

    # generating a random configuration
    evaluation, _ = generate_1_problem(random_state=random_state,
                                        add_complements=True,
                                        zeros=zeros)
    evaluation['sqrt'] = sqrt
    evaluation['beta_plus'] = 2
    evaluation['beta_minus'] = 2

    # extracting the problem
    base0 = problem[0]
    base1 = problem[1]
    target = problem[2]

    if tuple(sorted([base0, base1])) not in solutions:
        return

    # calculating the scores
    scores_dict = {base0: safe_call(functions[base0], evaluation, scores[base0].get('nans')),
                base1: safe_call(functions[base1], evaluation, scores[base1].get('nans')),
                target: safe_call(functions[target], evaluation, scores[target].get('nans'))}

    # skipping the test if all scores are NaN (indicated by None due to the safe call)
    if scores_dict[base0] is None or scores_dict[base1] is None or scores_dict[target] is None:
        return

    scores_dict['beta_plus'] = 2
    scores_dict['beta_minus'] = 2

    # determining the edge cases
    edge_original = (scores_dict[base0] in determine_edge_cases(base0,
                                                                evaluation['p'],
                                                                evaluation['n'],
                                                                2,
                                                                2)\
                and scores_dict[base1] in determine_edge_cases(base1,
                                                                evaluation['p'],
                                                                evaluation['n'],
                                                                2,
                                                                2))

    # generating a completely random new configuration
    evaluation, _ = generate_1_problem(random_state=(random_state + 1), add_complements=True)

    result = check_2v1(scores_dict, 1e-8, problem, evaluation['p'], evaluation['n'],
                        numerical_tolerance=5*1e-10)

    # checking if the new setup has edge case scores
    edges_new = len(result['edge_scores']) > 0

    assert (edge_original
            or edges_new
            or result['underdetermined']
            or result['inconsistency'])


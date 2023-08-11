"""
Testing the check functionality
"""

import pytest

from mlscorecheck.core import (check,
                                check_2v1,
                                create_intervals,
                                create_problems_2,
                                load_scores,
                                load_solutions,
                                score_functions,
                                score_functions_with_solutions,
                                evaluate_1_solution,
                                check_zero_division,
                                check_negative_base,
                                check_empty_interval,
                                check_intersection)
from mlscorecheck.utils import (generate_problem,
                                generate_problem_tp0)

scores = load_scores()
functions = score_functions_with_solutions()
solutions = load_solutions()

def test_check_negative_base():
    """
    Testing the check_negative_base function
    """
    tmp = check_negative_base({'message': 'negative base'})
    assert tmp['consistency'] == True
    assert check_negative_base({'results': {}}) is None

def test_evaluate_1_solution_negative_base():
    """
    Testing the evaluate_1_solution function for negative base
    """
    tmp = evaluate_1_solution(None, {'message': 'negative base'}, None, None, None)
    assert tmp['explanation'].startswith('negative base')

def test_create_intervals():
    """
    Testing the create intervals function
    """
    intervals = create_intervals({'acc': 0.6, 'tpr': 0.4, 'fnr': 0.6}, eps=1e-4)
    assert abs(intervals['acc'].lower_bound - 0.5999) < 1e-8
    assert abs(intervals['acc'].upper_bound - 0.6001) < 1e-8
    assert abs(intervals['sens'].lower_bound - 0.3999) < 1e-8
    assert abs(intervals['sens'].upper_bound - 0.4001) < 1e-8

    intervals = create_intervals({'fnr': 0.6}, eps=1e-4)
    assert abs(intervals['sens'].lower_bound - 0.3999) < 1e-8
    assert abs(intervals['sens'].upper_bound - 0.4001) < 1e-8

def test_create_problems_2():
    """
    Testing the create problems function
    """

    problems = create_problems_2(['acc', 'sens', 'spec', 'ppv'])

    assert len(problems) == 6*2

@pytest.mark.parametrize("problem", create_problems_2(list(functions.keys())))
def test_check_2v1(problem):
    """
    Testing the check_2v1 function
    """

    prob = generate_problem(random_seed=5)

    base0 = problem[0]
    base1 = problem[1]
    target = problem[2]

    scores_dict = {base0: functions[base0](**{arg: prob[arg] for arg in scores[base0]['args']}),
                base1: functions[base1](**{arg: prob[arg] for arg in scores[base1]['args']}),
                target: functions[target](**{arg: prob[arg] for arg in scores[target]['args']})}

    intervals = create_intervals(scores_dict, eps=1e-4)

    result = check_2v1(intervals, problem, prob['p'], prob['n'])

    assert result['consistency']

@pytest.mark.parametrize("problem", create_problems_2(list(functions.keys())))
def test_check_false_2v1(problem):
    """
    Testing the check_2v1 function
    """

    prob = generate_problem(random_seed=5)

    base0 = problem[0]
    base1 = problem[1]
    target = problem[2]

    scores_dict = {base0: functions[base0](**{arg: prob[arg] for arg in scores[base0]['args']}),
                base1: functions[base1](**{arg: prob[arg] for arg in scores[base1]['args']}),
                target: functions[target](**{arg: prob[arg] for arg in scores[target]['args']})}

    scores_dict[target] = scores_dict[target] * 2

    intervals = create_intervals(scores_dict, eps=1e-4)

    result = check_2v1(intervals, problem, prob['p'], prob['n'])

    print(result)

    assert not result['consistency']

def test_check():
    """
    Testing the check function
    """

    problem = generate_problem(random_seed=5)

    score_values = {score: functions[score](**{arg: problem[arg]
                                                for arg in scores[score]['args']})
                                                    for score in functions}

    result = check(score_values, problem['p'], problem['n'], eps=1e-4)

    assert result['overall_consistency']

def test_check_fail():
    """
    Testing the check function for failure
    """

    problem = generate_problem(random_seed=5)

    score_values = {score: functions[score](**{arg: problem[arg]
                                                for arg in scores[score]['args']})
                                                    for score in functions}

    result = check(score_values, problem['p']*2, problem['n']+50, eps=1e-4)

    assert not result['overall_consistency']

def test_check_tp0():
    """
    Testing the check function with tp=0
    """

    problem = generate_problem_tp0(random_seed=5)

    score_values = {score: functions[score](**{arg: problem[arg]
                                                for arg in scores[score]['args']})
                                                    for score in functions}

    result = check(score_values, problem['p'], problem['n'], eps=1e-4)

    assert result['overall_consistency']

def test_check_fail_tp0():
    """
    Testing the check function for failure with tp=0
    """

    problem = generate_problem_tp0(random_seed=5)

    score_values = {score: functions[score](**{arg: problem[arg]
                                                for arg in scores[score]['args']})
                                                    for score in functions}

    result = check(score_values, problem['p']*2, problem['n']+50, eps=1e-4)

    assert not result['overall_consistency']


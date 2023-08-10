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
                                score_functions_with_solutions)
from mlscorecheck.utils import (generate_problem,
                                generate_problem_tp0)

scores = load_scores()
functions = score_functions_with_solutions()
solutions = load_solutions()

def test_create_intervals():
    """
    Testing the create intervals function
    """
    intervals = create_intervals({'acc': 0.5}, eps=1e-4)
    assert intervals['acc'].lower_bound == 0.4999
    assert intervals['acc'].upper_bound == 0.5001

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

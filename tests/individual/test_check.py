"""
Testing the check functionality
"""

import pytest

from mlscorecheck.individual import (check,
                                check_2v1,
                                create_intervals,
                                create_problems_2,
                                load_solutions,
                                evaluate_1_solution,
                                check_zero_division,
                                check_negative_base,
                                check_empty_interval,
                                check_intersection,
                                Interval,
                                generate_problems)
from mlscorecheck.scores import (score_functions_with_solutions,
                                    score_specifications)

scores = score_specifications
functions = score_functions_with_solutions
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

def test_evaluate_1_scalar():
    """
    Testing the evaluate_1_solution with scalar tp and tn
    """
    tmp = evaluate_1_solution(Interval(0, 1), {'tp': 5, 'tn': 10, 'fn': 5, 'fp': 10}, p=10, n=20, score_function=functions['acc'])
    assert tmp['consistency']

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

    intervals = create_intervals({'kappa': 0.5}, eps=1e-4)
    assert abs(intervals['kappa'].lower_bound - 0.4999) < 1e-8
    assert abs(intervals['kappa'].upper_bound - 0.5001) < 1e-8

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

    evaluation, prob = generate_problems(n_problems=1, random_state=5, add_complements=True)

    base0 = problem[0]
    base1 = problem[1]
    target = problem[2]

    scores_dict = {base0: functions[base0](**{arg: evaluation[arg] for arg in scores[base0]['args']}),
                base1: functions[base1](**{arg: evaluation[arg] for arg in scores[base1]['args']}),
                target: functions[target](**{arg: evaluation[arg] for arg in scores[target]['args']})}

    intervals = create_intervals(scores_dict, eps=1e-4)

    result = check_2v1(intervals, problem, evaluation['p'], evaluation['n'])

    assert result['consistency']

@pytest.mark.parametrize("problem", create_problems_2(list(functions.keys())))
def test_check_false_2v1(problem):
    """
    Testing the check_2v1 function
    """

    evaluation, prob = generate_problems(n_problems=1, random_state=5, add_complements=True)

    base0 = problem[0]
    base1 = problem[1]
    target = problem[2]

    scores_dict = {base0: functions[base0](**{arg: evaluation[arg] for arg in scores[base0]['args']}),
                base1: functions[base1](**{arg: evaluation[arg] for arg in scores[base1]['args']}),
                target: functions[target](**{arg: evaluation[arg] for arg in scores[target]['args']})}

    scores_dict[target] = scores_dict[target] * 2

    intervals = create_intervals(scores_dict, eps=1e-4)

    result = check_2v1(intervals, problem, evaluation['p'], evaluation['n'])

    print(result)

    underdetermined = all([res['solution'].get('message') == 'zero division' for res in result['details']])

    assert underdetermined or not result['consistency']

@pytest.mark.parametrize("zeros", [[], ['tp'], ['tn'], ['fp'], ['fn'], ['tp', 'tn'], ['fp', 'fn'], ['tp', 'fp'], ['tn', 'fn']])
def test_check(zeros):
    """
    Testing the check function
    """

    evaluation, problem = generate_problems(n_problems=1, random_state=5, zeros=zeros, add_complements=True)

    score_values = {}

    for score in functions:
        nans = scores[score].get('nans')
        if nans is not None:
            flag2 = False
            for item in nans:
                flag = True
                for key in item:
                    flag = flag and item[key] == evaluation[key]
                if flag:
                    print('aaa', flag, item, score)
                    flag2 = True
                    break
            if flag2:
                continue
        score_values[score] = functions[score](**{arg: evaluation[arg]
                                            for arg in scores[score]['args']})

    result = check(score_values, evaluation['p'], evaluation['n'], eps=1e-4)

    print(problem)
    print(score_values)
    print(result['failed'])

    tmp = []
    for res in result['succeeded']:
        for tmp2 in res['details']:
            tmp.append(tmp2['solution'].get('message') == 'zero division' or tmp2['solution'].get('message') is None)

    underdetermined = all(tmp) and len(result['failed']) == 0

    assert underdetermined or result['overall_consistency']

@pytest.mark.parametrize("zeros", [[], ['tp'], ['tn'], ['fp'], ['fn'], ['tp', 'fp'], ['tn', 'fn'], ['tp', 'tn'], ['fp', 'fn']])
def test_check_failure(zeros):
    """
    Testing the failure
    """

    evaluation, problem = generate_problems(n_problems=1, random_state=5, zeros=zeros, add_complements=True)

    score_values = {}

    for score in functions:
        nans = scores[score].get('nans')
        if nans is not None:
            flag2 = False
            for item in nans:
                flag = True
                for key in item:
                    flag = flag and item[key] == evaluation[key]
                if flag:
                    print('aaa', flag, item, score)
                    flag2 = True
                    break
            if flag2:
                continue
        score_values[score] = functions[score](**{arg: evaluation[arg]
                                            for arg in scores[score]['args']})

    result = check(score_values, evaluation['p']*2, evaluation['n']+50, eps=1e-4)

    print(score_values)
    print(problem)
    print(result['failed'])

    tmp = []
    for res in result['succeeded']:
        for tmp2 in res['details']:
            tmp.append(tmp2['solution'].get('message') == 'zero division' or tmp2['solution'].get('message') is None)

    underdetermined = all(tmp) and len(result['failed']) == 0

    assert underdetermined or not result['overall_consistency']

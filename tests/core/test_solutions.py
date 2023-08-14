"""
This module tests the solutions
"""

import pytest

from mlscorecheck.core import (load_solutions,
                                load_scores,
                                score_functions_without_complements,
                                score_functions_standardized_without_complements,
                                Solution,
                                Solutions,
                                Interval,
                                IntervalUnion)

from mlscorecheck.utils import (generate_problem)

solutions = load_solutions()
scores = load_scores()
functions = score_functions_without_complements
functions_standardized = score_functions_standardized_without_complements

@pytest.mark.parametrize("sol", list(solutions.keys()))
@pytest.mark.parametrize("zeros", [[], ['tp'], ['tn'], ['fp'], ['fn'], ['tp', 'fp'], ['tn', 'fn']])
def test_solution(sol, zeros):
    """
    Testing a particular solution
    """

    problem = generate_problem(zeros=zeros)
    problem['beta_plus'] = 2
    problem['beta_minus'] = 2

    nans0 = scores[sol[0]].get('nans')
    nans1 = scores[sol[1]].get('nans')

    if nans0 is not None:
        for item in nans0:
            flag = True
            for key in item:
                flag = flag and item[key] == problem[key]
            if flag:
                return
    if nans1 is not None:
        for item in nans1:
            flag = True
            for key in item:
                flag = flag and item[key] == problem[key]
            if flag:
                return

    print(sol, problem)

    try:
        score0 = functions[sol[0]](**{key: value for key, value in problem.items() if key in scores[sol[0]]['args']})
        score1 = functions[sol[1]](**{key: value for key, value in problem.items() if key in scores[sol[1]]['args']})
    except:
        return

    result = solutions[sol].evaluate({**problem,
                                      **{sol[0]: score0,
                                            sol[1]: score1}})

    print(score0, score1)

    flags = []

    for res in result:
        print(res)

        if res.get('message') is not None:
            continue
        tp = res['tp']
        tn = res['tn']

        flags.append(abs(tp - problem['tp']) < 1e-5 and abs(tn - problem['tn']) < 1e-5)

    assert len(flags) == 0 or any(flags)

@pytest.mark.parametrize("sol", list(solutions.keys()))
@pytest.mark.parametrize("zeros", [[], ['tp'], ['tn'], ['fp'], ['fn'], ['tp', 'fp'], ['tn', 'fn']])
def test_solution_failure(sol, zeros):
    """
    Testing a particular solution
    """

    problem = generate_problem(zeros=zeros)
    problem['beta_plus'] = 2
    problem['beta_minus'] = 2

    nans0 = scores[sol[0]].get('nans')
    nans1 = scores[sol[1]].get('nans')

    if nans0 is not None:
        for item in nans0:
            flag = True
            for key in item:
                flag = flag and item[key] == problem[key]
            if flag:
                return
    if nans1 is not None:
        for item in nans1:
            flag = True
            for key in item:
                flag = flag and item[key] == problem[key]
            if flag:
                return

    try:
        score0 = functions[sol[0]](**{key: value for key, value in problem.items() if key in scores[sol[0]]['args']})
        score1 = functions[sol[1]](**{key: value for key, value in problem.items() if key in scores[sol[1]]['args']})
    except:
        return

    problem['tp'] = problem['tp'] * 10 + 5

    result = solutions[sol].evaluate({**problem,
                                      **{sol[0]: score0,
                                            sol[1]: score1}})

    flags = []

    for res in result:
        if res.get('message') is not None:
            continue
        tp = res['tp']
        tn = res['tn']

        flags.append(abs(tp - problem['tp']) < 1e-5 and abs(tn - problem['tn']) < 1e-5)

    assert len(flags) == 0 or not any(flags)

def test_solution_object():
    """
    Testing the solution abstraction
    """

    sol = Solution(solution={'tp': {'expression': 'p*sens', 'symbols': ['p', 'sens']}},
                    non_zero=[{'expression': 'p',
                                'symbols': ['p']}],
                    non_negative=[{'expression': 'p',
                                'symbols': ['p']}])

    assert len(sol.to_dict()) == 3

    res = sol.evaluate({'p': 3, 'sens': 1})

    assert len(res) > 0

def test_solution_non_negatives():
    """
    Testing the non-negative base detection in the solution
    """

    sol = Solution({'tp': {'expression': 'p', 'symbols': ['p']}},
                    non_zero=[],
                    non_negative=[{'expression': 'p', 'symbols': ['p']}])

    res = sol.evaluate({'p': Interval(-2, -1)})

    assert res['message'] == 'negative base'

    res = sol.evaluate({'p': IntervalUnion([Interval(-2, -1)])})

    assert res['message'] == 'negative base'

    res = sol.evaluate({'p': -1})

    assert res['message'] == 'negative base'

def test_solutions_to_dict():
    """
    Testing the to_dict functionality of Solutions
    """

    sol = solutions[list(solutions.keys())[0]]
    assert len(sol.to_dict()) == 2

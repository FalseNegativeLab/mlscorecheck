"""
This module tests the solutions
"""

import pytest

from mlscorecheck.core import (load_solutions,
                                load_scores,
                                score_functions,
                                score_functions_standardized,
                                Solution,
                                Solutions,
                                Interval,
                                IntervalUnion)

from mlscorecheck.utils import (generate_problem,
                                generate_problem_tp0)

solutions = load_solutions()
scores = load_scores()
functions = score_functions()
functions_standardized = score_functions_standardized()

problem = generate_problem()
problem['beta_plus'] = 2
problem['beta_minus'] = 2

problem_false = {**problem}
problem_false['tp'] = problem_false['tp']*10

problem_tp0 = generate_problem_tp0()

@pytest.mark.parametrize("sol", list(solutions.keys()))
def test_solution(sol):
    """
    Testing a particular solution
    """

    score0 = functions[sol[0]](**{key: value for key, value in problem.items() if key in scores[sol[0]]['args']})
    score1 = functions[sol[1]](**{key: value for key, value in problem.items() if key in scores[sol[1]]['args']})

    result = solutions[sol].evaluate({**problem,
                                      **{sol[0]: score0,
                                            sol[1]: score1}})

    flags = []

    for res in result:
        tp = res['tp']
        tn = res['tn']

        flags.append(abs(tp - problem['tp']) < 1e-8 and abs(tn - problem['tn']) < 1e-8)

    assert any(flags)

@pytest.mark.parametrize("sol", list(solutions.keys()))
def test_solution_false(sol):
    """
    Testing a particular solution
    """

    score0 = functions[sol[0]](**{key: value for key, value in problem.items() if key in scores[sol[0]]['args']})
    score1 = functions[sol[1]](**{key: value for key, value in problem.items() if key in scores[sol[1]]['args']})

    result = solutions[sol].evaluate({**problem, **{sol[0]: score0, sol[1]: score1}})

    flags = []

    for res in result:
        tp = res['tp']
        tn = res['tn']

        flags.append(abs(tp - problem_false['tp']) < 1e-8 and abs(tn - problem_false['tn']) < 1e-8)

    assert not any(flags)

@pytest.mark.parametrize("sol", list(solutions.keys()))
def test_solution_tp0(sol):
    """
    Testing a particular solution
    """

    score0 = functions[sol[0]](**{key: value for key, value in problem_tp0.items() if key in scores[sol[0]]['args']})
    score1 = functions[sol[1]](**{key: value for key, value in problem_tp0.items() if key in scores[sol[1]]['args']})

    result = solutions[sol].evaluate({**problem_tp0, **{sol[0]: score0, sol[1]: score1}})

    flags = []

    for res in result:
        if res.get('message', None) is None:
            tp = res['tp']
            tn = res['tn']

            flags.append(abs(tp - problem_tp0['tp']) < 1e-8 and abs(tn - problem_tp0['tn']) < 1e-8)

    assert flags == [] or any(flags)

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

    res = sol.evaluate({'p': Interval(-1, 1)})

    assert res['message'] == 'negative base'

    res = sol.evaluate({'p': IntervalUnion([Interval(-1, 1)])})

    assert res['message'] == 'negative base'

    res = sol.evaluate({'p': -1})

    assert res['message'] == 'negative base'

def test_solutions_to_dict():
    """
    Testing the to_dict functionality of Solutions
    """

    sol = solutions[list(solutions.keys())[0]]
    assert len(sol.to_dict()) == 2

"""
This module tests the solutions
"""

import pytest

from mlscorecheck.core import (load_solutions,
                                load_scores,
                                score_functions,
                                score_functions_standardized)

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

@pytest.mark.parametrize("sol", list(solutions.keys()))
def test_solution(sol):
    """
    Testing a particular solution
    """

    score0 = functions[sol[0]](**{key: value for key, value in problem.items() if key in scores[sol[0]]['args']})
    score1 = functions[sol[1]](**{key: value for key, value in problem.items() if key in scores[sol[1]]['args']})

    result = solutions[sol].evaluate({**problem, **{sol[0]: score0, sol[1]: score1}})

    flags = []

    for res in result:
        tp = res['results']['tp']
        tn = res['results']['tn']

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
        tp = res['results']['tp']
        tn = res['results']['tn']

        flags.append(abs(tp - problem_false['tp']) < 1e-8 and abs(tn - problem_false['tn']) < 1e-8)

    assert not any(flags)

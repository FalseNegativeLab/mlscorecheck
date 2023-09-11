"""
This module tests the solutions
"""

import pytest

from mlscorecheck.core import safe_call
from mlscorecheck.individual import (load_solutions,
                                Solution,
                                Interval,
                                IntervalUnion,
                                sqrt)

from mlscorecheck.scores import (score_functions_without_complements,
                                    score_functions_standardized_without_complements,
                                    score_specifications)

from mlscorecheck.individual import (generate_1_problem)

solutions = load_solutions()
scores = score_specifications
functions = score_functions_without_complements
functions_standardized = score_functions_standardized_without_complements

def check_tptn(tp, tn, result, eps=1e-5):
    """
    Checks the tp and tn values against the results

    Args:
        tp (int): the original tp value
        tn (int): the original tn value
        res (list(dict)): the results of the solution
        eps (float): the tolerance

    Returns:
        bool: whether the scores match
    """

    flags = []
    zero_division = False

    for res in result:
        if res.get('message') == 'zero division':
            zero_division = True
            continue
        if res.get('message') is not None:
            continue

        flags.append(abs(tp - res['tp']) < eps and abs(tn - res['tn']) < eps)

    return len(flags) == 0 or zero_division, not any(flags)

def adjust_evaluation(evaluation):
    """
    Making the evaluation inconsistent

    Args:
        evaluation (dict): the evaluation to be rendered inconsistent
    """
    if evaluation['tn'] != 0:
        evaluation['tn'] = evaluation['tn'] * 10 + 5
    elif evaluation['tp'] != 0:
        evaluation['tp'] = evaluation['tp'] * 10 + 5

@pytest.mark.parametrize("sol", list(solutions.keys()))
@pytest.mark.parametrize("zeros", [[], ['tp'], ['tn'], ['fp'], ['fn'], ['tp', 'fp'],
                                    ['tn', 'fn'], ['tn', 'tp'], ['fn', 'fp']])
@pytest.mark.parametrize("random_state", [3, 5, 7])
def test_solution(sol, zeros, random_state):
    """
    Testing a particular solution

    Args:
        sol (tuple(str)): the score name pair with a solution
        zeros (list(str)): the list of figures to set to zero
        random_state (int): the seed of the random state
    """

    evaluation, _ = generate_1_problem(zeros=zeros,
                                        add_complements=True,
                                        random_state=random_state)
    evaluation['beta_positive'] = 2
    evaluation['beta_negative'] = 2
    evaluation['sqrt'] = sqrt

    score0 = safe_call(functions[sol[0]], evaluation, scores[sol[0]].get('nans'))
    score1 = safe_call(functions[sol[1]], evaluation, scores[sol[1]].get('nans'))

    if score0 is None or score1 is None:
        return

    result = solutions[sol].evaluate({**evaluation,
                                      **{sol[0]: score0,
                                            sol[1]: score1}})

    print(score0, score1, result)

    solvable, failed = check_tptn(evaluation['tp'], evaluation['tn'], result)
    assert solvable or not failed

@pytest.mark.parametrize("sol", list(solutions.keys()))
@pytest.mark.parametrize("zeros", [[], ['tp'], ['tn'], ['fp'], ['fn'], ['tp', 'fp'],
                                    ['tn', 'fn']])
@pytest.mark.parametrize("random_state", [3, 5, 7])
def test_solution_failure(sol, zeros, random_state):
    """
    Testing a particular solution with failure

    Args:
        sol (tuple(str)): the score name pair with a solution
        zeros (list(str)): the list of figures to set to zero
        random_state (int): the seed of the random state
    """

    evaluation, _ = generate_1_problem(zeros=zeros,
                                        add_complements=True,
                                        random_state=random_state)
    evaluation['beta_positive'] = 2
    evaluation['beta_negative'] = 2
    evaluation['sqrt'] = sqrt

    score0 = safe_call(functions[sol[0]], evaluation, scores[sol[0]].get('nans'))
    score1 = safe_call(functions[sol[1]], evaluation, scores[sol[1]].get('nans'))

    if score0 is None or score1 is None:
        return

    adjust_evaluation(evaluation)

    result = solutions[sol].evaluate({**evaluation,
                                      **{sol[0]: score0,
                                            sol[1]: score1}})

    solvable, failed = check_tptn(evaluation['tp'], evaluation['tn'], result)
    assert solvable or failed

def test_solution_object():
    """
    Testing the solution abstraction
    """

    sol = Solution(solution={'tp': {'expression': 'p*sens', 'symbols': ['p', 'sens']}},
                    conditions=[{'expression': 'p', 'symbols': ['p'],
                                    'mode': 'non-negative', 'depth': 1}])

    assert len(sol.to_dict()) == 2

    res = sol.evaluate({'p': 3, 'sens': 1})

    assert len(res) > 0

    assert sol.check_non_zero(0.0)

    assert not sol.check_non_zero(1)

def test_solution_non_negatives():
    """
    Testing the non-negative base detection in the solution
    """

    sol = Solution({'tp': {'expression': 'p', 'symbols': ['p']}},
                    conditions=[{'expression': 'p', 'symbols': ['p'],
                                    'mode': 'non-negative', 'depth': 1}])

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

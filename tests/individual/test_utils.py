"""
Testing the check functionality
"""

from mlscorecheck.core import NUMERICAL_TOLERANCE
from mlscorecheck.individual import (
    create_intervals,
    create_problems_2,
    load_solutions,
    Interval,
    resolve_aliases_and_complements,
    is_less_than_zero,
    unify_results,
    IntervalUnion,
)
from mlscorecheck.scores import score_functions_with_solutions, score_specifications

scores = score_specifications
functions = score_functions_with_solutions
solutions = load_solutions()

random_seeds = [5]


def test_resolve_aliases_and_complements():
    """
    Resolve the score aliases and complemnets
    """

    assert resolve_aliases_and_complements({"err": 0.4, "f1": 0.8, "sens": 0.3}) == {
        "acc": 0.6,
        "f1p": 0.8,
        "sens": 0.3,
    }


def test_create_intervals():
    """
    Testing the create intervals function
    """
    scores_ = resolve_aliases_and_complements({"acc": 0.6, "tpr": 0.4, "fnr": 0.6})
    intervals = create_intervals(scores_, eps=1e-4)
    assert abs(intervals["acc"].lower_bound - 0.5999) <= 1.1 * NUMERICAL_TOLERANCE
    assert abs(intervals["acc"].upper_bound - 0.6001) <= 1.1 * NUMERICAL_TOLERANCE
    assert abs(intervals["sens"].lower_bound - 0.3999) <= 1.1 * NUMERICAL_TOLERANCE
    assert abs(intervals["sens"].upper_bound - 0.4001) <= 1.1 * NUMERICAL_TOLERANCE

    scores_ = resolve_aliases_and_complements({"fnr": 0.6})
    intervals = create_intervals(scores_, eps=1e-4)
    assert abs(intervals["sens"].lower_bound - 0.3999) <= 1.1 * NUMERICAL_TOLERANCE
    assert abs(intervals["sens"].upper_bound - 0.4001) <= 1.1 * NUMERICAL_TOLERANCE

    intervals = create_intervals({"kappa": 0.5}, eps=1e-4)
    assert abs(intervals["kappa"].lower_bound - 0.4999) <= 1.1 * NUMERICAL_TOLERANCE
    assert abs(intervals["kappa"].upper_bound - 0.5001) <= 1.1 * NUMERICAL_TOLERANCE


def test_create_problems_2():
    """
    Testing the create problems function
    """

    problems = create_problems_2(["acc", "sens", "spec", "ppv"])

    assert len(problems) == 6 * 2


def test_is_less_than_zero():
    """
    Testing the "is less than zero" functionality
    """
    assert is_less_than_zero(-5)
    assert is_less_than_zero(Interval(-4, -3))
    assert is_less_than_zero(IntervalUnion([Interval(-4, -3)]))


def test_unify_results():
    """
    Testing the unification of results
    """

    assert unify_results([]) is None
    assert unify_results([None, None]) is None
    assert unify_results([2, None]) == [2]
    assert unify_results([Interval(2, 3)]) == IntervalUnion([Interval(2, 3)])
    assert unify_results([IntervalUnion([Interval(2, 3)])]) == IntervalUnion(
        [Interval(2, 3)]
    )
    assert unify_results([Interval(2, 3), 5]) == IntervalUnion(
        [Interval(2, 3), Interval(5, 5)]
    )

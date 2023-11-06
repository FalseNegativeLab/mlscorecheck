"""
This module tests the problem solver
"""

import pytest

from mlscorecheck.symbolic import ProblemSolver, Symbols
from mlscorecheck.symbolic import get_symbolic_toolkit, get_all_objects
from mlscorecheck.symbolic import (
    _collect_denominators_and_bases,
    collect_denominators_and_bases,
)
from mlscorecheck.symbolic import check_recurrent_solution

symbolic_toolkit = get_symbolic_toolkit()


@pytest.mark.skipif(symbolic_toolkit is None, reason="no symbolic toolkit available")
def test_problem_solver():
    """
    Testing the problem solver
    """
    base_scores = get_all_objects(symbolic_toolkit)

    prob_sol = ProblemSolver(base_scores["sens"], base_scores["spec"])
    prob_sol.solve()
    prob_sol.edge_cases()
    assert prob_sol.get_solution() is not None

    prob_sol = ProblemSolver(base_scores["sens"], base_scores["acc"])
    prob_sol.solve()
    prob_sol.edge_cases()
    assert prob_sol.get_solution() is not None

    prob_sol = ProblemSolver(base_scores["acc"], base_scores["sens"])
    prob_sol.solve()
    prob_sol.edge_cases()
    assert prob_sol.get_solution() is not None

    prob_sol = ProblemSolver(base_scores["acc"], base_scores["spec"])
    prob_sol.solve()
    prob_sol.edge_cases()
    assert prob_sol.get_solution() is not None

    prob_sol = ProblemSolver(base_scores["spec"], base_scores["acc"])
    prob_sol.solve()
    prob_sol.edge_cases()
    assert prob_sol.get_solution() is not None

    prob_sol = ProblemSolver(base_scores["acc"], base_scores["f1p"])
    prob_sol.solve()
    prob_sol.edge_cases()
    assert prob_sol.get_solution() is not None

    prob_sol = ProblemSolver(base_scores["fm"], base_scores["ppv"])
    prob_sol.solve()
    prob_sol.edge_cases()
    assert prob_sol.get_solution() is not None

    prob_sol = ProblemSolver(base_scores["ppv"], base_scores["fm"])
    prob_sol.solve()
    prob_sol.edge_cases()
    assert prob_sol.get_solution() is not None

    prob_sol = ProblemSolver(base_scores["ppv"], base_scores["gm"])
    prob_sol.solve()
    prob_sol.edge_cases()
    assert prob_sol.get_solution() is not None


@pytest.mark.skipif(symbolic_toolkit is None, reason="no symbolic toolkit available")
def test_root_in_edge_cases():
    """
    Testing the detection of a root
    """
    symbols = Symbols(symbolic_toolkit)

    denoms = []
    bases = []

    _collect_denominators_and_bases(
        symbols.tp**0.5, denoms, bases, symbols.algebra, depth=1
    )

    assert bases

    denoms = []
    bases = []
    _collect_denominators_and_bases(
        (symbols.tp + symbols.tn) ** 0.5, denoms, bases, symbols.algebra, depth=1
    )

    assert bases


@pytest.mark.skipif(symbolic_toolkit is None, reason="no symbolic toolkit available")
def test_depths():
    """
    Testing the proper use of depths
    """
    symbols = Symbols(symbolic_toolkit)

    tp = symbols.tp
    tn = symbols.tn
    sin = symbols.algebra.algebra.sin

    expr = sin((8 + tp / (1 + tn)) ** 2) / sin(tp / (1 + tn))

    denoms, bases = collect_denominators_and_bases(expr, symbols.algebra)
    assert denoms

    expr = sin(8 + sin(tp**0.5)) / sin(tp**0.5)

    denoms, bases = collect_denominators_and_bases(expr, symbols.algebra)
    assert bases


def test_recurrent_solution():
    """
    Testing the recurrent solution test
    """
    assert check_recurrent_solution("a", ["a"])

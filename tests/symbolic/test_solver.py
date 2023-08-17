"""
This module tests the problem solver
"""

import pytest

from mlscorecheck.symbolic import ProblemSolver, Symbols
from mlscorecheck.symbolic import get_base_objects, get_symbolic_toolkit
from mlscorecheck.symbolic import _collect_denominators_and_bases

symbolic_toolkit = get_symbolic_toolkit()

@pytest.mark.skipif(symbolic_toolkit is None, reason='no symbolic toolkit available')
def test_problem_solver():
    """
    Testing the problem solver
    """
    base_scores = get_base_objects(symbolic_toolkit)

    ps = ProblemSolver(base_scores['sens'], base_scores['spec'])
    ps.solve()
    ps.edge_cases()
    assert ps.get_solution() is not None

    ps = ProblemSolver(base_scores['sens'], base_scores['acc'])
    ps.solve()
    ps.edge_cases()
    assert ps.get_solution() is not None

    ps = ProblemSolver(base_scores['acc'], base_scores['spec'])
    ps.solve()
    ps.edge_cases()
    assert ps.get_solution() is not None

    ps = ProblemSolver(base_scores['acc'], base_scores['f1p'])
    ps.solve()
    ps.edge_cases()
    assert ps.get_solution() is not None

@pytest.mark.skipif(symbolic_toolkit is None, reason='no symbolic toolkit available')
def test_root_in_edge_cases():
    """
    Testing the detection of a root
    """
    symbols = Symbols(symbolic_toolkit)

    denoms = []
    bases = []

    _collect_denominators_and_bases(symbols.tp**0.5, denoms, bases, symbols.algebra)

    assert bases

    denoms = []
    bases = []
    _collect_denominators_and_bases((symbols.tp + symbols.tn)**0.5, denoms, bases, symbols.algebra)

    assert bases

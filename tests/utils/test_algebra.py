"""
This module tests the functionalities of the algebra abstractions
"""

import pytest

from mlscorecheck.utils import symbolic_toolkits
from mlscorecheck.utils import Algebra, SympyAlgebra, SageAlgebra

@pytest.mark.skipif('sympy' not in symbolic_toolkits, reason='sympy not installed')
def test_sympy_algebra():
    """
    This module tests the algebra base class
    """

    alg = SympyAlgebra()

    assert alg.create_symbol('tp') is not None

    assert alg.create_symbol('tp', lower_bound=0, upper_bound=1) is not None

    tp = alg.create_symbol('tp')
    p = alg.create_symbol('p')

    assert alg.num_denom(tp/p) == (tp, p)

    assert alg.simplify(p) == p

    assert alg.solve(p, p) == [{p: 0}]

    assert alg.subs(p, {p: 5}) == 5

    assert alg.args(p + tp) == {p, tp}

    assert alg.is_trivial(None)
    assert alg.is_trivial(1)
    assert not alg.is_trivial(p)

    assert alg.is_root(tp**0.5)
    assert not alg.is_root(tp)

    assert alg.operands(tp + p**2) == (tp, p**2)

    assert sorted(alg.free_symbols(tp + p)) == sorted(['p', 'tp'])

@pytest.mark.skipif('sage' not in symbolic_toolkits, reason='sage not installed')
def test_sage_algebra():
    """
    This module tests the algebra base class
    """

    alg = SageAlgebra()

    assert alg.create_symbol('tp') is not None

    assert alg.create_symbol('tp', lower_bound=0, upper_bound=1) is not None

    tp = alg.create_symbol('tp')
    p = alg.create_symbol('p')

    assert alg.num_denom(tp/p) == (tp, p)

    assert alg.simplify(p) == p

    assert alg.solve(p, p) == [{p: 0}]

    assert alg.subs(p, {p: 5}) == 5

    assert alg.args(p + tp) == {p, tp}

    assert alg.is_trivial(None)
    assert alg.is_trivial(1)
    assert not alg.is_trivial(p)

    assert alg.is_root(tp**0.5)
    assert not alg.is_root(tp)

    assert alg.operands(tp + p**2) == (tp, p**2)

    assert sorted(alg.free_symbols(tp + p)) == sorted(['p', 'tp'])

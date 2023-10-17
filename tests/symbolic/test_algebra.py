"""
This module tests the functionalities of the algebra abstractions
"""

import pytest

from mlscorecheck.symbolic import symbolic_toolkits
from mlscorecheck.symbolic import SympyAlgebra, SageAlgebra, Symbols

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
    assert alg.is_trivial(alg.simplify(p/p))
    assert not alg.is_trivial(p)

    assert alg.is_root(tp**0.5)
    assert not alg.is_root(tp)

    assert sorted(map(str, list(alg.operands(tp + p**2)))) == sorted(map(str, [tp, p**2]))

    assert sorted(map(str, alg.free_symbols(tp + p))) == sorted(map(str, ['p', 'tp']))

    assert alg.is_power(tp**2)

    assert alg.is_division(tp/p)

    assert alg.is_division(p/tp)

    assert alg.is_division(1/p)

    assert not alg.is_division(p)

@pytest.mark.skipif('sage' not in symbolic_toolkits, reason='sage not installed')
def test_sage_algebra():
    """
    This module tests the algebra base class
    """

    alg = SageAlgebra()

    assert alg.create_symbol('tp') is not None

    assert alg.create_symbol('x', nonnegative=True) is not None
    assert alg.create_symbol('y', positive=True) is not None
    assert alg.create_symbol('z', negative=True) is not None
    assert alg.create_symbol('a', nonpositive=True) is not None
    assert alg.create_symbol('b', real=True) is not None

    assert alg.create_symbol('tp', lower_bound=0, upper_bound=1) is not None

    tp = alg.create_symbol('tp')
    p = alg.create_symbol('p')

    assert alg.num_denom(tp/p) == (tp, p)

    assert alg.simplify(p) == p

    assert alg.solve(p, p) == [{p: 0}]

    assert alg.subs(p, {p: 5}) == 5

    assert alg.args(p + tp) == {p, tp}

    assert alg.is_trivial(None)
    assert alg.is_trivial(alg.simplify(p/p))
    assert not alg.is_trivial(p)

    assert alg.is_root(tp**0.5)
    assert not alg.is_root(tp)

    assert sorted(map(str, list(alg.operands(tp + p**2)))) == sorted(map(str, [tp, p**2]))

    assert sorted(map(str, alg.free_symbols(tp + p))) == sorted(map(str, ['p', 'tp']))

    assert alg.is_power(tp**2)

    assert alg.is_division(tp/p)

    assert alg.is_division(p/tp)

    assert alg.is_division(1/p)

    assert not alg.is_division(p)

@pytest.mark.skipif('sympy' not in symbolic_toolkits, reason='sympy not installed')
def test_symbols_sympy():
    """
    Testing the base symbols with sympy
    """

    symbols = Symbols(algebraic_system='sympy')

    assert len(symbols.to_dict()) == 7
    assert symbols.get_algebra() is not None

@pytest.mark.skipif('sage' not in symbolic_toolkits, reason='sage not installed')
def test_symbols_sage():
    """
    Testing the base symbols with sympy
    """

    symbols = Symbols(algebraic_system='sage')

    assert len(symbols.to_dict()) == 7
    assert symbols.get_algebra() is not None

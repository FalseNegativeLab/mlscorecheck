"""
This module tests the functions in the availability sub-module
"""

from mlscorecheck.symbolic import check_importability, get_symbolic_toolkit

def test_test_importability():
    """
    Testing the importability function
    """

    assert check_importability('itertools') == 'itertools'
    assert check_importability('asdfasdfasdfasdfasdf') is None

def test_get_symbolic_toolkit():
    """
    Testing the get_symbolic_toolkit function
    """

    assert get_symbolic_toolkit() in [None, 'sympy', 'sage']

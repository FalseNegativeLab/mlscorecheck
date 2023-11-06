"""
Testing the safe eval and safe call functionalities
"""

from mlscorecheck.core import safe_eval, safe_call


def test_safe_eval():
    """
    Testing the safe_eval
    """

    assert safe_eval("a*b", {"a": 2, "b": 3}) == 6


def mock_function(*, a):
    """
    A mock function doubling its parameter

    Args:
        a (int|float): a number

    Returns:
        int|float: the double of the parameter
    """
    return a * 2


def test_safe_call():
    """
    Testing the safe call
    """
    assert safe_call(mock_function, {"a": 2, "b": 3}, [{"a": 2, "b": 3}]) is None
    assert safe_call(mock_function, {"a": 2, "b": 3}, [{"a": 2, "b": 4}]) == 4
    assert safe_call(mock_function, {"a": 2, "b": 3}, [{"a": "a"}]) is None
    assert safe_call(mock_function, {"a": 2, "b": 3}) == 4

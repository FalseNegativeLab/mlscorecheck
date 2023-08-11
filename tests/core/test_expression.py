"""
Testing the expression class
"""

from mlscorecheck.core import Expression

def test_expression():
    """
    Testing the expression abstraction
    """

    exp = Expression('a*b', ['a', 'b'])
    assert exp.evaluate({'a': 2, 'b': 3}) == 6
    assert len(exp.to_dict()) == 3

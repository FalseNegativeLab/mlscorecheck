"""
Testing the dict aggregation functionalities
"""

from mlscorecheck.core import dict_mean, dict_minmax


def test_dict_mean():
    """
    Testing the calculation of the mean of dicts
    """

    means = dict_mean([{"a": 1, "b": 2}, {"a": 3, "b": 4}])
    assert means["a"] == 2
    assert means["b"] == 3


def test_dict_minmax():
    """
    Testing calculating the minmax of dicts
    """

    minmax = dict_minmax([{"a": 1, "b": 2}, {"a": 3, "b": 4}, {"a": 0, "b": 2}])

    assert minmax["a"] == [0, 3]
    assert minmax["b"] == [2, 4]

"""
A function used to test the output of the linear programming
"""

import warnings

__all__ = ["evaluate_timeout"]


def evaluate_timeout(result: dict, inconsistent: bool):
    """
    Evaluate the stopped or succeeded tests

    Args:
        result (dict): the resulting dictionary
        inconsistent (bool): whether the test should be consistent or not
    """
    if result['lp_status'] != 'timeout':
        assert result['inconsistency'] == inconsistent
    else:
        warnings.warn("test timed out")

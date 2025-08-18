"""
Testing the timeout evaluation
"""

import warnings

from ._evaluate import evaluate_timeout


def test_evaluate_timeout():
    """
    Testing the timeout evaluation
    """

    with warnings.catch_warnings(record=True) as warn:
        evaluate_timeout({'lp_status': 'timeout'}, False)
        assert len(warn) == 1

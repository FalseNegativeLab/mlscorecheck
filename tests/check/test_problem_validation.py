"""
This module tests the problem validation
"""

import pytest

from mlscorecheck.check import validate_problem_specification

def test_validate_problem_specification():
    """
    Testing the problem validation
    """
    with pytest.raises(ValueError):
        validate_problem_specification({})

    with pytest.raises(ValueError):
        validate_problem_specification([{}])

    with pytest.raises(ValueError):
        validate_problem_specification([{'folds': [{}]}])

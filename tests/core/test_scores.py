"""
This module tests all scores
"""

import numpy as np

import pytest

import numbers

from mlscorecheck.utils import (generate_problem)
from mlscorecheck.core import (score_functions,
                                score_functions_standardized,
                                load_scores,
                                score_function_aliases)

functions = score_functions(complements=True)
functions_standardized = score_functions_standardized(complements=True)
scores = load_scores()
aliases = score_function_aliases()

problem = generate_problem()
problem['beta_plus'] = 2
problem['beta_minus'] = 2

@pytest.mark.parametrize("score", list(scores.keys()))
def test_score_and_standardized(score):
    """
    This module tests a score against the standardized score
    """

    value = functions[score](**{arg: problem[arg] for arg in scores[score]['args']})
    value_standard = functions_standardized[score](**{arg: problem[arg]
                                                for arg in scores[score]['args_standardized']})

    assert abs(value - value_standard) < 1e-8

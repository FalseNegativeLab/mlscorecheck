"""
This module tests all scores
"""

import numpy as np

import pytest

import numbers

from mlscorecheck.utils import (generate_problem)
from mlscorecheck.core import (score_functions_with_complements,
                                score_functions_standardized_with_complements,
                                load_scores,
                                score_function_aliases)

functions = score_functions_with_complements
functions_standardized = score_functions_standardized_with_complements
scores = load_scores()
aliases = score_function_aliases

short_formula_scores = {key: score for key, score in scores.items() if 'short_args' in score}
complement_scores = {key: score for key, score in scores.items() if 'complement' in score}

problem = generate_problem()
problem['beta_plus'] = 2
problem['beta_minus'] = 2
problem['sqrt'] = np.sqrt

for key in scores:
    problem[key] = functions[key](**{arg: problem[arg] for arg in scores[key]['args']})

for key, value in aliases.items():
    problem[key] = problem[value]

@pytest.mark.parametrize("score", list(scores.keys()))
def test_score_and_standardized(score):
    """
    This module tests a score against the standardized score
    """

    value = functions[score](**{arg: problem[arg] for arg in scores[score]['args']})
    value_standard = functions_standardized[score](**{arg: problem[arg]
                                                for arg in scores[score]['args_standardized']})

    assert abs(value - value_standard) < 1e-8

@pytest.mark.parametrize("score", list(short_formula_scores.keys()))
def test_short_formulas(score):
    """
    This module tests a score against the short formula
    """

    value = functions[score](**{arg: problem[arg] for arg in scores[score]['args']})

    short_formula = short_formula_scores[score]['short_formula']

    value_short = eval(short_formula, problem)

    assert abs(value - value_short) < 1e-8

@pytest.mark.parametrize("score", list(complement_scores.keys()))
def test_short_complements(score):
    """
    This module tests a score against the short formula
    """

    value = functions[score](**{arg: problem[arg] for arg in scores[score]['args']})
    comp_score = scores[score]['complement']
    comp_value = functions[comp_score](**{arg: problem[arg] for arg in scores[comp_score]['args']})

    assert abs(1 - (value + comp_value)) < 1e-8

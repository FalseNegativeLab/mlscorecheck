"""
This module tests all scores
"""

import numpy as np

import pytest

from mlscorecheck.core import safe_call, safe_eval
from mlscorecheck.individual import generate_1_problem
from mlscorecheck.scores import (score_functions_with_complements,
                                score_functions_standardized_with_complements,
                                score_specifications,
                                score_function_aliases)

functions = score_functions_with_complements
functions_standardized = score_functions_standardized_with_complements
scores = score_specifications
aliases = score_function_aliases

short_formula_scores = {key: score for key, score in scores.items() if 'short_args' in score}
complement_scores = {key: score for key, score in scores.items() if 'complement' in score}

evaluation, problem = generate_1_problem(random_state=5,
                                            add_complements=True)
evaluation['beta_plus'] = 2
evaluation['beta_minus'] = 2
evaluation['sqrt'] = np.sqrt

for key in scores:
    evaluation[key] = functions[key](**{arg: evaluation[arg] for arg in scores[key]['args']})

for key, val in aliases.items():
    evaluation[key] = evaluation[val]

@pytest.mark.parametrize("score", list(scores.keys()))
def test_score_and_standardized(score):
    """
    This module tests a score against the standardized score
    """

    value = safe_call(functions[score], evaluation)
    value_standard = safe_call(functions_standardized[score], evaluation)

    assert abs(value - value_standard) < 1e-8

@pytest.mark.parametrize("score", list(short_formula_scores.keys()))
def test_short_formulas(score):
    """
    This module tests a score against the short formula
    """
    value = safe_call(functions[score], evaluation)

    short_formula = short_formula_scores[score]['short_formula']

    value_short = safe_eval(short_formula, evaluation)

    assert abs(value - value_short) < 1e-8

@pytest.mark.parametrize("score", list(complement_scores.keys()))
def test_short_complements(score):
    """
    This module tests a score against the short formula
    """

    value = safe_call(functions[score], evaluation)

    comp_score = scores[score]['complement']
    comp_value = safe_call(functions[comp_score], evaluation)

    assert abs(1 - (value + comp_value)) < 1e-8

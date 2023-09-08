"""
This module tests the check functionalities based on the cascading
solution of scores for intervals.
"""

import pytest

import numpy as np

from mlscorecheck.core import NUMERICAL_TOLERANCE, safe_call
from mlscorecheck.individual import (check_scores_tptn,
                                        tptn_solutions,
                                        Interval,
                                        generate_1_problem,
                                        calculate_scores,
                                        sqrt)
from mlscorecheck.scores import (score_functions_standardized_without_complements,
                                    score_specifications)

score_functions = score_functions_standardized_without_complements

@pytest.mark.parametrize('score', list(tptn_solutions.keys()))
@pytest.mark.parametrize('figure', ['tp', 'tn'])
@pytest.mark.parametrize('zeros', [[], ['tp'], ['tn'], ['fp'], ['fn'],
                                    ['tp', 'tn'], ['tp', 'fp'], ['fp', 'fn'], ['tn', 'fn']])
@pytest.mark.parametrize('random_seed', list(range(20)))
def test_tptn_solutions(score, figure, zeros, random_seed):
    """
    This function tests the tp-tn solutions
    """
    if tptn_solutions[score][figure] is None:
        return

    evaluation, _ = generate_1_problem(random_state=random_seed,
                                                zeros=zeros)

    evaluation['beta_plus'] = 1
    evaluation['beta_minus'] = 1
    evaluation['sqrt'] = sqrt

    score_val = safe_call(score_functions[score],
                            evaluation,
                            score_specifications[score].get('nans_standardized'))

    if score_val is None:
        return

    print(score, figure, evaluation, score_val)

    figure_value = tptn_solutions[score][figure](**evaluation, **{score: score_val})

    if (figure_value is None):
        return None
    if isinstance(figure_value, list):
        if all(figure_tmp is None for figure_tmp in figure_value):
            return
        figure_value = [tmp for tmp in figure_value if not tmp is None]

    print(figure_value)

    assert np.any(np.abs(np.array(figure_value) - evaluation[figure]) <= 1e-6)

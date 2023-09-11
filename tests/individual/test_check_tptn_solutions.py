"""
This module tests the check functionalities based on the cascading
solution of scores for intervals.
"""

import pytest

from mlscorecheck.individual import (check_scores_tptn_pairs,
                                        _check_scores_tptn_pairs,
                                        generate_1_problem,
                                        calculate_scores,
                                        sqrt)
from mlscorecheck.scores import (score_functions_standardized_without_complements)

score_functions = score_functions_standardized_without_complements

@pytest.mark.parametrize('figure', ['tp', 'tn', None])
@pytest.mark.parametrize('zeros', [[], ['tp'], ['tn'], ['fp'], ['fn'],
                                    ['tp', 'tn'], ['tp', 'fp'], ['fp', 'fn'], ['tn', 'fn']])
@pytest.mark.parametrize('random_seed', list(range(10)))
def test_check_scores_tptn_pairs(figure, zeros, random_seed):
    """
    This function tests the check functionality

    Args:
        figure (str): the figure to solve_for (``tp``/``tn``)
        zeros (list(str)): the list of figures to set zero
        random_seed (int): the random seed to use
    """

    evaluation, problem = generate_1_problem(random_state=random_seed,
                                                zeros=zeros)

    evaluation['beta_positive'] = 1
    evaluation['beta_negative'] = 1
    evaluation['sqrt'] = sqrt

    scores = calculate_scores(evaluation)
    scores = {key: value for key, value in scores.items() if value is not None}

    scores['beta_positive'] = 1
    scores['beta_negative'] = 1

    results = _check_scores_tptn_pairs(p=problem['p'],
                                        n=problem['n'],
                                        scores=scores,
                                        eps=1e-4,
                                        solve_for=figure)

    assert not results['inconsistency']

@pytest.mark.parametrize('prefilter_by_pairs', [True, False])
@pytest.mark.parametrize('figure', ['tp', 'tn', None])
@pytest.mark.parametrize('zeros', [[], ['tp'], ['tn'], ['fp'], ['fn'],
                                    ['tp', 'tn'], ['tp', 'fp'], ['fp', 'fn'], ['tn', 'fn']])
@pytest.mark.parametrize('random_seed', list(range(10)))
def test_check_scores_tptn_pairs_prefilter(figure, zeros, random_seed, prefilter_by_pairs):
    """
    This function tests the check functionality with prefiltering

    Args:
        figure (str): the figure to solve_for (``tp``/``tn``)
        zeros (list(str)): the list of figures to set zero
        random_seed (int): the random seed to use
        prefilter_by_pairs (bool): whether to prefilter by the pairwise solutions
    """

    evaluation, problem = generate_1_problem(random_state=random_seed,
                                                zeros=zeros)

    evaluation['beta_positive'] = 1
    evaluation['beta_negative'] = 1
    evaluation['sqrt'] = sqrt

    scores = calculate_scores(evaluation)
    scores = {key: value for key, value in scores.items() if value is not None}

    scores['beta_positive'] = 1
    scores['beta_negative'] = 1

    results = check_scores_tptn_pairs(p=problem['p'],
                                        n=problem['n'],
                                        scores=scores,
                                        eps=1e-4,
                                        solve_for=figure,
                                        prefilter_by_pairs=prefilter_by_pairs)

    assert not results['inconsistency']

@pytest.mark.parametrize('prefilter_by_pairs', [True, False])
@pytest.mark.parametrize('figure', ['tp', 'tn', None])
@pytest.mark.parametrize('zeros', [[], ['tp'], ['tn'], ['fp'], ['fn'],
                                    ['tp', 'tn'], ['tp', 'fp'], ['fp', 'fn'], ['tn', 'fn']])
@pytest.mark.parametrize('random_seed', list(range(10)))
def test_check_scores_tptn_pairs_prefilter_failure(figure,
                                                    zeros,
                                                    random_seed,
                                                    prefilter_by_pairs):
    """
    This function tests the check functionality with failure

    Args:
        figure (str): the figure to solve_for (``tp``/``tn``)
        zeros (list(str)): the list of figures to set zero
        random_seed (int): the random seed to use
        prefilter_by_pairs (bool): whether to prefilter by pair-solutions
    """

    evaluation, problem = generate_1_problem(random_state=random_seed,
                                                zeros=zeros)

    evaluation['beta_positive'] = 1
    evaluation['beta_negative'] = 1
    evaluation['sqrt'] = sqrt

    scores = calculate_scores(evaluation)
    scores = {key: value for key, value in scores.items() if value is not None}
    scores['bacc'] = 0.8
    scores['sens'] = 0.85
    scores['spec'] = 0.86

    scores['beta_positive'] = 1
    scores['beta_negative'] = 1

    results = check_scores_tptn_pairs(p=problem['p'],
                                        n=problem['n'],
                                        scores=scores,
                                        eps=1e-4,
                                        solve_for=figure,
                                        prefilter_by_pairs=prefilter_by_pairs)

    assert results['inconsistency']

def test_check_parametrization():
    """
    Testing the parametrization
    """

    with pytest.raises(ValueError):
        _check_scores_tptn_pairs(5, 10, scores={}, eps=1e-4, solve_for='dummy')

"""
Testing the linear programming components
"""

import pytest

from mlscorecheck.aggregated import (add_bounds,
                                        calculate_scores_lp,
                                        _initialize_folds,
                                        create_lp_problem_inner,
                                        create_target,
                                        populate_solution,
                                        check_aggregated_scores,
                                        generate_problems_with_evaluations,
                                        calculate_scores_datasets,
                                        _expand_datasets)

@pytest.mark.parametrize("strategy", [('mor', 'mor'), ('rom', 'rom'), ('mor', 'rom')])
@pytest.mark.parametrize("random_state", [3, 5, 7, 11, 23])
def test_check_aggregated_scores_success(strategy, random_state):
    """
    Testing the checking of aggregated scores with success
    """

    figures, problems = generate_problems_with_evaluations(n_problems=3,
                                                            random_state=random_state)

    scores = calculate_scores_datasets(figures,
                                        strategy=strategy)

    inconsistency = check_aggregated_scores(scores,
                                        eps=1e-4,
                                        datasets=problems,
                                        strategy=strategy)

    assert not inconsistency

@pytest.mark.parametrize("strategy", [('mor', 'mor'), ('rom', 'rom'), ('mor', 'rom')])
@pytest.mark.parametrize("random_state", [3, 5, 7, 11, 23])
def test_check_aggregated_scores_failure(strategy, random_state):
    """
    Testing the checking of aggregated scores with failure
    """

    figures, problems = generate_problems_with_evaluations(n_problems=3,
                                                            random_state=random_state)

    scores = calculate_scores_datasets(figures,
                                        strategy=strategy)

    scores['bacc'] = 0.9
    scores['acc'] = 0.1

    inconsistency = check_aggregated_scores(scores,
                                        eps=1e-4,
                                        datasets=problems,
                                        strategy=strategy)

    assert inconsistency

@pytest.mark.parametrize("strategy", [('mor', 'mor'), ('rom', 'rom'), ('mor', 'rom')])
@pytest.mark.parametrize("random_state", [3, 5, 7, 11, 23])
def test_check_aggregated_scores_success_score_bounds(strategy, random_state):
    """
    Testing the checking of aggregated scores with success and bounds on scores
    """

    figures, problems = generate_problems_with_evaluations(n_problems=3,
                                                            random_state=random_state)

    scores = calculate_scores_datasets(figures,
                                        strategy=strategy)

    inconsistency = check_aggregated_scores(scores,
                                        eps=1e-4,
                                        datasets=problems,
                                        strategy=strategy)

    assert not inconsistency

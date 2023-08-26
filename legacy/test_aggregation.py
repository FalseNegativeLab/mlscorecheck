"""
Testing the linear programming components
"""

import pytest

import numpy as np

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

def determine_bounds(datasets, mode='score'):
    if mode in {'score', 'fold_score', 'fold_tptn'}:
        if mode in ('score', 'fold_score'):
            bounds = {'acc': [np.inf, -np.inf],
                        'sens': [np.inf, -np.inf],
                        'spec': [np.inf, -np.inf],
                        'bacc': [np.inf, -np.inf]}
        elif mode == 'fold_tptn':
            bounds = {'tp': [np.inf, -np.inf],
                        'tn': [np.inf, -np.inf]}

        for dataset in datasets:
            for fold in dataset['folds']:
                for key in bounds:
                    if fold[key] < bounds[key][0]:
                        bounds[key][0] = fold[key]
                    if fold[key] > bounds[key][1]:
                        bounds[key][1] = fold[key]
    elif mode in {'tptn'}:
        bounds = {'tp': [np.inf, -np.inf],
                    'tn': [np.inf, -np.inf]}
        for dataset in datasets:
            sums = {'tp': 0,
                    'tn': 0}
            for fold in dataset['folds']:
                for key in sums:
                    sums[key] += fold[key]
            for key in bounds:
                if sums[key] < bounds[key][0]:
                    bounds[key][0] = sums[key]
                if sums[key] > bounds[key][1]:
                    bounds[key][1] = sums[key]

    return bounds

@pytest.mark.parametrize("strategy", [('mor', 'mor'), ('rom', 'rom'), ('mor', 'rom')])
@pytest.mark.parametrize("random_state", [3, 5, 7, 11, 23])
@pytest.mark.parametrize("modes", [[], ['score'], ['tptn'], ['fold_score'], ['fold_tptn'],
                                ['score', 'tptn'], ['fold_score', 'tptn'], ['score', 'fold_tptn'],
                                ['fold_score', 'fold_tptn']])
def test_check_aggregated_scores_success_score_bounds(strategy, modes, random_state):
    """
    Testing the checking of aggregated scores with success and bounds on scores
    """

    figures, problems = generate_problems_with_evaluations(n_problems=3,
                                                            random_state=random_state)

    scores, populated = calculate_scores_datasets(figures,
                                                    strategy=strategy,
                                                    return_populated=True)

    print(strategy)
    print(populated)

    for mode in modes:
        bounds = determine_bounds(populated, mode=mode)
        print(mode, bounds)
        for dataset in problems:
            dataset[f'{mode}_bounds'] = bounds

    inconsistency, details = check_aggregated_scores(scores,
                                                    eps=1e-3,
                                                    datasets=problems,
                                                    strategy=strategy,
                                                    return_details=True)

    assert not inconsistency

    for mode in modes:
        if mode in ('score', 'tptn'):
            assert f'{mode}_bounds_check' in details['configuration'][0]
        elif mode in ('fold_score', 'fold_tptn'):
            assert f'{mode[5:]}_bounds_check' in details['configuration'][0]['folds'][0]

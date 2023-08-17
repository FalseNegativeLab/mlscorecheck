"""
Testing the use case regarding multiple datasets aggregated in mor
and kfold aggregated in mor manner.
"""

import numpy as np

from mlscorecheck.check import (check_multiple_datasets_mor_kfold_mor_scores)
from mlscorecheck.utils import (generate_problems_with_folds,
                                calculate_scores)

from ._score_ranges import (score_ranges,
                            calculate_scores_for_folds)

k = 4
eps = 10**(-k)

def test_consistent():
    """
    Testing a consistent configuration
    """
    folds, problem = generate_problems_with_folds(n_problems=3,
                                                    n_folds=3,
                                                    n_repeats=2,
                                                    random_seed=5)

    scores = calculate_scores(folds,
                                strategy=('mor', 'mor'),
                                rounding_decimals=k)

    flag, details = check_multiple_datasets_mor_kfold_mor_scores(scores,
                                                                eps=eps,
                                                                datasets=problem,
                                                                return_details=True)

    assert flag

    scores_new = calculate_scores(details['configuration'],
                                    strategy=('mor', 'mor'))

    for key in scores_new:
        assert abs(scores[key] - scores_new[key]) <= eps

def test_consistent_differing_configurations():
    """
    Testing a consistent configuration
    """
    random_state = np.random.RandomState(5)

    folds, problem = [], []

    for _ in range(5):
        fold, prob = generate_problems_with_folds(n_problems=1,
                                                    n_folds=random_state.randint(2, 5),
                                                    n_repeats=random_state.randint(2, 5),
                                                    random_seed=5)
        folds.append(fold)
        problem.append(prob)

    scores = calculate_scores(folds,
                                strategy=('mor', 'mor'),
                                rounding_decimals=k)

    flag, details = check_multiple_datasets_mor_kfold_mor_scores(scores,
                                                                eps=eps,
                                                                datasets=problem,
                                                                return_details=True)

    assert flag

    scores_new = calculate_scores(details['configuration'],
                                    strategy=('mor', 'mor'))

    for key in scores_new:
        assert abs(scores[key] - scores_new[key]) <= eps

def test_failure():
    """
    Testing a failure
    """
    folds, problem = generate_problems_with_folds(n_problems=3,
                                                    n_folds=3,
                                                    n_repeats=2,
                                                    random_seed=5)

    scores = calculate_scores(folds,
                                strategy=('mor', 'mor'),
                                rounding_decimals=k)
    scores['bacc'] = 0.9

    flag, details = check_multiple_datasets_mor_kfold_mor_scores(scores,
                                                        eps=eps,
                                                        datasets=problem,
                                                        return_details=True)

    assert not flag

def test_failure_differing_configurations():
    """
    Testing a consistent configuration
    """
    random_state = np.random.RandomState(5)

    folds, problem = [], []

    for _ in range(5):
        fold, prob = generate_problems_with_folds(n_problems=1,
                                                    n_folds=random_state.randint(2, 5),
                                                    n_repeats=random_state.randint(2, 5),
                                                    random_seed=5)
        folds.append(fold)
        problem.append(prob)

    scores = calculate_scores(folds,
                                strategy=('mor', 'mor'),
                                rounding_decimals=k)

    scores['bacc'] = 0.9

    flag, details = check_multiple_datasets_mor_kfold_mor_scores(scores,
                                                                eps=eps,
                                                                datasets=problem,
                                                                return_details=True)

    assert not flag

def test_dataset():
    """
    Testing success with real dataset
    """
    problem = [{'dataset': 'common_datasets.ADA', 'n_folds': 2, 'n_repeats': 3},
                {'dataset': 'common_datasets.ecoli1', 'n_folds': 3, 'n_repeats': 5}]

    scores = {'acc': 0.9,
                'sens': 0.89,
                'spec': 0.91}

    flag = check_multiple_datasets_mor_kfold_mor_scores(scores,
                                                eps=0.01,
                                                datasets=problem,
                                                return_details=False)

    assert flag

def test_dataset_failure():
    """
    Testing failure with real dataset with extreme precision and random scores
    """
    problem = [{'dataset': 'common_datasets.ADA', 'n_folds': 2, 'n_repeats': 3},
                {'dataset': 'common_datasets.ecoli1', 'n_folds': 3, 'n_repeats': 5}]

    scores = {'acc': 0.9,
                'sens': 0.89,
                'spec': 0.91}

    flag = check_multiple_datasets_mor_kfold_mor_scores(scores,
                                                eps=0.00001,
                                                datasets=problem,
                                                return_details=False)

    assert not flag

def test_score_bounds_consistency():
    """
    Testing the consistency with score bounds
    """
    folds, problem = generate_problems_with_folds(n_problems=3,
                                                    n_repeats=3,
                                                    n_folds=2,
                                                    random_seed=5)

    scores = calculate_scores(folds,
                                strategy=('mor', 'mor'),
                                rounding_decimals=k)

    bounds = score_ranges(calculate_scores_for_folds(folds))

    for prob in problem:
        prob['score_bounds'] = bounds

    flag, details = check_multiple_datasets_mor_kfold_mor_scores(scores,
                                            eps=eps,
                                            datasets=problem,
                                            return_details=True)

    assert flag

    scores_new = calculate_scores(details['configuration'],
                                    strategy=('mor', 'mor'))

    for key in scores_new:
        assert abs(scores[key] - scores_new[key]) <= eps

def test_score_bounds_failure():
    """
    Testing the failure with score bounds
    """
    folds, problem = generate_problems_with_folds(n_problems=3,
                                                    n_repeats=2,
                                                    n_folds=3,
                                                    random_seed=5)

    scores = calculate_scores(folds,
                                strategy=('mor', 'mor'),
                                rounding_decimals=k)

    bounds = score_ranges(calculate_scores_for_folds(folds))
    bounds['acc'] = (0.99, 1.0)

    for prob in problem:
        prob['score_bounds'] = bounds

    flag, details = check_multiple_datasets_mor_kfold_mor_scores(scores,
                                            eps=eps,
                                            datasets=problem,
                                            return_details=True)

    assert not flag

def test_fold_configurations_success():
    """
    Testing success with manual fold configurations
    """
    """
    Testing a consistent configuration
    """
    problem = [{'fold_configuration': [{'p': 5, 'n': 20},
                                        {'p': 20, 'n': 5}]},
                {'fold_configuration': [{'p': 10, 'n': 5,
                                            'p': 20, 'n': 40}]}]

    scores = {'acc': 0.8,
                'sens': 0.79,
                'spec': 0.81}

    flag = check_multiple_datasets_mor_kfold_mor_scores(scores,
                                                eps=0.01,
                                                datasets=problem,
                                                return_details=False)

    assert flag

def test_fold_configurations_failure():
    """
    Testing failure with manual fold configurations
    """

    problem = [{'fold_configuration': [{'p': 5, 'n': 20},
                                        {'p': 20, 'n': 5}]},
                {'fold_configuration': [{'p': 10, 'n': 5,
                                            'p': 20, 'n': 40}]}]

    scores = {'acc': 0.8,
                'sens': 0.79,
                'spec': 0.81}

    flag = check_multiple_datasets_mor_kfold_mor_scores(scores,
                                                eps=0.00001,
                                                datasets=problem,
                                                return_details=False)

    assert not flag

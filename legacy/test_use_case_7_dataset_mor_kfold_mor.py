"""
Testing the use case regarding multiple datasets aggregated in mor
and kfold aggregated in mor manner.
"""

import numpy as np

from mlscorecheck.check import (check_multiple_datasets_mor_kfold_mor_scores)
from mlscorecheck.aggregated import generate_problems_with_evaluations
from mlscorecheck.aggregated import calculate_scores_datasets
from mlscorecheck.individual import calculate_scores

from ._score_ranges import (score_ranges,
                            calculate_scores_for_folds,
                            calculate_scores_for_datasets)

k = 4
eps = 10**(-k)

def test_consistent():
    """
    Testing a consistent configuration
    """
    evals, problem = generate_problems_with_evaluations(n_problems=3,
                                                    n_folds=3,
                                                    n_repeats=2,
                                                    random_state=5)

    scores = calculate_scores_datasets(evals,
                                strategy=('mor', 'mor'),
                                rounding_decimals=k)

    flag, details = check_multiple_datasets_mor_kfold_mor_scores(scores,
                                                                eps=eps,
                                                                datasets=problem,
                                                                return_details=True)

    assert flag

    scores_new = calculate_scores_datasets(details['configuration'],
                                    strategy=('mor', 'mor'))

    for key in ['acc', 'sens', 'spec', 'bacc']:
        assert abs(scores[key] - scores_new[key]) <= eps

def test_consistent_differing_configurations():
    """
    Testing a consistent configuration
    """
    evals, problem = generate_problems_with_evaluations(n_problems=3,
                                                    n_folds=3,
                                                    n_repeats=2,
                                                    random_state=5)

    scores = calculate_scores_datasets(evals,
                                strategy=('mor', 'mor'),
                                rounding_decimals=k)

    flag, details = check_multiple_datasets_mor_kfold_mor_scores(scores,
                                                                eps=eps,
                                                                datasets=problem,
                                                                return_details=True)

    assert flag

    scores_new = calculate_scores_datasets(details['configuration'],
                                    strategy=('mor', 'mor'))

    for key in ['acc', 'sens', 'spec', 'bacc']:
        assert abs(scores[key] - scores_new[key]) <= eps

def test_failure():
    """
    Testing a failure
    """
    evals, problem = generate_problems_with_evaluations(n_problems=3,
                                                    n_folds=3,
                                                    n_repeats=2,
                                                    random_state=5)

    scores = calculate_scores_datasets(evals,
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

    evals, problem = generate_problems_with_evaluations(n_problems=3,
                                                    n_folds=3,
                                                    n_repeats=2,
                                                    random_state=5)

    scores = calculate_scores_datasets(evals,
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
    evals, problem = generate_problems_with_evaluations(n_problems=3,
                                                    n_repeats=3,
                                                    n_folds=2,
                                                    random_state=5)

    scores = calculate_scores_datasets(evals,
                                strategy=('mor', 'mor'),
                                rounding_decimals=k)

    bounds = score_ranges(calculate_scores_for_datasets(evals))

    scores = []
    for dataset in evals:
        tmp = []
        for fold in dataset['folds']:
            tmp.append(calculate_scores(fold))
        scores.append(tmp)
    print(scores)

    #for score in bounds:
    #    bounds[score] = (bounds[score][0] - 1e-3, bounds[score][1] + 1e-3)
    #del bounds['bacc']

    for prob in problem:
        prob['score_bounds'] = {**bounds}

    flag, details = check_multiple_datasets_mor_kfold_mor_scores(scores,
                                            eps=eps,
                                            datasets=problem,
                                            return_details=True)

    print(problem)
    print(evals)
    print(details)

    assert flag

    scores_new = calculate_scores_datasets(details['configuration'],
                                    strategy=('mor', 'mor'))

    for key in ['acc', 'sens', 'spec', 'bacc']:
        if key in scores:
            assert abs(scores[key] - scores_new[key]) <= eps

def test_score_bounds_failure():
    """
    Testing the failure with score bounds
    """
    evals, problem = generate_problems_with_evaluations(n_problems=3,
                                                    n_repeats=2,
                                                    n_folds=3,
                                                    random_state=5)

    scores = calculate_scores_datasets(evals,
                                strategy=('mor', 'mor'),
                                rounding_decimals=k)

    bounds = score_ranges(calculate_scores_for_datasets(evals))
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
    problem = [{'folds': [{'p': 5, 'n': 20},
                            {'p': 20, 'n': 5}]},
                {'folds': [{'p': 10, 'n': 5,
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

    problem = [{'folds': [{'p': 5, 'n': 20},
                            {'p': 20, 'n': 5}]},
                {'folds': [{'p': 10, 'n': 5,
                            'p': 20, 'n': 40}]}]

    scores = {'acc': 0.8,
                'sens': 0.79,
                'spec': 0.81}

    flag = check_multiple_datasets_mor_kfold_mor_scores(scores,
                                                eps=0.00001,
                                                datasets=problem,
                                                return_details=False)

    assert not flag

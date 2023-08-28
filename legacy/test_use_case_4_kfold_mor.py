"""
Testing the use case regarding kfold on one dataset in mor manner
"""

from mlscorecheck.check import (check_kfold_mor_scores)
from mlscorecheck.aggregated import generate_1_problem_with_evaluations
from mlscorecheck.aggregated import calculate_scores_dataset
from ._score_ranges import (calculate_scores_for_folds,
                            score_ranges)

k = 4
eps = 10**(-k)

def test_consistent():
    """
    Testing a consistent configuration
    """
    evals, problem = generate_1_problem_with_evaluations(n_repeats=2,
                                                    n_folds=5,
                                                    random_state=5)

    scores = calculate_scores_dataset(evals,
                                strategy='mor',
                                rounding_decimals=k)

    flag, details = check_kfold_mor_scores(scores,
                                            eps=eps,
                                            dataset=problem,
                                            return_details=True)

    assert flag

    scores_new = calculate_scores_dataset(details['configuration'],
                                    strategy='mor')

    for key in ['acc', 'sens', 'spec', 'bacc']:
        assert abs(scores[key] - scores_new[key]) <= eps

def test_failure():
    """
    Testing a failure
    """
    evals, problem = generate_1_problem_with_evaluations(n_repeats=2,
                                                    n_folds=5,
                                                    random_state=5)

    scores = calculate_scores_dataset(evals,
                                strategy='mor',
                                rounding_decimals=k)

    scores['bacc'] = 0.9

    flag, details = check_kfold_mor_scores(scores,
                                            eps=eps,
                                            dataset=problem,
                                            return_details=True)

    assert not flag

def test_dataset_consistency():
    """
    Testing consistency with a dataset
    """

    problem = {'dataset': 'common_datasets.ecoli1', 'n_repeats': 5, 'n_folds': 3}

    scores = {'acc': 0.9,
                'sens': 0.89,
                'spec': 0.91}

    flag, details = check_kfold_mor_scores(scores,
                                            eps=0.01,
                                            dataset=problem,
                                            return_details=True)

    assert flag

def test_dataset_failure():
    """
    Testing failure with a dataset
    """

    problem = {'dataset': 'common_datasets.ecoli1', 'n_repeats': 5, 'n_folds': 3}

    scores = {'acc': 0.9,
                'sens': 0.89,
                'spec': 0.91}

    flag, details = check_kfold_mor_scores(scores,
                                            eps=0.00001,
                                            dataset=problem,
                                            return_details=True)

    assert not flag

def test_score_bounds_consistency():
    """
    Testing the consistency with score bounds
    """
    evals, problem = generate_1_problem_with_evaluations(n_repeats=2,
                                                    n_folds=5,
                                                    random_state=5)

    scores = calculate_scores_dataset(evals,
                                strategy='mor',
                                rounding_decimals=k)

    bounds = score_ranges(calculate_scores_for_folds(evals['folds']))

    problem['score_bounds'] = {key: value for key, value in bounds.items() if key in ['acc', 'sens', 'spec', 'bacc']}

    flag, details = check_kfold_mor_scores(scores,
                                            eps=eps,
                                            dataset=problem,
                                            return_details=True)

    assert flag

    scores_new = calculate_scores_dataset(details['configuration'],
                                    strategy='mor')

    for key in ['acc', 'sens', 'spec', 'bacc']:
        assert abs(scores[key] - scores_new[key]) <= eps

def test_score_bounds_failure():
    """
    Testing the failure with score bounds
    """
    evals, problem = generate_1_problem_with_evaluations(n_repeats=2,
                                                    n_folds=5,
                                                    random_state=5)

    scores = calculate_scores_dataset(evals,
                                strategy='mor',
                                rounding_decimals=k)

    bounds = score_ranges(calculate_scores_for_folds(evals['folds']))
    bounds = {key: value for key, value in bounds.items() if key in ['acc', 'sens', 'spec', 'bacc']}
    bounds['acc'] = (0.99, 1.0)

    problem['score_bounds'] = bounds

    flag, details = check_kfold_mor_scores(scores,
                                            eps=eps,
                                            dataset=problem,
                                            return_details=True)

    assert not flag

def test_tptn_bounds_consistency():
    """
    Testing the consistency with tptn bounds
    """
    evals, problem = generate_1_problem_with_evaluations(n_repeats=2,
                                                    n_folds=5,
                                                    random_state=5)

    scores = calculate_scores_dataset(evals,
                                strategy='mor',
                                rounding_decimals=k)

    bounds = score_ranges(calculate_scores_for_folds(evals['folds']))

    problem['fold_tptn_bounds'] = {'tp': bounds['tp'],
                                'tn': bounds['tn']}

    flag, details = check_kfold_mor_scores(scores,
                                            eps=eps,
                                            dataset=problem,
                                            return_details=True)

    assert flag

    scores_new = calculate_scores_dataset(details['configuration'],
                                    strategy='mor')

    for key in ['acc', 'sens', 'spec', 'bacc']:
        assert abs(scores[key] - scores_new[key]) <= eps

def test_tptn_bounds_failure():
    """
    Testing the failure with tptn bounds
    """
    evals, problem = generate_1_problem_with_evaluations(n_repeats=2,
                                                    n_folds=5,
                                                    random_state=5)

    scores = calculate_scores_dataset(evals,
                                strategy='mor',
                                rounding_decimals=k)

    problem['fold_tptn_bounds'] = {'tp': (int(problem['p']*0.95), problem['p']),
                                'tn': (int(problem['n']*0.95), problem['n'])}

    flag, details = check_kfold_mor_scores(scores,
                                            eps=eps,
                                            dataset=problem,
                                            return_details=True)

    print(problem)
    print(details['configuration'])
    print(details)

    assert not flag

def test_fold_configurations_success():
    """
    Testing the success with manual fold configurations
    """
    problem = {'p': 10,
                'n': 20,
                'folds': [{'p': 4, 'n': 16},
                                        {'p': 6, 'n': 4},
                                        {'p': 5, 'n': 15},
                                        {'p': 5, 'n': 5}]}

    scores = {'acc': 0.8,
                'sens': 0.79,
                'spec': 0.81}

    flag, details = check_kfold_mor_scores(scores,
                                            eps=0.01,
                                            dataset=problem,
                                            return_details=True)

    assert flag

def test_fold_configurations_failure():
    """
    Testing the failure with manual fold configuration
    """
    problem = {'p': 10,
                'n': 20,
                'folds': [{'p': 4, 'n': 16},
                            {'p': 6, 'n': 4},
                            {'p': 5, 'n': 15},
                            {'p': 5, 'n': 5}]}

    scores = {'acc': 0.8,
                'sens': 0.79,
                'spec': 0.81}

    flag, details = check_kfold_mor_scores(scores,
                                            eps=0.00001,
                                            dataset=problem,
                                            return_details=True)

    assert not flag

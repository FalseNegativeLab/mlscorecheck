"""
Testing the use case regarding kfold on one dataset in mor manner
"""

from mlscorecheck.check import (check_kfold_mor_scores)
from mlscorecheck.utils import (generate_problems_with_folds,
                                calculate_scores)
from ._score_ranges import (calculate_scores_for_folds,
                            score_ranges)

k = 4
eps = 10**(-k)

def test_consistent():
    """
    Testing a consistent configuration
    """
    folds, problem = generate_problems_with_folds(n_repeats=2,
                                                    n_folds=5,
                                                    random_seed=5)

    scores = calculate_scores(folds,
                                strategy='mor',
                                rounding_decimals=k)

    flag, details = check_kfold_mor_scores(scores,
                                            eps=eps,
                                            dataset=problem,
                                            return_details=True)

    assert flag

    scores_new = calculate_scores(details['configuration'],
                                    strategy='mor')

    for key in scores_new:
        assert abs(scores[key] - scores_new[key]) <= eps

def test_failure():
    """
    Testing a failure
    """
    folds, problem = generate_problems_with_folds(n_repeats=2,
                                                    n_folds=5,
                                                    random_seed=5)

    scores = calculate_scores(folds,
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
    folds, problem = generate_problems_with_folds(n_repeats=2,
                                                    n_folds=5,
                                                    random_seed=5)

    scores = calculate_scores(folds,
                                strategy='mor',
                                rounding_decimals=k)

    bounds = score_ranges(calculate_scores_for_folds(folds))

    problem['score_bounds'] = bounds

    flag, details = check_kfold_mor_scores(scores,
                                            eps=eps,
                                            dataset=problem,
                                            return_details=True)

    assert flag

    scores_new = calculate_scores(details['configuration'],
                                    strategy='mor')

    for key in scores_new:
        assert abs(scores[key] - scores_new[key]) <= eps

def test_score_bounds_failure():
    """
    Testing the failure with score bounds
    """
    folds, problem = generate_problems_with_folds(n_repeats=2,
                                                    n_folds=5,
                                                    random_seed=5)

    scores = calculate_scores(folds,
                                strategy='mor',
                                rounding_decimals=k)

    bounds = score_ranges(calculate_scores_for_folds(folds))
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
    folds, problem = generate_problems_with_folds(n_repeats=2,
                                                    n_folds=5,
                                                    random_seed=5)

    scores = calculate_scores(folds,
                                strategy='mor',
                                rounding_decimals=k)

    bounds = score_ranges(calculate_scores_for_folds(folds))

    problem['tptn_bounds'] = {'tp': bounds['tp'],
                                'tn': bounds['tn']}

    flag, details = check_kfold_mor_scores(scores,
                                            eps=eps,
                                            dataset=problem,
                                            return_details=True)

    assert flag

    scores_new = calculate_scores(details['configuration'],
                                    strategy='mor')

    for key in scores_new:
        assert abs(scores[key] - scores_new[key]) <= eps

def test_tptn_bounds_failure():
    """
    Testing the failure with tptn bounds
    """
    folds, problem = generate_problems_with_folds(n_repeats=2,
                                                    n_folds=5,
                                                    random_seed=5)

    scores = calculate_scores(folds,
                                strategy='mor',
                                rounding_decimals=k)

    problem['tptn_bounds'] = {'tp': (int(problem['p']*0.9), problem['p']),
                                'tn': (int(problem['n']*0.9), problem['n'])}

    flag, details = check_kfold_mor_scores(scores,
                                            eps=eps,
                                            dataset=problem,
                                            return_details=True)

    assert not flag

def test_fold_configurations_success():
    """
    Testing the success with manual fold configurations
    """
    problem = {'p': 10,
                'n': 20,
                'fold_configuration': [{'p': 4, 'n': 16},
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
                'fold_configuration': [{'p': 4, 'n': 16},
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

"""
Testing the use case regarding multiple datasets in mor manner
"""

from mlscorecheck.check import (check_multiple_datasets_mor_scores)
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
    folds, problem = generate_problems_with_folds(n_problems=3,
                                                    n_folds=1,
                                                    n_repeats=1,
                                                    random_seed=5)

    scores = calculate_scores(folds,
                                strategy='mor',
                                rounding_decimals=k)

    flag, details = check_multiple_datasets_mor_scores(scores,
                                                        eps=eps,
                                                        datasets=problem,
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
    folds, problem = generate_problems_with_folds(n_problems=3,
                                                    n_repeats=1,
                                                    n_folds=1,
                                                    random_seed=5)

    scores = calculate_scores(folds,
                                strategy='mor',
                                rounding_decimals=k)

    scores['bacc'] = 0.9

    flag, details = check_multiple_datasets_mor_scores(scores,
                                            eps=eps,
                                            datasets=problem,
                                            return_details=True)

    assert not flag

def test_dataset_consistency():
    """
    Testing consistency with a dataset
    """

    problem = [{'dataset': 'common_datasets.ecoli1'},
                {'dataset': 'common_datasets.ADA'}]

    scores = {'acc': 0.9,
                'sens': 0.89,
                'spec': 0.91}

    flag, details = check_multiple_datasets_mor_scores(scores,
                                            eps=0.01,
                                            datasets=problem,
                                            return_details=True)

    assert flag

def test_dataset_failure():
    """
    Testing failure with a dataset
    """

    problem = [{'dataset': 'common_datasets.ecoli1'},
                {'dataset': 'common_datasets.ADA'}]

    scores = {'acc': 0.9,
                'sens': 0.89,
                'spec': 0.91}

    flag, details = check_multiple_datasets_mor_scores(scores,
                                            eps=0.00001,
                                            datasets=problem,
                                            return_details=True)

    assert not flag

def test_score_bounds_consistency():
    """
    Testing the consistency with score bounds
    """
    folds, problem = generate_problems_with_folds(n_problems=3,
                                                    n_repeats=1,
                                                    n_folds=1,
                                                    random_seed=5)

    scores = calculate_scores(folds,
                                strategy='mor',
                                rounding_decimals=k)

    bounds = score_ranges(calculate_scores_for_folds(folds))

    for prob in problem:
        prob['score_bounds'] = bounds

    flag, details = check_multiple_datasets_mor_scores(scores,
                                            eps=eps,
                                            datasets=problem,
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
    folds, problem = generate_problems_with_folds(n_problems=3,
                                                    n_repeats=1,
                                                    n_folds=1,
                                                    random_seed=5)

    scores = calculate_scores(folds,
                                strategy='mor',
                                rounding_decimals=k)

    bounds = score_ranges(calculate_scores_for_folds(folds))
    bounds['acc'] = (0.99, 1.0)

    for prob in problem:
        prob['score_bounds'] = bounds

    flag, details = check_multiple_datasets_mor_scores(scores,
                                            eps=eps,
                                            datasets=problem,
                                            return_details=True)

    assert not flag

def test_tptn_bounds_consistency():
    """
    Testing the consistency with tptn bounds
    """
    folds, problem = generate_problems_with_folds(n_problems=3,
                                                    n_repeats=1,
                                                    n_folds=1,
                                                    random_seed=5)

    scores = calculate_scores(folds,
                                strategy='mor',
                                rounding_decimals=k)

    bounds = score_ranges(calculate_scores_for_folds(folds))

    for prob in problem:
        prob['tptn_bounds'] = {'tp': bounds['tp'],
                                'tn': bounds['tn']}

    flag, details = check_multiple_datasets_mor_scores(scores,
                                            eps=eps,
                                            datasets=problem,
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
    folds, problem = generate_problems_with_folds(n_problems=3,
                                                    n_repeats=1,
                                                    n_folds=1,
                                                    random_seed=5)

    scores = calculate_scores(folds,
                                strategy='mor',
                                rounding_decimals=k)

    for prob in problem:
        prob['tptn_bounds'] = {'tp': (int(prob['p']*0.9), prob['p']),
                                'tn': (int(prob['n']*0.9), prob['n'])}

    flag, details = check_multiple_datasets_mor_scores(scores,
                                            eps=eps,
                                            datasets=problem,
                                            return_details=True)

    assert not flag

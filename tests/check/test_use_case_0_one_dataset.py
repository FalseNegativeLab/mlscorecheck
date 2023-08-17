"""
Testing the use case regarding one dataset
"""

from mlscorecheck.check import (check_scores)
from mlscorecheck.utils import (generate_problems_with_folds,
                                calculate_scores)

k = 4
eps = 10**(-k)

def test_consistent():
    """
    Testing a consistent configuration
    """
    folds, problem = generate_problems_with_folds(n_folds=1,
                                                    n_repeats=1,
                                                    random_seed=5)

    scores = calculate_scores(folds,
                                rounding_decimals=k)

    flag, details = check_scores(scores,
                                eps=eps,
                                dataset=problem,
                                return_details=True)

    assert flag

def test_failure():
    """
    Testing a failure
    """
    folds, problem = generate_problems_with_folds(n_folds=1,
                                                    n_repeats=1,
                                                    random_seed=5)

    scores = calculate_scores(folds,
                                rounding_decimals=k)
    scores['bacc'] = 0.9

    flag, details = check_scores(scores,
                                eps=eps,
                                dataset=problem,
                                return_details=True)

    assert not flag

def test_dataset():
    """
    Testing success with real dataset
    """
    problem = {'dataset': 'common_datasets.ADA'}

    scores = {'acc': 0.9,
                'sens': 0.89,
                'spec': 0.91}

    flag = check_scores(scores,
                        eps=0.01,
                        dataset=problem,
                        return_details=False)

    assert flag

def test_dataset_failure():
    """
    Testing failure with real dataset with extreme precision and random scores
    """
    problem = {'dataset': 'common_datasets.ADA'}

    scores = {'acc': 0.9,
                'sens': 0.89,
                'spec': 0.91}

    flag = check_scores(scores,
                        eps=0.00001,
                        dataset=problem,
                        return_details=False)

    assert not flag

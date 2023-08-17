"""
Testing the use case regarding multiple datasets
"""

from mlscorecheck.check import (check_multiple_datasets_rom_scores)
from mlscorecheck.utils import (generate_problems_with_folds,
                                calculate_scores)

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
                                strategy='rom',
                                rounding_decimals=k)

    flag, details = check_multiple_datasets_rom_scores(scores,
                                                        eps=eps,
                                                        datasets=problem,
                                                        return_details=True)

    assert flag

def test_failure():
    """
    Testing a failure
    """
    folds, problem = generate_problems_with_folds(n_problems=3,
                                                    n_folds=1,
                                                    n_repeats=1,
                                                    random_seed=5)

    scores = calculate_scores(folds,
                                strategy='rom',
                                rounding_decimals=k)
    scores['bacc'] = 0.9

    flag, details = check_multiple_datasets_rom_scores(scores,
                                                        eps=eps,
                                                        datasets=problem,
                                                        return_details=True)

    assert not flag

def test_dataset():
    """
    Testing success with real dataset
    """
    problem = [{'dataset': 'common_datasets.ADA'},
                {'dataset': 'common_datasets.ecoli1'}]

    scores = {'acc': 0.9,
                'sens': 0.89,
                'spec': 0.91}

    flag = check_multiple_datasets_rom_scores(scores,
                                                eps=0.01,
                                                datasets=problem,
                                                return_details=False)

    assert flag

def test_dataset_failure():
    """
    Testing failure with real dataset with extreme precision and random scores
    """
    problem = [{'dataset': 'common_datasets.ADA'},
                {'dataset': 'common_datasets.ecoli1'}]

    scores = {'acc': 0.9,
                'sens': 0.89,
                'spec': 0.91}

    flag = check_multiple_datasets_rom_scores(scores,
                                                eps=0.00001,
                                                datasets=problem,
                                                return_details=False)

    assert not flag

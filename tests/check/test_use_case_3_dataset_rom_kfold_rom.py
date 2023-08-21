"""
Testing the use case regarding kfold on multiple datasets
"""

import numpy as np

from mlscorecheck.check import (check_multiple_datasets_rom_kfold_rom_scores)
from mlscorecheck.aggregated import generate_problems_with_evaluations
from mlscorecheck.aggregated import calculate_scores_datasets

k = 4
eps = 10**(-k)

def test_consistent():
    """
    Testing a consistent configuration
    """
    evals, problems = generate_problems_with_evaluations(n_problems=3,
                                                    n_folds=3,
                                                    n_repeats=2,
                                                    random_state=5)

    scores = calculate_scores_datasets(evals,
                                strategy=('rom', 'rom'),
                                rounding_decimals=k)

    flag, details = check_multiple_datasets_rom_kfold_rom_scores(scores,
                                                                eps=eps,
                                                                datasets=problems,
                                                                return_details=True)

    assert flag

def test_consistent_differing_configurations():
    """
    Testing a consistent configuration
    """
    evals, problems = generate_problems_with_evaluations(n_problems=5,
                                                n_folds=5,
                                                n_repeats=3,
                                                random_state=5)

    scores = calculate_scores_datasets(evals,
                                strategy=('rom', 'rom'),
                                rounding_decimals=k)

    flag, details = check_multiple_datasets_rom_kfold_rom_scores(scores,
                                                                eps=eps,
                                                                datasets=problems,
                                                                return_details=True)

    assert flag

def test_failure():
    """
    Testing a failure
    """
    evals, problems = generate_problems_with_evaluations(n_problems=3,
                                                    n_folds=3,
                                                    n_repeats=2,
                                                    random_state=5)

    scores = calculate_scores_datasets(evals,
                                strategy=('rom', 'rom'),
                                rounding_decimals=k)
    scores['bacc'] = 0.9

    flag, details = check_multiple_datasets_rom_kfold_rom_scores(scores,
                                                        eps=eps,
                                                        datasets=problems,
                                                        return_details=True)

    assert not flag

def test_failure_differing_configurations():
    """
    Testing a consistent configuration
    """
    evals, problems = generate_problems_with_evaluations(n_problems=5,
                                                n_folds=5,
                                                n_repeats=3,
                                                random_state=5)

    scores = calculate_scores_datasets(evals,
                                strategy=('rom', 'rom'),
                                rounding_decimals=k)

    scores['bacc'] = 0.9

    flag, details = check_multiple_datasets_rom_kfold_rom_scores(scores,
                                                                eps=eps,
                                                                datasets=problems,
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

    flag = check_multiple_datasets_rom_kfold_rom_scores(scores,
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

    flag = check_multiple_datasets_rom_kfold_rom_scores(scores,
                                                eps=0.00001,
                                                datasets=problem,
                                                return_details=False)

    assert not flag

"""
Testing the use case regarding one dataset
"""

from mlscorecheck.check import (check_scores)
from mlscorecheck.aggregated import (generate_problems_with_evaluations)
from mlscorecheck.individual import calculate_scores, generate_1_problem

k = 4
eps = 10**(-k)

def test_consistent():
    """
    Testing a consistent configuration
    """
    evaluation, problem = generate_1_problem(random_state=5)

    scores = calculate_scores(evaluation,
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
    evaluation, problem = generate_1_problem(random_state=5)

    scores = calculate_scores(evaluation,
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

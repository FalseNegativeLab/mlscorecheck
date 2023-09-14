"""
Testing the checking of scores for 1 dataset with no assumption on the aggregation
or the fold structure.
"""

import pytest

import numpy as np

from mlscorecheck.check import check_1_dataset_kfold_scores
from mlscorecheck.aggregated import generate_evaluation, generate_dataset, Evaluation

def generate_evaluation_unknown(random_seed, rounding_decimals):
    random_state = np.random.RandomState(random_seed)

    if random_state.randint(2) == 0:
        dataset = {'n': random_state.randint(1, 200),
                    'p': random_state.randint(1, 25)}
        folding = {'n_folds': random_state.randint(1, 6),
                    'n_repeats': random_state.randint(1, 3),
                    'strategy': 'stratified_sklearn'}
        folding['n_folds'] = min(folding['n_folds'], dataset['p'], dataset['n'])

        evaluation = Evaluation(dataset=dataset,
                                folding=folding,
                                aggregation='mor')
        evaluation.sample_figures()

        scores = evaluation.calculate_scores(rounding_decimals)

        evaluation = evaluation.to_dict()

        if random_state.randint(2) == 0:
            del evaluation['folding']['strategy']

        return evaluation, scores

    return generate_evaluation(random_state=random_seed,
                                rounding_decimals=rounding_decimals,
                                return_scores=True)

@pytest.mark.parametrize('random_seed', list(range(10)))
@pytest.mark.parametrize('rounding_decimals', [4])
def test_success(random_seed, rounding_decimals):
    """
    Testing with successful configuration
    """
    evaluation, scores = generate_evaluation_unknown(random_seed, rounding_decimals)

    del evaluation['aggregation']

    results = check_1_dataset_kfold_scores(scores=scores,
                                            eps=(10**(-rounding_decimals)),
                                            evaluation=evaluation)

    assert not results['inconsistency']

@pytest.mark.parametrize('random_seed', list(range(10)))
@pytest.mark.parametrize('rounding_decimals', [4])
def test_failure(random_seed, rounding_decimals):
    """
    Testing with successful configuration
    """
    evaluation, scores = generate_evaluation_unknown(random_seed, rounding_decimals)

    del evaluation['aggregation']

    results = check_1_dataset_kfold_scores(scores={'acc': 0.9, 'sens': 0.3, 'spec': 0.3, 'bacc': 0.2},
                                            eps=(10**(-rounding_decimals)),
                                            evaluation=evaluation)

    assert results['inconsistency']

def test_exception():
    """
    Testing if the exception is thrown
    """

    with pytest.raises(ValueError):
        check_1_dataset_kfold_scores(evaluation={'aggregation': 'rom'},
                                        eps=1e-4,
                                        scores={})

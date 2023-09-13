"""
Testing the checking of scores on multiple datasets using kfold with
ratio-of-means aggregation over the folds and mean-of-ratios aggregation
over the datasets
"""

import pytest

from mlscorecheck.check import check_n_datasets_mor_kfold_rom_scores
from mlscorecheck.aggregated import (generate_experiment)

@pytest.mark.parametrize('random_seed', list(range(10)))
@pytest.mark.parametrize('rounding_decimals', [3, 4])
def test_consistency(random_seed, rounding_decimals):
    """
    Testing with a consistent setup
    """

    experiment, scores = generate_experiment(aggregation='mor',
                                                aggregation_folds='rom',
                                                random_state=random_seed,
                                                rounding_decimals=rounding_decimals,
                                                return_scores=True)

    result = check_n_datasets_mor_kfold_rom_scores(experiment=experiment,
                                                    scores=scores,
                                                    eps=10**(-rounding_decimals))

    assert not result['inconsistency']

@pytest.mark.parametrize('random_seed', list(range(10)))
@pytest.mark.parametrize('rounding_decimals', [3, 4])
def test_failure(random_seed, rounding_decimals):
    """
    Testing with an inconsistent setup
    """
    experiment, scores = generate_experiment(aggregation='mor',
                                                aggregation_folds='rom',
                                                random_state=random_seed,
                                                rounding_decimals=rounding_decimals,
                                                return_scores=True)

    scores = {'acc': 0.9, 'sens': 0.3, 'spec': 0.5, 'f1': 0.1}

    result = check_n_datasets_mor_kfold_rom_scores(experiment=experiment,
                                                scores=scores,
                                                eps=10**(-rounding_decimals))

    assert result['inconsistency']

@pytest.mark.parametrize('random_seed', list(range(10)))
@pytest.mark.parametrize('rounding_decimals', [3, 4])
def test_consistency_bounds(random_seed, rounding_decimals):
    """
    Testing with a consistent setup and bounds
    """
    experiment, scores = generate_experiment(aggregation='mor',
                                                aggregation_folds='rom',
                                                random_state=random_seed,
                                                rounding_decimals=rounding_decimals,
                                                return_scores=True,
                                                feasible_dataset_score_bounds=True)

    result = check_n_datasets_mor_kfold_rom_scores(experiment=experiment,
                                                scores=scores,
                                                eps=10**(-rounding_decimals),
                                                timeout=2)

    assert not result['inconsistency'] or result['lp_status'] == 'timeout'

@pytest.mark.parametrize('random_seed', list(range(10)))
@pytest.mark.parametrize('rounding_decimals', [3, 4])
def test_failure_bounds(random_seed, rounding_decimals):
    """
    Testing with a inconsistent setup and bounds
    """
    experiment, scores = generate_experiment(aggregation='mor',
                                                aggregation_folds='rom',
                                                random_state=random_seed,
                                                rounding_decimals=rounding_decimals,
                                                return_scores=True,
                                                feasible_dataset_score_bounds=False)

    result = check_n_datasets_mor_kfold_rom_scores(experiment=experiment,
                                                scores=scores,
                                                eps=10**(-rounding_decimals),
                                                timeout=2)
    print(result)

    assert result['inconsistency'] or result['lp_status'] == 'timeout'

def test_exception():
    """
    Testing the throwing of an exception
    """

    with pytest.raises(ValueError):
        check_n_datasets_mor_kfold_rom_scores(experiment={'aggregation': 'rom',
                                                            'evaluations': []},
                                                scores={},
                                                eps=1e-4)

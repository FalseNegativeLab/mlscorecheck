"""
Testing the checking of scores on multiple datasets using kfold with
ratio-of-means aggregation on both levels
"""

import pytest

from mlscorecheck.check import check_n_datasets_rom_kfold_rom_scores
from mlscorecheck.aggregated import (generate_experiment)

@pytest.mark.parametrize('random_seed', list(range(10)))
@pytest.mark.parametrize('rounding_decimals', [3, 4])
def test_consistency(random_seed, rounding_decimals):
    """
    Testing with a consistent setup
    """

    experiment, scores = generate_experiment(aggregation='rom',
                                            evaluation_params={'aggregation': 'rom'},
                                            random_state=random_seed,
                                            return_scores=True,
                                            rounding_decimals=rounding_decimals)

    result = check_n_datasets_rom_kfold_rom_scores(experiment=experiment,
                                                    scores=scores,
                                                    eps=10**(-rounding_decimals))

    assert not result['inconsistency']

@pytest.mark.parametrize('random_seed', list(range(10)))
@pytest.mark.parametrize('rounding_decimals', [3, 4])
def test_failure(random_seed, rounding_decimals):
    """
    Testing with an inconsistent setup
    """

    experiment, scores = generate_experiment(aggregation='rom',
                                            evaluation_params={'aggregation': 'rom'},
                                            random_state=random_seed,
                                            return_scores=True,
                                            rounding_decimals=rounding_decimals)

    scores = {'acc': 0.9, 'sens': 0.3, 'spec': 0.5, 'f1': 0.1}

    result = check_n_datasets_rom_kfold_rom_scores(experiment=experiment,
                                                    scores=scores,
                                                    eps=10**(-rounding_decimals))

    assert result['inconsistency']

def test_exception():
    """
    Testing the throwing of an exception
    """

    with pytest.raises(ValueError):
        check_n_datasets_rom_kfold_rom_scores(experiment={'aggregation': 'mor',
                                                            'evaluations': []},
                                                scores={},
                                                eps=1e-4)

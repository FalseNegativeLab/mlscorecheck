"""
This file tests the test bundle for the CHASEDB1 dataset
"""

import pytest

from mlscorecheck.check.bundles.retina import (check_chasedb1_vessel_image,
                                            check_chasedb1_vessel_aggregated)

from mlscorecheck.experiments import get_experiment

from mlscorecheck.aggregated import generate_scores_for_testsets

@pytest.mark.parametrize('random_state', [1, 2, 3, 4, 5])
def test_success_mos(random_state):
    """
    Testing a consistent setup with MoS aggregation

    Args:
        random_state (int): the random seed to use
    """

    data = get_experiment('retina.chase_db1')['manual1']['images']

    scores = generate_scores_for_testsets(data,
                                            aggregation='mos',
                                            rounding_decimals=4,
                                            random_state=random_state)
    results = check_chasedb1_vessel_aggregated(imageset='all',
                                        annotator='manual1',
                                        scores=scores,
                                        eps=1e-4,
                                        verbosity=0)

    assert not results['inconsistency']['inconsistency_mos']

    results = check_chasedb1_vessel_aggregated(imageset=[img['identifier'] for img in data],
                                        annotator='manual1',
                                        scores=scores,
                                        eps=1e-4,
                                        verbosity=0)

    assert not results['inconsistency']['inconsistency_mos']

@pytest.mark.parametrize('random_state', [1, 2, 3, 4, 5])
def test_failure_mos(random_state):
    """
    Testing an inconsistent setup with MoS aggregation

    Args:
        random_state (int): the random seed to use
    """

    data = get_experiment('retina.chase_db1')['manual1']['images']

    scores = generate_scores_for_testsets(data,
                                            aggregation='mos',
                                            rounding_decimals=4,
                                            random_state=random_state)
    scores['acc'] = (1.0 + scores['spec'])/2.0

    results = check_chasedb1_vessel_aggregated(imageset='all',
                                        annotator='manual1',
                                        scores=scores,
                                        eps=1e-4,
                                        verbosity=0)

    assert results['inconsistency']['inconsistency_mos']

    results = check_chasedb1_vessel_aggregated(imageset=[img['identifier'] for img in data],
                                        annotator='manual1',
                                        scores=scores,
                                        eps=1e-4,
                                        verbosity=0)

    assert results['inconsistency']['inconsistency_mos']

@pytest.mark.parametrize('random_state', [1, 2, 3, 4, 5])
def test_success_som(random_state):
    """
    Testing a consistent setup with SoM aggregation

    Args:
        random_state (int): the random seed to use
    """

    data = get_experiment('retina.chase_db1')['manual1']['images']

    scores = generate_scores_for_testsets(data,
                                            aggregation='som',
                                            rounding_decimals=4,
                                            random_state=random_state)

    results = check_chasedb1_vessel_aggregated(imageset='all',
                                        annotator='manual1',
                                        scores=scores,
                                        eps=1e-4)

    assert not results['inconsistency']['inconsistency_som']

    results = check_chasedb1_vessel_aggregated(imageset=[img['identifier'] for img in data],
                                        annotator='manual1',
                                        scores=scores,
                                        eps=1e-4)

    assert not results['inconsistency']['inconsistency_som']

@pytest.mark.parametrize('random_state', [1, 2, 3, 4, 5])
def test_failure_som(random_state):
    """
    Testing an inconsistent setup with SoM aggregation

    Args:
        random_state (int): the random seed to use
    """

    data = get_experiment('retina.chase_db1')['manual1']['images']

    scores = generate_scores_for_testsets(data,
                                            aggregation='som',
                                            rounding_decimals=4,
                                            random_state=random_state)

    scores['acc'] = (1.0 + scores['spec'])/2.0

    results = check_chasedb1_vessel_aggregated(imageset='all',
                                            annotator='manual1',
                                            scores=scores,
                                            eps=1e-4)

    assert results['inconsistency']['inconsistency_som']

    results = check_chasedb1_vessel_aggregated(imageset=[img['identifier'] for img in data],
                                            annotator='manual1',
                                            scores=scores,
                                            eps=1e-4)

    assert results['inconsistency']['inconsistency_som']

@pytest.mark.parametrize('random_state', [1, 2, 3, 4, 5])
def test_success_image(random_state):
    """
    Testing a consistent setup for an image

    Args:
        random_state (int): the random seed to use
    """

    data = get_experiment('retina.chase_db1')['manual1']['images']

    scores = generate_scores_for_testsets([data[0]],
                                            aggregation='som',
                                            rounding_decimals=4,
                                            random_state=random_state)

    results = check_chasedb1_vessel_image(image_identifier=data[0]['identifier'],
                                        annotator='manual1',
                                        scores=scores,
                                        eps=1e-4)

    assert not results['inconsistency']

@pytest.mark.parametrize('random_state', [1, 2, 3, 4, 5])
def test_failure_image(random_state):
    """
    Testing an inconsistent setup for an image

    Args:
        random_state (int): the random seed to use
    """
    data = get_experiment('retina.chase_db1')['manual1']['images']

    scores = generate_scores_for_testsets([data[0]],
                                            aggregation='som',
                                            rounding_decimals=4,
                                            random_state=random_state)

    scores['acc'] = (1.0 + scores['spec'])/2.0

    results = check_chasedb1_vessel_image(image_identifier=data[0]['identifier'],
                                        annotator='manual1',
                                        scores=scores,
                                        eps=1e-4)

    assert results['inconsistency']

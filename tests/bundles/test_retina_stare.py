"""
This file tests the test bundle for the STARE dataset
"""

import pytest

from mlscorecheck.check.bundles.retina import (check_stare_vessel_image,
                                            check_stare_vessel_aggregated)

from mlscorecheck.experiments import get_experiment

from mlscorecheck.aggregated import generate_scores_for_testsets

@pytest.mark.parametrize('random_state', [1, 2, 3, 4, 5])
def test_success_mos(random_state):
    """
    Testing a consistent setup with MoS aggregation

    Args:
        random_state (int): the random seed to use
    """

    dataah = get_experiment('retina.stare')['ah']['images']

    scores = generate_scores_for_testsets(dataah,
                                            aggregation='mos',
                                            rounding_decimals=4,
                                            random_state=random_state)
    results = check_stare_vessel_aggregated(imageset='all',
                                        annotator='ah',
                                        scores=scores,
                                        eps=1e-4,
                                        verbosity=0)

    assert not results['inconsistency']['inconsistency_mos']

    results = check_stare_vessel_aggregated(imageset=[img['identifier'] for img in dataah],
                                        annotator='ah',
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

    dataah = get_experiment('retina.stare')['ah']['images']

    scores = generate_scores_for_testsets(dataah,
                                            aggregation='mos',
                                            rounding_decimals=4,
                                            random_state=random_state)
    scores['acc'] = (1.0 + scores['spec'])/2.0

    results = check_stare_vessel_aggregated(imageset='all',
                                        annotator='ah',
                                        scores=scores,
                                        eps=1e-4,
                                        verbosity=0)

    assert results['inconsistency']['inconsistency_mos']

    results = check_stare_vessel_aggregated(imageset=[img['identifier'] for img in dataah],
                                        annotator='ah',
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

    dataah = get_experiment('retina.stare')['ah']['images']

    scores = generate_scores_for_testsets(dataah,
                                            aggregation='som',
                                            rounding_decimals=4,
                                            random_state=random_state)

    results = check_stare_vessel_aggregated(imageset='all',
                                        annotator='ah',
                                        scores=scores,
                                        eps=1e-4)

    assert not results['inconsistency']['inconsistency_som']

    results = check_stare_vessel_aggregated(imageset=[img['identifier'] for img in dataah],
                                        annotator='ah',
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

    dataah = get_experiment('retina.stare')['ah']['images']

    scores = generate_scores_for_testsets(dataah,
                                            aggregation='som',
                                            rounding_decimals=4,
                                            random_state=random_state)

    scores['acc'] = (1.0 + scores['spec'])/2.0

    results = check_stare_vessel_aggregated(imageset='all',
                                            annotator='ah',
                                            scores=scores,
                                            eps=1e-4)

    assert results['inconsistency']['inconsistency_som']

    results = check_stare_vessel_aggregated(imageset=[img['identifier'] for img in dataah],
                                            annotator='ah',
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

    dataah = get_experiment('retina.stare')['ah']['images']

    scores = generate_scores_for_testsets([dataah[0]],
                                            aggregation='som',
                                            rounding_decimals=4,
                                            random_state=random_state)

    results = check_stare_vessel_image(image_identifier=dataah[0]['identifier'],
                                        annotator='ah',
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
    dataah = get_experiment('retina.stare')['ah']['images']

    scores = generate_scores_for_testsets([dataah[0]],
                                            aggregation='som',
                                            rounding_decimals=4,
                                            random_state=random_state)

    scores['acc'] = (1.0 + scores['spec'])/2.0

    results = check_stare_vessel_image(image_identifier=dataah[0]['identifier'],
                                        annotator='ah',
                                        scores=scores,
                                        eps=1e-4)

    assert results['inconsistency']

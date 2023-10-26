"""
This module tests the loading of experiments
"""

import pytest

from mlscorecheck.experiments import (load_drive,
                                        load_ehg,
                                        load_stare,
                                        load_isic2016,
                                        load_isic2017,
                                        load_chase_db1,
                                        load_diaretdb0,
                                        load_diaretdb1,
                                        load_hrf,
                                        load_drishti_gs,
                                        get_experiment)

@pytest.mark.parametrize('key', ['retina.drive',
                                    'retina.stare',
                                    'retina.chase_db1',
                                    'retina.diaretdb0',
                                    'retina.diaretdb1',
                                    'retina.hrf',
                                    'retina.drishti_gs',
                                    'ehg.ehg',
                                    'skinlesion.isic2016',
                                    'skinlesion.isic2017'])
def test_get_experiment(key):
    """
    Testing the get experiment function
    """

    # called twice to cover the caching
    assert len(get_experiment(key)) > 0
    assert len(get_experiment(key)) > 0

def test_get_experiment_exception():
    """
    Testing if get_experiment throws the exception
    """

    with pytest.raises(ValueError):
        get_experiment('dummy')

def test_load_drishti_gs():
    """
    Testing the loading of DRISHTI_GS
    """

    assert len(load_drishti_gs()) > 0

def test_load_hrf():
    """
    Testing the loading of HRF
    """

    assert len(load_hrf()) > 0

def test_load_diaretdb1():
    """
    Testing the loading of DIARETDB1
    """

    assert len(load_diaretdb1()) > 0

def test_load_diaretdb0():
    """
    Testing the loading of DIARETDB0
    """

    assert len(load_diaretdb0()) > 0

def test_load_chase_db1():
    """
    Testing the loading of CHASE db 1
    """

    assert len(load_chase_db1()) > 0

def test_load_isic2017():
    """
    Testing the loading of ISIC2017
    """

    assert len(load_isic2017()) > 0

def test_load_isic2016():
    """
    Testing the loading of ISIC2016
    """

    assert len(load_isic2016()) > 0

def test_load_stare():
    """
    Testing the loading of STARE
    """

    assert len(load_stare()) > 0

def test_load_drive():
    """
    Testing the loading of drive
    """

    assert len(load_drive()) == 4

def test_load_ehg():
    """
    Testing the loading of the EHG dataset
    """

    assert 'p' in load_ehg()

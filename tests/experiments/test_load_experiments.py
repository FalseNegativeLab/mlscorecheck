"""
This module tests the loading of experiments
"""

from mlscorecheck.experiments import load_drive, load_ehg

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

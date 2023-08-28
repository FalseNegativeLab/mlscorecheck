"""
This module tests the loading of experiments
"""

from mlscorecheck.experiments import load_drive

def test_load_drive():
    """
    Testing the loading of drive
    """

    assert len(load_drive()) == 4

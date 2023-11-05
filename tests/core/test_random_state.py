"""
This module tests the initialization of random states
"""

import numpy as np

from mlscorecheck.core import init_random_state


def test_init_random_state():
    """
    Testing the init random state functionality
    """
    random_state = np.random.RandomState(5)

    assert init_random_state(random_state) == random_state
    assert init_random_state() != random_state

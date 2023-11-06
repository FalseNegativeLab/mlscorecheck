"""
This module implements some functionalities related to random states
"""

import numpy as np

__all__ = ["init_random_state"]


def init_random_state(random_state=None) -> np.random.RandomState:
    """
    Initializes a random state

    Args:
        random_state (int|np.random.RandomState|None): the random state/seed to initialize

    Returns:
        np.random.RandomState: the initialized random state
    """
    if random_state is None or not isinstance(random_state, np.random.RandomState):
        return np.random.RandomState(random_state)
    return random_state

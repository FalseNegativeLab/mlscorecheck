"""
This module implements the rounding of scores
"""

import numpy as np

__all__ = ["round_scores"]


def round_scores(to_round: dict, rounding_decimals: int = None):
    """
    Rounds the scores

    Args:
        to_round (float|dict): the score(s) to round
        rounding_decimals (int|None): the number of decimal places to round to

    Returns:
        float|dict: the founded scores
    """
    if rounding_decimals is None:
        return {**to_round}

    if not isinstance(to_round, dict):
        return float(np.round(to_round, rounding_decimals))

    return {
        key: float(np.round(value, rounding_decimals))
        for key, value in to_round.items()
    }

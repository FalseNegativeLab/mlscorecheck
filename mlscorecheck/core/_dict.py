"""
This module implements some operations related to dictionaries
"""

import numpy as np

__all__ = ["dict_mean", "dict_minmax"]


def dict_mean(dicts: list) -> dict:
    """
    Calculates the mean of scores in a dictionary

    Args:
        dicts (list(dict)): a list of dictionaries

    Returns:
        dict: the dictionary with the mean figures
    """

    result = {key: 0 for key in dicts[0]}

    for tmp in dicts:
        for key in tmp:
            result[key] += tmp[key]
    for key in result:
        result[key] /= len(dicts)

    return result


def dict_minmax(dicts: list) -> dict:
    """
    Calculates the min-max of scores in a dictionary

    Args:
        dicts (list(dict)): a list of dictionaries

    Returns:
        dict (str,tuple(float,float)): the dictionary with the min-max figures
    """
    result = {key: [np.inf, -np.inf] for key in dicts[0]}

    for tmp in dicts:
        for key in result:
            if tmp[key] < result[key][0]:
                result[key][0] = tmp[key]
            if tmp[key] > result[key][1]:
                result[key][1] = tmp[key]

    return result

"""
This module contains the general numerical tolerance
"""

import numpy as np

__all__ = [
    "NUMERICAL_TOLERANCE",
    "check_uncertainty_and_tolerance",
    "update_uncertainty",
]

NUMERICAL_TOLERANCE = 1e-6


def check_uncertainty_and_tolerance(eps: float, numerical_tolerance: float):
    """
    Checking the relation of the numerical tolerance and the uncertainty

    Args:
        eps (dict|float): the numerical uncertainties
        numerical_tolerance (float): the numberical tolerance

    Raises:
        ValueError: if the uncertainties are not at least an order of magnitude
        greater than the numerical tolerance
    """
    eps_min = np.min(list(eps.values())) if isinstance(eps, dict) else eps
    if eps_min <= 10 * numerical_tolerance:
        raise ValueError(
            "The numerical tolerance is comparable to the numerical "
            "uncertainty. Please lower the numerical tolerance."
        )


def update_uncertainty(eps: float, numerical_tolerance: float):
    """
    Adjusts the specified uncertainty by the numerical tolerance

    Args:
        eps (dict|float): the numerical uncertainties
        numerical_tolerance (float): the numberical tolerance

    Returns:
        dict|float: the adjusted uncertainty
    """
    if isinstance(eps, dict):
        return {key: value + numerical_tolerance for key, value in eps.items()}
    return eps + numerical_tolerance

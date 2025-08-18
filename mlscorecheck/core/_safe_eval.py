"""
This module implements functions for safe evaluations
"""

from typing import Any, Union
from types import FunctionType

__all__ = ["safe_eval", "safe_call"]


def safe_eval(expression: str, subs: dict[str, float]) -> Any:
    """
    Evaluates a str mathematical expression in a safe way
    # TODO: write better

    Args:
        expression (str): the expression to evaluate
        subs (dict): the substitution to use

    Returns:
        obj: the value of the expression
    """

    return eval(expression, subs)  # pylint: disable=eval-used


def check_applicability(
    params: dict[str, float], non_applicable: list[dict[str, str | float]] | None = None
) -> bool:
    """
    Checks if a parameter configuration is applicable according to the
    non-applicability configurations

    Args:
        params (dict): the parameters to evaluate
        non_applicable (list(dict)): a list of parameter configurations
                                    which are considered non-applicable

    Returns:
        bool: True if the parameter configuration in params does not
        match any of the non-applicable configurations, False
        otherwise
    """
    if non_applicable is None:
        return True

    for configuration in non_applicable:
        flag = True
        for key, value in configuration.items():
            if isinstance(value, str):
                actual_value = params.get(value)
            else:
                actual_value = value
            flag = flag and (params.get(key) == actual_value)
        if flag:
            return False
    return True


def safe_call(
    function: Union[FunctionType, Any],
    params: dict[str, float],
    non_applicable: list[dict[str, str | float]] | None = None,
) -> Any:
    """
    Safe call to a function

    Args:
        function (callable): a function to call
        params (dict): the parameters to call the function with
        non_applicable (list(dict)): a list of parameter configurations which
                                    are considered non-applicable

    Returns:
        obj: the result of the function call
    """

    if not check_applicability(params, non_applicable):
        return None

    args = list(function.__code__.co_varnames[: function.__code__.co_kwonlyargcount])

    return function(**{arg: params[arg] for arg in args if arg in params})

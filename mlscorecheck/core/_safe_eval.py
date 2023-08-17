"""
This module implements functions for safe evaluations
"""

__all__ = ['safe_eval',
            'safe_call']

def safe_eval(expression, subs):
    # TODO: write better
    return eval(expression, subs)

def check_applicability_configurations(params, configurations=None):
    if configurations is None:
        return True

    for configuration in configurations:
        flag = True
        for key, value in configuration.items():
            flag = flag and (params.get(key) == value)
        if flag:
            return False
    return True

def safe_call(function, params, configurations=None):
    if not check_applicability_configurations(params, configurations):
        return None

    args = list(function.__code__.co_varnames[:function.__code__.co_kwonlyargcount])
    return function(**{arg: params[arg] for arg in args})

"""
This module implements functions for safe evaluations
"""

__all__ = ['safe_eval',
            'safe_call']

def safe_eval(expression, subs):
    # TODO: write better
    return eval(expression, subs)

def safe_call(function, params):
    args = list(function.__code__.co_varnames[:function.__code__.co_kwonlyargcount])
    return function(**{arg: params[arg] for arg in args})

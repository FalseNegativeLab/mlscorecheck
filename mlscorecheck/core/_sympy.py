"""
This module implements some functionalities related to sympy
"""

__all__ = ['collect_with_respect_to_vars']

def collect_with_respect_to_vars(eq, vars):
    """
    Collect all terms of the specified variables and their coefficients

    Inspired by https://stackoverflow.com/questions/59363674/sympy-how-to-collect-multi-variable-terms

    Args:
        eq (sympy Expression): the expression to collect terms from
        vars (list): the list of variables

    Returns:
        dict: the combinations of variables and their coefficients
    """
    assert isinstance(vars, list)
    eq = eq.expand()

    if len(vars) == 0:
        return {1: eq}

    var_map = eq.collect(vars[0], evaluate=False)
    final_var_map = {}

    for var_power in var_map:
        sub_expression = var_map[var_power]
        sub_var_map = collect_with_respect_to_vars(sub_expression, vars[1:])
        for sub_var_power in sub_var_map:
            final_var_map[var_power*sub_var_power] = sub_var_map[sub_var_power]

    return final_var_map
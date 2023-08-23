"""
Some general purpose linear programming functionalities
"""
import pulp as pl

import numpy as np

from ..core import logger

__all__ = ['add_bounds',
            'solve',
            'create_lp_target',
            'check_bounds']

def check_bounds(scores, bounds):
    if bounds is None:
        return True

    flag = True
    for key in bounds:
        if bounds[key][0] is not None and not np.isnan(bounds[key][0]):
            flag = flag and (bounds[key][0] <= scores[key])
        if bounds[key][1] is not None and not np.isnan(bounds[key][1]):
            flag = flag and (scores[key] <= bounds[key][1])

    return flag

def add_bounds(lp_program, variables, bounds=None):
    if bounds is None:
        return lp_program

    for variable in bounds:
        if variable not in variables:
            raise ValueError("variable {variable} not accessible yet")

        if bounds[variable][0] is not None and not np.isnan(bounds[variable][0]):
            lp_program += bounds[variable][0] <= variable
        else:
            logger.info(f'No upper bound for variable {variable}')
        if bounds[variable][1] is not None and not np.isnan(bounds[variable][1]):
            lp_program += variable <= bounds[variable][1]
        else:
            logger.info(f'No upper bound for variable {variable}')

    return lp_program

def solve(obj, scores, eps):
    lp_program = pl.LpProblem('feasibility')

    lp_program = obj.init_lp(lp_program)

    lp_program = create_lp_target(obj, scores, eps, lp_program)

    lp_program += obj.linear_programming['tp']

    lp_program.solve()

    return lp_program

def create_lp_target(obj, scores, eps, lp_program):
    print(scores)
    print(eps)
    print(obj.linear_programming)
    for key in scores:
        if key in ['acc', 'sens', 'spec', 'bacc']:
            print(scores[key], eps[key], obj.linear_programming[key])
            lp_program += (obj.linear_programming[key] >= (scores[key] - eps[key]))
            lp_program += (obj.linear_programming[key] <= (scores[key] + eps[key]))

    return lp_program
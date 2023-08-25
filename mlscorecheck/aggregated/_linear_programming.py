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
    """
    Checks the bounds for the scores

    Args:
        scores (dict(str,float/int)): a dictionary of scores
        bounds (dict(str,tuple(float/int,float/int))): the dictionary of bounds

    Returns:
        None/bool: None if the bounds are not specified, otherwise a flag
                    if the scores are within the bounds
    """

    if bounds is None:
        return None

    flag = True
    for key in bounds:
        if bounds[key][0] is not None and not np.isnan(bounds[key][0]):
            flag = flag and (bounds[key][0] <= scores[key])
        if bounds[key][1] is not None and not np.isnan(bounds[key][1]):
            flag = flag and (scores[key] <= bounds[key][1])

    return flag

def add_bounds(lp_problem, variables, bounds, label):
    """
    Adding bounds to a linear program

    Args:
        lp_problem (pl.LpProblem): the linear programming problem
        variables (dict(str,pl.LpVariable)): the variables to add bounds to
        bounds (dict(str,tuple(float/int,float/int))): the bounds to add
        label (str): the label of the entity containing the variables
                        bounds are added to

    Returns:
        pl.LpProblem: the adjusted linear programming problem
    """
    if bounds is None:
        return lp_problem

    for variable in bounds:

        if bounds[variable][0] is not None and not np.isnan(bounds[variable][0]):
            logger.info(f'{label}: adding lower bound {bounds[variable][0]} for {variable}')
            lp_problem += bounds[variable][0] <= variables[variable]
        else:
            logger.info(f'{label}: No upper bound for variable {variable}')
        if bounds[variable][1] is not None and not np.isnan(bounds[variable][1]):
            logger.info(f'{label}: adding upper bound {bounds[variable][1]} for {variable}')
            lp_problem += variables[variable] <= bounds[variable][1]
        else:
            logger.info(f'{label}: No upper bound for variable {variable}')

    return lp_problem

def create_lp_target(obj, scores, eps, lp_problem):
    """
    Creates the linear programming target

    Args:
        obj (object): the object to create the target conditions for
        scores (dict(str,float)): the scores to match
        eps (dict(str,float)/float): the numerical uncertainty
        lp_problem (pl.LpProblem): the linear programming problem

    Returns:
        pl.LpProblem: the updated linear programming problem
    """
    for key in scores:
        if key in ['acc', 'sens', 'spec', 'bacc']:
            logger.info(f'{obj.id}: adding condition {key} >= {scores[key] - eps[key]}')
            lp_problem += (obj.linear_programming[key] >= (scores[key] - eps[key]))
            logger.info(f'{obj.id}: adding condition {key} <= {scores[key] + eps[key]}')
            lp_problem += (obj.linear_programming[key] <= (scores[key] + eps[key]))

    return lp_problem


def solve(obj, scores, eps):
    """
    Solving a problem.

    Args:
        obj (object): an object to solve
        scores (dict(str,float)): the scores to match
        eps (dict(str,float)/float): the numerical uncertainty

    Returns:
        pl.LpProblem: the solved linear programming problem
    """
    if not isinstance(eps, dict):
        eps = {key: eps for key in ['acc', 'sens', 'spec', 'bacc']}

    lp_program = pl.LpProblem('feasibility')

    lp_program = obj.init_lp(lp_program)

    lp_program = create_lp_target(obj, scores, eps, lp_program)

    lp_program += obj.linear_programming['tp']

    lp_program.solve()

    return lp_program

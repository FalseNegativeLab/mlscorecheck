"""
Some general purpose linear programming functionalities
"""
import pulp as pl

import numpy as np

from ..core import logger

from ._utils import random_identifier

__all__ = ['add_bounds',
            'solve',
            'create_lp_target']

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
            logger.info('%s: adding lower bound %s for %s',
                        label, str(bounds[variable][0]), variable)
            lp_problem += bounds[variable][0] <= variables[variable]
        else:
            logger.info('%s: No lower bound for variable %s', label, variable)
        if bounds[variable][1] is not None and not np.isnan(bounds[variable][1]):
            logger.info('%s: adding upper bound %s for %s', label, bounds[variable][1], variable)
            lp_problem += variables[variable] <= bounds[variable][1]
        else:
            logger.info('%s: No upper bound for variable %s', label, variable)

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
            logger.info('%s: adding condition %s >= %f',
                        obj.identifier, key, scores[key] - eps[key])
            lp_problem += (obj.linear_programming[key] >= (scores[key] - eps[key]))
            logger.info('%s: adding condition %s <= %f',
                        obj.identifier, key, scores[key] + eps[key])
            lp_problem += (obj.linear_programming[key] <= (scores[key] + eps[key]))

    return lp_problem

def solve(obj, scores, eps, others=None, solver=None, solver_name='PULP_CBC_CMD', timeout=None):
    """
    Solving a problem.

    Args:
        obj (object): an object to solve
        scores (dict(str,float)): the scores to match
        eps (dict(str,float)/float): the numerical uncertainty
        solver_name (str): the name of the pulp solver to be used, check
                            pl.listSolvers(onlyAvailable=True) for the options
        timeout (int): the time limit in seconds

    Returns:
        pl.LpProblem: the solved linear programming problem
    """
    if not isinstance(eps, dict):
        eps = {key: eps for key in ['acc', 'sens', 'spec', 'bacc']}

    lp_program = pl.LpProblem('feasibility_' + random_identifier(8))

    lp_program = obj.init_lp(lp_program, scores)

    lp_program = create_lp_target(obj, scores, eps, lp_program)

    #lp_program += obj.linear_programming['objective']
    lp_program += 1

    lp_program.solve(solver=solver)

    return lp_program

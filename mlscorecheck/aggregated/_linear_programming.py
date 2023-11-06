"""
Some general purpose linear programming functionalities
"""
import pulp as pl

import numpy as np

from ..core import logger

from ._utils import random_identifier

__all__ = ["add_bounds", "solve", "create_lp_target"]


def add_bounds(
    lp_problem: pl.LpProblem, variables: dict, bounds: dict, label: str
) -> pl.LpProblem:
    """
    Adding bounds to a linear program

    Args:
        lp_problem (pl.LpProblem): the linear programming problem
        variables (dict(str,pl.LpVariable)): the variables to add bounds to
        bounds (dict(str,tuple(float|int,float|int))): the bounds to add
        label (str): the label of the entity containing the variables
                        bounds are added to

    Returns:
        pl.LpProblem: the adjusted linear programming problem
    """
    if bounds is None:
        return lp_problem

    for variable in bounds:
        if variable in variables:
            if bounds[variable][0] is not None and not np.isnan(bounds[variable][0]):
                logger.info(
                    "%s: adding lower bound %s for %s",
                    label,
                    str(bounds[variable][0]),
                    variable,
                )
                lp_problem += bounds[variable][0] <= variables[variable]
            else:
                logger.info("%s: No lower bound for variable %s", label, variable)
            if bounds[variable][1] is not None and not np.isnan(bounds[variable][1]):
                logger.info(
                    "%s: adding upper bound %s for %s",
                    label,
                    bounds[variable][1],
                    variable,
                )
                lp_problem += variables[variable] <= bounds[variable][1]
            else:
                logger.info("%s: No upper bound for variable %s", label, variable)

    return lp_problem


def create_lp_target(obj, scores: dict, eps, lp_problem: pl.LpProblem) -> pl.LpProblem:
    """
    Add the target and the score conditions to the linear programming problem

    Args:
        obj (Evaluation/Experiment): the object to process
        scores (dict(str,float)): the scores
        eps (dict(str,float)|float): the numerical uncertainties
        lp_problem (pl.LpProblem): the linear programming problem

    Returns:
        pl.LpProblem: the updated linear programming problem
    """
    for key in scores:
        if key in ["acc", "sens", "spec", "bacc"] and key in obj.scores:
            lp_problem += obj.scores[key] >= scores[key] - eps[key]
            lp_problem += obj.scores[key] <= scores[key] + eps[key]

    # adding the objective
    lp_problem += 1

    return lp_problem


def solve(obj, scores: dict, eps, solver=None) -> pl.LpProblem:
    """
    Solving a problem.

    Args:
        obj (object): an object to solve
        scores (dict(str,float)): the scores to match
        eps (dict(str,float)|float): the numerical uncertainty
        solver (obj): the solver object to use

    Returns:
        pl.LpProblem: the solved linear programming problem
    """
    if not isinstance(eps, dict):
        eps = {key: eps for key in ["acc", "sens", "spec", "bacc"]}

    lp_problem = pl.LpProblem(f"feasibility_{random_identifier(8)}")

    lp_problem = obj.init_lp(lp_problem, scores)

    lp_problem = create_lp_target(obj, scores, eps, lp_problem)

    lp_problem.solve(solver=solver)

    return lp_problem

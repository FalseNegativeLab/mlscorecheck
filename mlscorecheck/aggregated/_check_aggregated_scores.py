"""
This module provides a high level interface to the underlying structures
"""

from ..aggregated import compare_scores
from ..core import logger

from ._experiment import Experiment
from ._linear_programming import solve

__all__ = ['validate_experiment_specification',
            'check_aggregated_scores']

PREFERRED_SOLVER = 'PULP_CBC_CMD'
solvers = pl.listSolvers(onlyAvailable=True)

def check_aggregated_scores(experiment,
                            scores,
                            eps,
                            *,
                            solver_name=None,
                            timeout=None,
                            return_details=True):
    """
    Check aggregated scores

    Args:
        experiment (dict): the experiment specification
        scores (dict): the scores to match
        eps (dict/float): the numerical uncertainty
        solver_name (str): the name of the solver to be used, check
                            pulp.listSolvers(onlyAvailable) for the available list
        timeout (int): the number of seconds to time out
        return_details (bool): whether to return the details

    Returns:
        bool[, dict]: a flag which is True if inconsistency is identified, False otherwise
                        and optionally the details in a dictionary
    """
    experiment = Experiment(**experiment)

    solver_name = PREFERRED_SOLVER if solver_name is None else solver_name
    if solver_name not in solvers:
        logger.info('solver %s not available, using %s', solver_name, solvers[0])

    solver = pl.getSolver(solver_name, timeLimit=timeout)

    result = solve(experiment, scores, eps, solver)

    details = {}

    if result.status == 1:
        flag = False
        populated = experiment.populate(result)
        comp_flag = compare_scores(scores, populated.calculate_scores(), eps)
        details = populated.check_bounds()
        bounds_flag = details['bounds_flag']
    elif result.status == 0:
        flag = False
        # timed out
    else:
        flag = True
        # unsolvable

    return (flag, details) if return_details else flag

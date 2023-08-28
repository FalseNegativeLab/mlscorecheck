"""
This module provides a high level interface to the underlying aggregated
calculation structures
"""

import pulp as pl

from ..core import logger
from ..individual import resolve_aliases_and_complements

from ._experiment import Experiment
from ._linear_programming import solve
from ._utils import compare_scores, aggregated_scores

__all__ = ['check_aggregated_scores']

PREFERRED_SOLVER = 'PULP_CBC_CMD'
solvers = pl.listSolvers(onlyAvailable=True)

def check_aggregated_scores(*,
                            experiment,
                            scores,
                            eps,
                            solver_name=None,
                            timeout=None,
                            verbosity=1):
    """
    Check aggregated scores

    Args:
        experiment (dict/Experiment): the experiment specification
        scores (dict): the scores to match
        eps (dict/float): the numerical uncertainty
        solver_name (str): the name of the solver to be used, check
                            pulp.listSolvers(onlyAvailable) for the available list
        timeout (int): the number of seconds to time out
        verbosity (int): controls the verbosity level of the pulp based
                            linear programming solver. 0: no output; non-zero:
                            print output

    Returns:
        bool[, dict]: a flag which is True if inconsistency is identified, False otherwise
                        and optionally the details in a dictionary
    """
    scores = resolve_aliases_and_complements(scores)

    if all(score not in aggregated_scores for score in scores):
        logger.info('there are no scores suitable for aggregated checks')
        return {'inconsistency': False,
                'message': 'no scores suitable for aggregated consistency checks'}

    experiment = Experiment(**experiment) if isinstance(experiment, dict) else experiment

    solver_name = PREFERRED_SOLVER if solver_name is None else solver_name
    if solver_name not in solvers:
        logger.info('solver %s not available, using %s', solver_name, solvers[0])
        solver_name = solvers[0]

    solver = pl.getSolver(solver_name, timeLimit=timeout, msg=verbosity)

    result = solve(experiment, scores, eps, solver)

    populated = experiment.populate(result)
    configuration_details = populated.check_bounds()

    details = {}

    if result.status == 1:
        # the problem is feasible
        comp_flag = compare_scores(scores, populated.calculate_scores(), eps)
        bounds_flag = configuration_details['bounds_flag']
        return {'inconsistency': False,
                'lp_status': 'feasible',
                'lp_configuration_scores_match': comp_flag,
                'lp_configuration_bounds_match': bounds_flag,
                'lp_configuration': configuration_details}
    elif result.status == 0:
        # timed out
        return {'inconsistency': False,
                'lp_status': 'timeout',
                'lp_configuration': configuration_details}
    else:
        # infeasible
        return {'inconsistency': True,
                'lp_status': 'infeasible',
                'lp_configuration': configuration_details}


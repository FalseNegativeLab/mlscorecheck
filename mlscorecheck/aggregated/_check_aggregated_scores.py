"""
This module provides a high level interface to the underlying aggregated
calculation structures
"""

import pulp as pl

from ..core import (
    logger,
    NUMERICAL_TOLERANCE,
    check_uncertainty_and_tolerance,
    update_uncertainty,
)
from ..individual import resolve_aliases_and_complements

from ._experiment import Experiment
from ._linear_programming import solve
from ._utils import compare_scores, aggregated_scores

__all__ = ["check_aggregated_scores"]

solvers = pl.listSolvers(onlyAvailable=True)
PREFERRED_SOLVER = "PULP_CBC_CMD" if "PULP_CBC_CMD" in solvers else solvers[0]


def check_aggregated_scores(
    *,
    experiment: dict,
    scores: dict,
    eps,
    solver_name: str = None,
    timeout: int = None,
    verbosity: int = 1,
    numerical_tolerance: float = NUMERICAL_TOLERANCE
) -> dict:
    """
    Check aggregated scores

    Args:
        experiment (dict|Experiment): the experiment specification
        scores (dict): the scores to match
        eps (dict|float): the numerical uncertainty
        solver_name (str): the name of the solver to be used, check
                            pulp.listSolvers(onlyAvailable) for the available list
        timeout (int): the number of seconds to time out
        verbosity (int): controls the verbosity level of the pulp based
                            linear programming solver. 0: no output; non-zero:
                            print output
        numerical_tolerance (float): in practice, beyond the numerical uncertainty of
                                    the scores, some further tolerance is applied. This is
                                    orders of magnitude smaller than the uncertainty of the
                                    scores. It does ensure that the specificity of the test
                                    is 1, it might slightly decrease the sensitivity.

    Returns:
        dict: the details of the test, under the key 'inconsistency', one can find
        the flag indicating if inconsistency was identified

    Raises:
        ValueError: if the problem is not specified properly
    """

    check_uncertainty_and_tolerance(eps, numerical_tolerance)
    eps = update_uncertainty(eps, numerical_tolerance)

    scores = resolve_aliases_and_complements(scores)

    if all(score not in aggregated_scores for score in scores):
        logger.info("there are no scores suitable for aggregated checks")
        return {
            "inconsistency": False,
            "message": "no scores suitable for aggregated consistency checks",
        }

    experiment = (
        Experiment(**experiment) if isinstance(experiment, dict) else experiment
    )

    if experiment.aggregation == "som" and any(
        evaluation.aggregation == "mos" for evaluation in experiment.evaluations
    ):
        raise ValueError(
            "experiment level MoS aggregation with dataset level SoM "
            "aggregation is an unlikely situation, it is not supported in this high level "
            "interface."
        )

    solver_name = (
        PREFERRED_SOLVER
        if solver_name is None or solver_name not in solvers
        else solver_name
    )

    solver = pl.getSolver(
        solver_name,
        timeLimit=timeout,
        msg=verbosity,
        options=["RandomS 1", "RandomC 1", "randomS 1", "randomC 1"],
    )

    result = solve(experiment, scores, eps, solver)

    populated = experiment.populate(result)
    populated.calculate_scores(score_subset=list(scores.keys()))
    configuration_details = populated.check_bounds()

    if result.status == 1:
        # the problem is feasible
        comp_flag = compare_scores(scores, populated.scores, eps + numerical_tolerance)
        bounds_flag = configuration_details["bounds_flag"]
        return {
            "inconsistency": False,
            "lp_scores": populated.scores,
            "lp_status": "feasible",
            "lp_configuration_scores_match": comp_flag,
            "lp_configuration_bounds_match": bounds_flag,
            "lp_configuration": configuration_details,
        }
    if result.status == 0:
        # timed out
        return {
            "inconsistency": False,
            "lp_status": "timeout",
            "lp_configuration": configuration_details,
        }

    # infeasible
    return {
        "inconsistency": True,
        "lp_status": "infeasible",
        "lp_configuration": configuration_details,
    }

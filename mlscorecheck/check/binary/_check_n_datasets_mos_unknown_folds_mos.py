"""
This module implements the top level check function for
scores calculated by the mean of scores aggregation
in a kfold scenarios and mean of scores aggregation on multiple datasets,
but without knowing the fold configurations.
"""

import copy

import numpy as np

from ._check_n_datasets_mos_known_folds_mos import check_n_datasets_mos_known_folds_mos
from ._check_1_dataset_unknown_folds_mos import estimate_n_evaluations
from ...core import NUMERICAL_TOLERANCE
from ...aggregated import experiment_kfolds_generator

__all__ = ["check_n_datasets_mos_unknown_folds_mos", "estimate_n_experiments"]


def estimate_n_experiments(evaluations: list, available_scores: list = None) -> int:
    """
    Estimates the number of estimations with different fold combinations.

    Args:
        evaluations (list): a list of evaluation specifications

    Returns:
        int: the estimated number of different fold configurations.
    """
    available_scores = [] if available_scores is None else available_scores

    counts = [
        estimate_n_evaluations(
            dataset=evaluation["dataset"],
            folding=evaluation["folding"],
            available_scores=available_scores,
        )
        for evaluation in evaluations
    ]
    return np.prod(counts)


def check_n_datasets_mos_unknown_folds_mos(
    evaluations: list,
    scores: dict,
    eps,
    dataset_score_bounds: dict = None,
    *,
    solver_name: str = None,
    timeout: int = None,
    verbosity: int = 1,
    numerical_tolerance: float = NUMERICAL_TOLERANCE
) -> dict:
    """
    Checking the consistency of scores calculated in k-fold cross validation on multiple
    datasets, in mean-of-scores fashion, without knowing the fold configurations.
    The function generates all possible fold configurations and tests the
    consistency of each. The scores are inconsistent if all the k-fold configurations
    lead to inconsistencies identified.

    The test operates by constructing a linear program describing the experiment and checkings its
    feasibility.

    The test can only check the consistency of the 'acc', 'sens', 'spec' and 'bacc'
    scores. For a stronger test, one can add dataset_score_bounds when, for example, the minimum and
    the maximum scores over the datasets are also provided.

    Note that depending on the size of the dataset (especially the number of minority instances)
    and the folding configuration, this test might lead to an untractable number of problems to
    be solved. Use the function ``estimate_n_experiments`` to get an upper bound estimate
    on the number of fold combinations.

    The evaluation of possible fold configurations stops when a feasible configuration is found.

    Args:
        evaluations (list(dict)): the list of evaluation specifications
        scores (dict(str,float)): the scores to check
        eps (float|dict(str,float)): the numerical uncertainty(ies) of the scores
        dataset_score_bounds (None|dict(str,dict(float,float))): bounds on the scores in the
                                                                    datasets
        solver_name (None|str): the solver to use
        timeout (None|int): the timeout for the linear programming solver in seconds
        verbosity (int): the verbosity of the pulp linear programming solver,
                            0: silent, non-zero: verbose
        numerical_tolerance (float): in practice, beyond the numerical uncertainty of
                                    the scores, some further tolerance is applied. This is
                                    orders of magnitude smaller than the uncertainty of the
                                    scores. It does ensure that the specificity of the test
                                    is 1, it might slightly decrease the sensitivity.

    Returns:
        dict: A dictionary containing the results of the consistency check. The dictionary
        includes the following keys:

            - ``'inconsistency'``:
                A boolean flag indicating whether the set of feasible true
                positive (tp) and true negative (tn) pairs is empty. If True,
                it indicates that the provided scores are not consistent with the experiment.
            - ``'details'``:
                A list of dictionaries containing the details of the consistency tests. Each
                entry contains the specification of the folds being tested and the
                outcome of the ``check_n_datasets_known_folds_mos`` function.

    Raises:
        ValueError: if the problem is not specified properly

    Examples:
        >>> from mlscorecheck.check.binary import check_n_datasets_mos_unknown_folds_mos
        >>> evaluation0 = {'dataset': {'p': 13, 'n': 73},
                        'folding': {'n_folds': 4, 'n_repeats': 1}}
        >>> evaluation1 = {'dataset': {'p': 7, 'n': 26},
                        'folding': {'n_folds': 3, 'n_repeats': 1}}
        >>> evaluations = [evaluation0, evaluation1]
        >>> scores = {'acc': 0.357, 'sens': 0.323, 'spec': 0.362, 'bacc': 0.343}
        >>> result = check_n_datasets_mos_unknown_folds_mos(evaluations=evaluations,
                                                            scores=scores,
                                                            eps=1e-3)
        >>> result['inconsistency']
        # False

        >>> evaluation0 = {'dataset': {'p': 13, 'n': 73},
                        'folding': {'n_folds': 4, 'n_repeats': 1}}
        >>> evaluation1 = {'dataset': {'p': 7, 'n': 26},
                        'folding': {'n_folds': 3, 'n_repeats': 1}}
        >>> evaluations = [evaluation0, evaluation1]
        >>> scores = {'acc': 0.357, 'sens': 0.323, 'spec': 0.362, 'bacc': 0.9}
        >>> result = check_n_datasets_mos_unknown_folds_mos(evaluations=evaluations,
                                                            scores=scores,
                                                            eps=1e-3)
        >>> result['inconsistency']
        # True
    """
    if any(evaluation.get("aggregation", "mos") != "mos" for evaluation in evaluations):
        raise ValueError(
            'the aggregation specified in each dataset must be "mor" or nothing.'
        )
    if any(
        evaluation.get("fold_score_bounds") is not None for evaluation in evaluations
    ):
        raise ValueError("do not specify fold score bounds through this interface")

    evaluations = copy.deepcopy(evaluations)

    for evaluation in evaluations:
        evaluation["aggregation"] = "mos"

    experiment = {
        "evaluations": evaluations,
        "dataset_score_bounds": dataset_score_bounds,
        "aggregation": "mos",
    }

    results = {"details": [], "inconsistency": True}

    for experiment in experiment_kfolds_generator(experiment, list(scores.keys())):
        result = check_n_datasets_mos_known_folds_mos(
            evaluations=experiment["evaluations"],
            dataset_score_bounds=experiment.get("dataset_score_bounds"),
            scores=scores,
            eps=eps,
            timeout=timeout,
            solver_name=solver_name,
            verbosity=verbosity,
            numerical_tolerance=numerical_tolerance,
        )

        results["details"].append(result)
        results["inconsistency"] = results["inconsistency"] and result["inconsistency"]

        if not result["inconsistency"]:
            break

    return results

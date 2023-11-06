"""
This module implements the top level check function for
scores calculated by the score of means aggregation
in k-fold scenarios and mean of scores aggregation on multiple datasets.
"""

import copy

from ...aggregated import check_aggregated_scores, Experiment
from ...core import NUMERICAL_TOLERANCE

__all__ = ["check_n_datasets_mos_kfold_som"]


def check_n_datasets_mos_kfold_som(
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
    This function checks the consistency of scores calculated on multiple datasets with k-fold
    cross-validation, applying score of means aggregation over the folds and mean of scores
    aggregation over the datasets.

    The test operates by constructing a linear program describing the experiment and checkings its
    feasibility.

    The test can only check the consistency of the 'acc', 'sens', 'spec' and 'bacc'
    scores. For a stronger test, one can add ``dataset_score_bounds`` when, for example, the minimum
    and the maximum scores over the datasets are also provided.

    Args:
        evaluations (list(dict)): the list of evaluation specifications
        scores (dict(str,float)): the scores to check
        eps (float|dict(str,float)): the numerical uncertainty(ies) of the scores
        dataset_score_bounds (None|dict(str,tuple(float,float))): the potential bounds on the
                                                                scores in the datasets
        solver_name (None|str): the solver to use
        timeout (None|int): the timeout for the linear programming solver in seconds
        verbosity (int): the verbosity of the linear programming solver,
                            0: silent, 1: verbose.
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
            - ``'lp_status'``:
                The status of the lp solver.
            - ``'lp_configuration_scores_match'``:
                A flag indicating if the scores from the lp configuration match the scores
                provided.
            - ``'lp_configuration_bounds_match'``:
                Indicates if the specified bounds match the actual figures.
            - ``'lp_configuration'``:
                Contains the actual configuration of the linear programming solver.

    Raises:
        ValueError: if the problem is not specified properly

    Examples:
        >>> from mlscorecheck.check.binary import check_n_datasets_mos_kfold_som
        >>> evaluation0 = {'dataset': {'p': 39, 'n': 822},
                            'folding': {'n_folds': 5, 'n_repeats': 3,
                                        'strategy': 'stratified_sklearn'}}
        >>> evaluation1 = {'dataset': {'dataset_name': 'common_datasets.winequality-white-3_vs_7'},
                            'folding': {'n_folds': 5, 'n_repeats': 3,
                                        'strategy': 'stratified_sklearn'}}
        >>> evaluations = [evaluation0, evaluation1]
        >>> scores = {'acc': 0.312, 'sens': 0.45, 'spec': 0.312, 'bacc': 0.381}
        >>> result = check_n_datasets_mos_kfold_som(evaluations=evaluations,
                                                    dataset_score_bounds={'acc': (0.0, 0.5)},
                                                    eps=1e-4,
                                                    scores=scores)
        >>> result['inconsistency']
        # False

        >>> evaluation0 = {'dataset': {'p': 39, 'n': 822},
                            'folding': {'n_folds': 5, 'n_repeats': 3,
                                        'strategy': 'stratified_sklearn'}}
        >>> evaluation1 = {'dataset': {'dataset_name': 'common_datasets.winequality-white-3_vs_7'},
                            'folding': {'n_folds': 5, 'n_repeats': 3,
                                        'strategy': 'stratified_sklearn'}}
        >>> evaluations = [evaluation0, evaluation1]
        >>> scores = {'acc': 0.412, 'sens': 0.45, 'spec': 0.312, 'bacc': 0.381}
        >>> result = check_n_datasets_mos_kfold_som(evaluations=evaluations,
                                                    dataset_score_bounds={'acc': (0.5, 1.0)},
                                                    eps=1e-4,
                                                    scores=scores)
        >>> result['inconsistency']
        # True
    """

    if any(evaluation.get("aggregation", "som") != "som" for evaluation in evaluations):
        raise ValueError(
            'the aggregation specified in each dataset must be "rom" or nothing.'
        )

    if any(
        evaluation.get("fold_score_bounds") is not None for evaluation in evaluations
    ):
        raise ValueError("do not specify fold_score_bounds for a SoM evaluation")

    evaluations = copy.deepcopy(evaluations)

    for evaluation in evaluations:
        evaluation["aggregation"] = "som"

    experiment = Experiment(
        evaluations=evaluations,
        dataset_score_bounds=dataset_score_bounds,
        aggregation="mos",
    )

    return check_aggregated_scores(
        experiment=experiment.to_dict(),
        scores=scores,
        eps=eps,
        solver_name=solver_name,
        timeout=timeout,
        verbosity=verbosity,
        numerical_tolerance=numerical_tolerance,
    )

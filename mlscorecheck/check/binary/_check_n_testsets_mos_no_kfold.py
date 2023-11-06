"""
This module implements the top level check function for
scores calculated by the mean of scores aggregation over multiple testsets
with no k-fold cross-validation.
"""

from ...aggregated import check_aggregated_scores, Experiment, Dataset
from ...core import NUMERICAL_TOLERANCE

__all__ = ["check_n_testsets_mos_no_kfold"]


def check_n_testsets_mos_no_kfold(
    testsets: list,
    scores: dict,
    eps,
    testset_score_bounds: dict = None,
    *,
    solver_name: str = None,
    timeout: int = None,
    verbosity: int = 1,
    numerical_tolerance: float = NUMERICAL_TOLERANCE,
) -> dict:
    """
    This function checks the consistency of scores calculated on multiple testsets with no k-fold
    and aggregating the figures over the testsets in the mean of scores fashion.

    The test operates by constructing a linear program describing the experiment and checkings its
    feasibility.

    The test can only check the consistency of the 'acc', 'sens', 'spec' and 'bacc'
    scores. For a stronger test, one can add ``testset_score_bounds`` when, for example, the minimum
    and the maximum scores over the testsets are also provided.

    Args:
        testsets (list(dict)): the list of testset specifications
        scores (dict(str,float)): the scores to check
        eps (float|dict(str,float)): the numerical uncertainty(ies) of the scores
        testset_score_bounds (None|dict(str,tuple(float,float))): the potential bounds on the
                                                                scores in the testsets
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
        >>> from mlscorecheck.check.binary import check_n_testsets_mos_no_kfold
        >>> testsets = [{'p': 349, 'n': 50},
                        {'p': 478, 'n': 323},
                        {'p': 324, 'n': 83},
                        {'p': 123, 'n': 145}]
        >>> scores = {'acc': 0.6441, 'sens': 0.6706, 'spec': 0.3796, 'bacc': 0.5251}
        >>> result = check_n_testsets_mos_no_kfold(testsets=testsets,
                                                    eps=1e-4,
                                                    scores=scores)
        >>> result['inconsistency']
        # False

        >>> scores['sens'] = 0.6756
        >>> result = check_n_datasets_mos_no_kfold(testsets=testsets,
                                                    eps=1e-4,
                                                    scores=scores)
        >>> result['inconsistency']
        # True
    """

    datasets = [Dataset(**dataset) for dataset in testsets]

    evaluations = [
        {
            "dataset": dataset.to_dict(),
            "folding": {
                "folds": [
                    {
                        "p": dataset.p,
                        "n": dataset.n,
                        "identifier": f"{dataset.identifier}_{idx}",
                    }
                ]
            },
            "aggregation": "mos",
        }
        for idx, dataset in enumerate(datasets)
    ]

    experiment = Experiment(
        evaluations=evaluations,
        dataset_score_bounds=testset_score_bounds,
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

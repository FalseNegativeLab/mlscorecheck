"""
This module implements the consistency testing for multiclass macro averages
supposing one evaluation set
"""

from ...core import NUMERICAL_TOLERANCE
from ...aggregated import transform_multiclass_fold_to_binary

from ..binary import check_1_dataset_known_folds_mos

__all__ = ["check_1_testset_no_kfold_macro"]


def check_1_testset_no_kfold_macro(
    testset: dict,
    scores: dict,
    eps,
    *,
    class_score_bounds: dict = None,
    solver_name: str = None,
    timeout: int = None,
    verbosity: int = 1,
    numerical_tolerance: float = NUMERICAL_TOLERANCE
) -> dict:
    """
    The function tests the consistency of scores calculated by taking the macro average of
    class level scores on one single multiclass dataset.

    The test operates by constructing a linear programming problem representing the experiment
    and checking its feasibility.

    Note that this test can only check the consistency of the 'acc', 'sens', 'spec'
    and 'bacc' scores. Note that without bounds, if there is a large number of classes, it is
    likely that there will be a configuration matching the scores provided. In order to
    increase the strength of the test, one can add ``class_scores_bounds`` when, for example,
    besides the average score, the minimum and the maximum scores over the classes
    are also provided.

    Args:
        testset (dict): the specification of the testset
        scores (dict(str,float)): the scores to check
        eps (float|dict(str,float)): the numerical uncertainty(ies) of the scores
        class_score_bounds (None|dict(str,tuple(float,float))): bounds on the scores in the
                                                                classes
        solver_name (None|str): the solver to use
        timeout (None|int): the timeout for the linear programming solver in seconds
        verbosity (int): the verbosity level of the pulp linear programming solver
                            0: silent, non-zero: verbose.
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
        >>> from mlscorecheck.check.multiclass import check_1_testset_no_kfold_macro
        >>> testset = {0: 10, 1: 100, 2: 80}
        >>> scores = {'acc': 0.6, 'sens': 0.3417, 'spec': 0.6928, 'f1p': 0.3308}
        >>> results = check_1_testset_no_kfold_macro(scores=scores, testset=testset, eps=1e-4)
        >>> results['inconsistency']
        # False

        >>> scores['acc'] = 0.6020
        >>> results = check_1_testset_no_kfold_macro(scores=scores, testset=testset, eps=1e-4)
        >>> results['inconsistency']
        # True
    """
    folds = transform_multiclass_fold_to_binary(testset)
    dataset = {
        "p": sum(fold["p"] for fold in folds),
        "n": sum(fold["n"] for fold in folds),
    }

    return check_1_dataset_known_folds_mos(
        scores=scores,
        eps=eps,
        dataset=dataset,
        folding={"folds": folds},
        fold_score_bounds=class_score_bounds,
        solver_name=solver_name,
        timeout=timeout,
        verbosity=verbosity,
        numerical_tolerance=numerical_tolerance,
    )

"""
This module implements the multiclass tests in a k-fold MoS scenario with macro averaging
of scores of classes.
"""

from ...core import NUMERICAL_TOLERANCE
from ...aggregated import transform_multiclass_fold_to_binary, create_folds_multiclass

from ..binary import check_n_datasets_mos_known_folds_mos

__all__ = ["check_1_dataset_known_folds_mos_macro"]


def check_1_dataset_known_folds_mos_macro(
    dataset: dict,
    folding: dict,
    scores: dict,
    eps,
    *,
    class_score_bounds: dict = None,
    fold_score_bounds: dict = None,
    solver_name: str = None,
    timeout: int = None,
    verbosity: int = 1,
    numerical_tolerance: float = NUMERICAL_TOLERANCE
) -> dict:
    """
    Checking the consistency of scores calculated by taking the macro average
    of class-level scores on one single multiclass dataset with k-fold cross-validation.

    The test operates by constructing a linear program describing the experiment and checkings its
    feasibility.

    The test can only check the consistency of the 'acc', 'sens', 'spec' and 'bacc'
    scores. For a stronger test, one can add ``class_score_bounds`` or ``fold_score_bounds``
    when, for example, the minimum and the maximum scores over the classes or folds are available.

    Args:
        testset (dict): the specification of the testset
        scores (dict(str,float)): the scores to check
        eps (float|dict(str,float)): the numerical uncertainty(ies) of the scores
        class_score_bounds (None|dict(str,tuple(float,float))): the potential bounds on the
                                                                scores for the classes
        fold_score_bounds (None|dict(str,tuple(float,float))): the potential bounds on the
                                                                scores in the folds
        solver_name (None|str, optional): The solver to use. Defaults to None.
        timeout (None|int, optional): The timeout for the linear programming solver in seconds.
                                                    Defaults to None.
        verbosity (int, optional): The verbosity level of the pulp linear programming solver.
                                    0: silent, non-zero: verbose. Defaults to 1.
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
        >>> from mlscorecheck.check.multiclass import check_1_dataset_known_folds_mos_macro
        >>> dataset = {0: 149, 1: 118, 2: 83, 3: 154}
        >>> folding = {'n_folds': 4, 'n_repeats': 2, 'strategy': 'stratified_sklearn'}
        >>> scores = {'acc': 0.626, 'sens': 0.2483, 'spec': 0.7509, 'f1p': 0.2469}
        >>> result = check_1_dataset_known_folds_mos_macro(dataset=dataset,
                                                            folding=folding,
                                                            scores=scores,
                                                            eps=1e-4)
        >>> results['inconsistency']
        # False

        >>> scores['acc'] = 0.656
        >>> result = check_1_dataset_known_folds_mos_macro(dataset=dataset,
                                                        folding=folding,
                                                        scores=scores,
                                                        eps=1e-4)
        >>> result['inconsistency']
        # True
    """
    folds = create_folds_multiclass(dataset, folding)
    binary_folds = [transform_multiclass_fold_to_binary(fold) for fold in folds]

    evaluations = []

    for binary_folding in binary_folds:
        folding = {"folds": binary_folding}
        dataset = {
            "p": sum(tmp["p"] for tmp in binary_folding),
            "n": sum(tmp["n"] for tmp in binary_folding),
        }
        evaluations.append(
            {
                "dataset": dataset,
                "folding": folding,
                "fold_score_bounds": class_score_bounds,
            }
        )

    return check_n_datasets_mos_known_folds_mos(
        evaluations=evaluations,
        scores=scores,
        eps=eps,
        dataset_score_bounds=fold_score_bounds,
        solver_name=solver_name,
        timeout=timeout,
        verbosity=verbosity,
        numerical_tolerance=numerical_tolerance,
    )

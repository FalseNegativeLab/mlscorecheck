"""
This module implements the multiclass tests in a k-fold MoS scenario with micro
averaging of scores.
"""

from ...core import NUMERICAL_TOLERANCE
from ...aggregated import transform_multiclass_fold_to_binary, create_folds_multiclass

from ..binary import check_n_datasets_mos_kfold_som

__all__ = ["check_1_dataset_known_folds_mos_micro"]


def check_1_dataset_known_folds_mos_micro(
    dataset: dict,
    folding: dict,
    scores: dict,
    eps,
    *,
    fold_score_bounds: dict = None,
    solver_name: str = None,
    timeout: int = None,
    verbosity: int = 1,
    numerical_tolerance: float = NUMERICAL_TOLERANCE
) -> dict:
    """
    This function checks the consistency of scores calculated by taking the micro average
    on a single multiclass dataset with known folds.

    The test operates by constructing a linear program describing the experiment and checkings its
    feasibility.

    The test can only check the consistency of the 'acc', 'sens', 'spec' and 'bacc'
    scores. For a stronger test, one can add ``fold_score_bounds`` when, for example, the minimum
    and the maximum scores over the folds are available.

    Args:
        dataset (dict): The specification of the dataset.
        folding (dict): The specification of the folding strategy.
        scores (dict(str,float)): The scores to check.
        eps (float|dict(str,float)): The numerical uncertainty(ies) of the scores.
        fold_score_bounds (None|dict(str,tuple(float,float))): the potential bounds on the
                                                                scores in the folds
        solver_name (None|str, optional): The solver to use. Defaults to None.
        timeout (None|int, optional): The timeout for the linear programming solver in seconds.
                                        Defaults to None.
        verbosity (int, optional): The verbosity level of the pulp linear programming solver.
                                    0: silent, non-zero: verbose. Defaults to 1.
        numerical_tolerance (float, optional): Beyond the numerical uncertainty of
                                                the scores, some further tolerance is applied.
                                                This is orders of magnitude smaller than the
                                                uncertainty of the scores. It ensures that the
                                                specificity of the test is 1, it might slightly
                                                decrease the sensitivity. Defaults to
                                                NUMERICAL_TOLERANCE.

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
        >>> from mlscorecheck.check.multiclass import check_1_dataset_known_folds_mos_micro
        >>> dataset = {0: 66, 1: 178, 2: 151}
        >>> folding = {'folds': [{0: 33, 1: 89, 2: 76}, {0: 33, 1: 89, 2: 75}]}
        >>> scores = {'acc': 0.5646, 'sens': 0.3469, 'spec': 0.6734, 'f1p': 0.3469}
        >>> result = check_1_dataset_known_folds_mos_micro(dataset=dataset,
                                                folding=folding,
                                                scores=scores,
                                                eps=1e-4)
        >>> result['inconsistency']
        # False

        >>> scores['acc'] = 0.5746
        >>> result = check_1_dataset_known_folds_mos_micro(dataset=dataset,
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
        evaluations.append({"dataset": dataset, "folding": folding})

    return check_n_datasets_mos_kfold_som(
        evaluations=evaluations,
        scores=scores,
        eps=eps,
        dataset_score_bounds=fold_score_bounds,
        solver_name=solver_name,
        timeout=timeout,
        verbosity=verbosity,
        numerical_tolerance=numerical_tolerance,
    )

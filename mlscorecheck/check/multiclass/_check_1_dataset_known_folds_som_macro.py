"""
This module implements the macro-averaged multiclass tests in a k-fold scenario
with SoM aggregation over the folds.
"""

import copy

from ...core import NUMERICAL_TOLERANCE
from ...aggregated import create_folds_multiclass

from ._check_1_testset_no_kfold_macro import check_1_testset_no_kfold_macro

__all__ = ["check_1_dataset_known_folds_som_macro"]


def check_1_dataset_known_folds_som_macro(
    dataset: dict,
    folding: dict,
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
    This function checks the consistency of scores calculated by taking the macro average
    on a single multiclass dataset and averaging the scores across the folds in the SoM manner.

    The test operates by constructing a linear program describing the experiment and checkings its
    feasibility.

    The test can only check the consistency of the 'acc', 'sens', 'spec' and 'bacc'
    scores. For a stronger test, one can add ``class_score_bounds`` when, for example, the minimum
    and the maximum scores over the classes are available.

    Args:
        dataset (dict): The specification of the dataset.
        folding (dict): The specification of the folding.
        scores (dict(str,float)): The scores to check.
        eps (float|dict(str,float)): The numerical uncertainty(ies) of the scores.
        class_score_bounds (None|dict(str,tuple(float,float))): the potential bounds on the
                                                                scores for the classes
        solver_name (None|str, optional): The solver to use. Defaults to None.
        timeout (None|int, optional): The timeout for the linear programming solver in seconds.
                                        Defaults to None.
        verbosity (int, optional): The verbosity level of the pulp linear programming solver.
                                    0: silent, non-zero: verbose. Defaults to 1.
        numerical_tolerance (float, optional): In practice, beyond the numerical uncertainty of
                                                the scores, some further tolerance is applied.
                                                This is orders of magnitude smaller than the
                                                uncertainty of the scores. It does ensure that the
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
        ValueError: If the provided scores are not consistent with the dataset.

    Examples:
        >>> from mlscorecheck.check.multiclass import check_1_dataset_known_folds_som_macro
        >>> dataset = {0: 129, 1: 81, 2: 135}
        >>> folding = {'n_folds': 2, 'n_repeats': 2, 'strategy': 'stratified_sklearn'}
        >>> scores = {'acc': 0.5662, 'sens': 0.3577, 'spec': 0.6767, 'f1p': 0.3481}
        >>> result = check_1_dataset_known_folds_som_macro(dataset=dataset,
                                                folding=folding,
                                                scores=scores,
                                                eps=1e-4)
        >>> result['inconsistency']
        # False

        >>> scores['acc'] = 0.6762
        >>> result = check_1_dataset_known_folds_som_macro(dataset=dataset,
                                                folding=folding,
                                                scores=scores,
                                                eps=1e-4)
        >>> result['inconsistency']
        # True
    """
    folds = create_folds_multiclass(dataset, folding)

    testset = copy.deepcopy(folds[0])
    for fold in folds[1:]:
        for key in fold:
            testset[key] += fold[key]

    return check_1_testset_no_kfold_macro(
        testset=testset,
        scores=scores,
        eps=eps,
        class_score_bounds=class_score_bounds,
        solver_name=solver_name,
        timeout=timeout,
        verbosity=verbosity,
        numerical_tolerance=numerical_tolerance,
    )

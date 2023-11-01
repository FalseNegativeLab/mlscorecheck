"""
This module implements the micro-averaged multiclass tests in a k-fold scenario.
"""

import copy

from ...core import NUMERICAL_TOLERANCE
from ...aggregated import transform_multiclass_fold_to_binary, _create_folds_multiclass, _create_binary_folds_multiclass

from ._check_1_testset_no_kfold_micro import check_1_testset_no_kfold_micro
from ._check_1_testset_no_kfold_macro import check_1_testset_no_kfold_macro

__all__ = ['check_1_dataset_known_folds_som']

def check_1_dataset_known_folds_som(dataset: dict,
                                    folding: dict,
                                    scores: dict,
                                    eps,
                                    average: str,
                                    *,
                                    class_score_bounds: dict = None,
                                    solver_name: str = None,
                                    timeout: int = None,
                                    verbosity: int = 1,
                                    numerical_tolerance: float = NUMERICAL_TOLERANCE) -> dict:
    """
    This function checks the consistency of scores calculated by taking the micro or macro average
    on a single multiclass dataset and averaging across the folds in the SoM manner.

    Args:
        dataset (dict): The specification of the dataset.
        folding (dict): The specification of the folding strategy.
        scores (dict(str,float)): The scores to check.
        eps (float|dict(str,float)): The numerical uncertainty(ies) of the scores.
        average (str): The type of averaging to be used.
        class_score_bounds (dict, optional): The bounds for the class scores. Defaults to None.
        solver_name (str, optional): The name of the solver. Defaults to None.
        timeout (int, optional): The maximum time allowed for the operation. Defaults to None.
        verbosity (int, optional): The level of verbosity. Defaults to 1.
        numerical_tolerance (float, optional): Beyond the numerical uncertainty of
                                               the scores, some further tolerance is applied. This is
                                               orders of magnitude smaller than the uncertainty of the
                                               scores. Defaults to NUMERICAL_TOLERANCE.

    Returns:
        dict: A dictionary containing the results of the consistency check. The dictionary includes the following keys:
            - 'inconsistency': A boolean flag indicating whether the set of feasible true positive (tp) and true negative (tn) pairs is empty. If True, it indicates that the provided scores are not consistent with the dataset.
            - 'details': A list providing further details from the analysis of the scores one after the other. Each entry in the list corresponds to the analysis result for one score.
            - 'n_valid_tptn_pairs': The number of tp and tn pairs that are compatible with all scores. This gives an indication of how many different classification outcomes could have led to the provided scores.
            - 'prefiltering_details': The results of the prefiltering by using the solutions for the score pairs. This provides additional information about the process of checking the scores.

    Raises:
        ValueError: If the provided scores are not consistent with the dataset.
    """
    folds = _create_folds_multiclass(dataset, folding)

    print(folds)

    testset = copy.deepcopy(folds[0])
    for fold in folds[1:]:
        for key in fold:
            testset[key] += fold[key]
    print(testset)

    if average == 'micro':
        return check_1_testset_no_kfold_micro(testset=testset,
                                                scores=scores,
                                                eps=eps,
                                                numerical_tolerance=numerical_tolerance)
    elif average == 'macro':
        return check_1_testset_no_kfold_macro(testset=testset,
                                                scores=scores,
                                                eps=eps,
                                                class_score_bounds=class_score_bounds,
                                                solver_name=solver_name,
                                                timeout=timeout,
                                                verbosity=verbosity,
                                                numerical_tolerance=numerical_tolerance)

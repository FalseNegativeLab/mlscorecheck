"""
This module implements the micro-averaged multiclass tests in a k-fold scenario
with SoM aggregation.
"""

import copy

from ...core import NUMERICAL_TOLERANCE
from ...aggregated import transform_multiclass_fold_to_binary, _create_folds_multiclass, _create_binary_folds_multiclass

from ._check_1_testset_no_kfold_micro import check_1_testset_no_kfold_micro
from ._check_1_testset_no_kfold_macro import check_1_testset_no_kfold_macro

__all__ = ['check_1_dataset_known_folds_som_micro']

def check_1_dataset_known_folds_som_micro(dataset: dict,
                                    folding: dict,
                                    scores: dict,
                                    eps,
                                    *,
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

    Examples:
        >>> from mlscorecheck.check.multiclass import check_1_dataset_known_folds_som_micro
        >>> dataset = {0: 86, 1: 96, 2: 59, 3: 105}
        >>> folding = {'folds': [{0: 43, 1: 48, 2: 30, 3: 52}, {0: 43, 1: 48, 2: 29, 3: 53}]}
        >>> scores =  {'acc': 0.6272, 'sens': 0.2543, 'spec': 0.7514, 'f1p': 0.2543}
        >>> result = check_1_dataset_known_folds_som_micro(dataset=dataset,
                                                            folding=folding,
                                                            scores=scores,
                                                            eps=1e-4)
        >>> result['inconsistency']
        # False

        >>> scores['acc'] = 0.8756
        >>> result = check_1_dataset_known_folds_som_micro(dataset=dataset,
                                                folding=folding,
                                                scores=scores,
                                                eps=1e-4)
        >>> result['inconsistency']
        # True
    """
    folds = _create_folds_multiclass(dataset, folding)

    print(folds)

    testset = copy.deepcopy(folds[0])
    for fold in folds[1:]:
        for key in fold:
            testset[key] += fold[key]

    return check_1_testset_no_kfold_micro(testset=testset,
                                            scores=scores,
                                            eps=eps,
                                            numerical_tolerance=numerical_tolerance)

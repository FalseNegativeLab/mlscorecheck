"""
This module implements the multiclass tests in a k-fold MoS scenario with micro
averaging of scores.
"""

from ...core import NUMERICAL_TOLERANCE
from ...aggregated import (transform_multiclass_fold_to_binary,
                            create_folds_multiclass)

from ..binary import check_n_datasets_mos_kfold_som

__all__ = ['check_1_dataset_known_folds_mos_micro']

def check_1_dataset_known_folds_mos_micro(dataset: dict,
                                    folding: dict,
                                    scores: dict,
                                    eps,
                                    *,
                                    fold_score_bounds: dict = None,
                                    solver_name: str = None,
                                    timeout: int = None,
                                    verbosity: int = 1,
                                    numerical_tolerance: float = NUMERICAL_TOLERANCE) -> dict:
    """
    This function checks the consistency of scores calculated by taking the micro average
    on a single multiclass dataset with known folds. The test follows the methodology of the
    1_dataset_som case, but is specifically designed for multiclass classification problems
    where the folds of the dataset are known beforehand.

    Args:
        dataset (dict): The specification of the dataset.
        folding (dict): The specification of the folding strategy.
        scores (dict(str,float)): The scores to check.
        eps (float|dict(str,float)): The numerical uncertainty(ies) of the scores.
        fold_score_bounds (None|dict, optional): Bounds on the scores in the folds.
                                                    Defaults to None.
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
        dict: A summary of the results. The dictionary includes the following keys:
            - ``inconsistency``: A boolean flag indicating whether the set of feasible
                true positive (tp) and true negative (tn) pairs is empty. If True, it
                indicates that the provided scores are not consistent with the dataset.
            - ``details``: A list providing further details from the analysis of the
                scores one after the other. Each entry in the list corresponds to the
                analysis result for one score.
            - ``n_valid_tptn_pairs``: The number of tp and tn pairs that are compatible
                with all scores. This gives an indication of how many different classification
                outcomes could have led to the provided scores.
            - ``prefiltering_details``: The results of the prefiltering by using the
                solutions for the score pairs. This provides additional information about the
                process of checking the scores.


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
        >>> result['inconsistency']
        # False

        >>> scores['acc'] = 0.8745
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
        folding = {'folds': binary_folding}
        dataset = {'p': sum(tmp['p'] for tmp in binary_folding),
                    'n': sum(tmp['n'] for tmp in binary_folding)}
        evaluations.append({'dataset': dataset,
                            'folding': folding})

    return check_n_datasets_mos_kfold_som(evaluations=evaluations,
                                            scores=scores,
                                            eps=eps,
                                            dataset_score_bounds=fold_score_bounds,
                                            solver_name=solver_name,
                                            timeout=timeout,
                                            verbosity=verbosity,
                                            numerical_tolerance=numerical_tolerance)

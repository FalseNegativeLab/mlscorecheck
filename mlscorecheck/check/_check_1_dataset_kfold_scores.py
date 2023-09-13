"""
This module implements a general testing function for k-folded evaluations
"""

from ..core import NUMERICAL_TOLERANCE

from ._check_1_dataset_kfold_mor_scores import check_1_dataset_kfold_mor_scores
from ._check_1_dataset_kfold_rom_scores import check_1_dataset_kfold_rom_scores

def check_1_dataset_kfold_scores(scores,
                                    eps,
                                    dataset,
                                    *,
                                    solver_name=None,
                                    timeout=None,
                                    verbosity=1,
                                    numerical_tolerance=NUMERICAL_TOLERANCE):
    """
    Checking the consistency of scores calculated in a k-fold cross validation on a single
    dataset, without knowing if the aggregation was ratio-of-means or mean-of-ratios. The
    function tests the scores by both tests, and unifies the results by the rule that
    if both tests identify inconsistencies, then the configuration is inconsistent,
    however, if any of the tests passes without identifying inconsistencies, then
    the discovery of inconsistencies is not confirmed.

    Args:
        scores (dict(str,float)): the scores to check
        eps (float|dict(str,float)): the numerical uncertainty(ies) of the scores
        dataset (dict): the dataset specification
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
        dict: the dictionary of the results of the analysis, the
        ``inconsistency`` entry indicates if inconsistencies have
        been found. The details of the mean-of-ratios and ratio-of-means
        testing can be found under the ``mor_results`` and ``rom_results``
        keys.

    Raises:
        ValueError: if the problem is not specified properly

    Examples:
        >>> dataset = {'folds': [{'p': 52, 'n': 94}, {'p': 74, 'n': 37}]}
        >>> scores = {'acc': 0.573, 'sens': 0.768, 'bacc': 0.662}
        >>> result = check_1_dataset_kfold_scores(scores=scores,
                                                eps=1e-3,
                                                dataset=dataset)
        >>> result['inconsistency']
        # False

        >>> dataset = {'p': 398,
                        'n': 569,
                        'n_folds': 4,
                        'n_repeats': 2,
                        'folding': 'stratified_sklearn'}
        >>> scores = {'acc': 0.9, 'spec': 0.9, 'sens': 0.6}
        >>> result = check_1_dataset_kfold_scores(scores=scores,
                                                eps=1e-2,
                                                dataset=dataset)
        >>> result['inconsistency']
        # True

        >>> dataset = {'name': 'common_datasets.glass_0_1_6_vs_2',
                        'n_folds': 4,
                        'n_repeats': 2,
                        'folding': 'stratified_sklearn',
                        'fold_score_bounds': {'acc': (0.8, 1.0)}}
        >>> scores = {'acc': 0.9, 'spec': 0.9, 'sens': 0.6, 'bacc': 0.1, 'f1p': 0.95}
        >>> result = check_1_dataset_kfold_scores(scores=scores,
                                                eps=1e-2,
                                                dataset=dataset)
        >>> result['inconsistency']
        # True

    """
    results_mor = check_1_dataset_kfold_mor_scores(scores=scores,
                                                    eps=eps,
                                                    dataset=dataset,
                                                    solver_name=solver_name,
                                                    timeout=timeout,
                                                    verbosity=verbosity,
                                                    numerical_tolerance=numerical_tolerance)
    results_rom = check_1_dataset_kfold_rom_scores(scores=scores,
                                                    eps=eps,
                                                    dataset=dataset,
                                                    solver_name=solver_name,
                                                    timeout=timeout,
                                                    verbosity=verbosity,
                                                    numerical_tolerance=numerical_tolerance)

    return {'inconsistency': results_mor['inconsistency'] and results_rom['inconsistency'],
            'mor_results': results_mor,
            'rom_results': results_rom}

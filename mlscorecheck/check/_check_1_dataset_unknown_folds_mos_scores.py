"""
This module implements consistency testing for scores calculated in a k-fold cross-validation
scenario with unknown fold structures.
"""

from ..core import NUMERICAL_TOLERANCE
from ..aggregated import (Dataset,
                            repeated_kfolds_generator,
                            kfolds_generator)
from ._check_1_dataset_known_folds_mos_scores import check_1_dataset_known_folds_mos_scores

__all__ = ['check_1_dataset_unknown_folds_mos_scores',
            'estimate_n_evaluations']

def estimate_n_evaluations(dataset: dict,
                            folding: dict,
                            available_scores: list = None) -> int:
    """
    Estimates the number of estimations with different fold combinations.

    Args:
        dataset (dict): the dataset specification
        folding (dict): the folding specification
        available_scores (list): the list of available scores

    Returns:
        int: the estimated number of different fold configurations.
    """
    dataset = Dataset(**dataset)
    n_repeats = folding.get('n_repeats', 1)

    available_scores = [] if available_scores is None else available_scores

    count = sum(1 for _ in kfolds_generator({'dataset': dataset.to_dict(),
                                                'folding': folding},
                                            available_scores))

    return count**n_repeats

def check_1_dataset_unknown_folds_mos_scores(
                                        dataset: dict,
                                        folding: dict,
                                        scores: dict,
                                        eps,
                                        fold_score_bounds: dict = None,
                                        *,
                                        solver_name: str = None,
                                        timeout: int = None,
                                        verbosity: int = 1,
                                        numerical_tolerance: float = NUMERICAL_TOLERANCE) -> dict:
    """
    Checking the consistency of scores calculated in a k-fold cross validation on a single
    dataset, in a mean-of-scores fashion, without knowing the folding strategy.
    The function generates all possible foldings of k-valid folds and evaluates the
    consistency on each of them. The scores are inconsistent if all the k-fold configurations
    lead to inconsistencies identified.

    Note that depending on the size of the dataset (especially the number of minority instances)
    and the folding configuration, this test might lead to an untractable number of problems to
    be solved. Use the function ``estimate_n_evaluations`` to get a rough upper bound estimate
    on the number of fold combinations.

    Args:
        dataset (dict): the dataset specification
        folding (dict): the folding specification
        scores (dict(str,float)): the scores to check
        eps (float|dict(str,float)): the numerical uncertainty(ies) of the scores
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
        been found. The details of the mean-of-scores checks and all fold configurations
        can be found under the ``details`` key.

    Raises:
        ValueError: if the problem is not specified properly

    Examples:
        >>> dataset = {'p': 126, 'n': 131}
        >>> folding = {'n_folds': 2, 'n_repeats': 1}
        >>> scores = {'acc': 0.573, 'sens': 0.768, 'bacc': 0.662}
        >>> result = check_1_dataset_unknown_folds_mos_scores(dataset=dataset,
                                                                folding=folding,
                                                                scores=scores,
                                                                eps=1e-3)
        >>> result['inconsistency']
        # False

        >>> dataset = {'p': 19, 'n': 97}
        >>> folding = {'n_folds': 3, 'n_repeats': 1}
        >>> scores = {'acc': 0.9, 'spec': 0.9, 'sens': 0.6}
        >>> result = check_1_dataset_unknown_folds_mos_scores(dataset=dataset,
                                                                folding=folding,
                                                                scores=scores,
                                                                eps=1e-4)
        >>> result['inconsistency']
        # True
    """
    evaluation = {'dataset': dataset,
                    'folding': folding,
                    'fold_score_bounds': fold_score_bounds,
                    'aggregation': 'mos'}

    results = {'details': []}

    idx = 0
    for evaluation_0 in repeated_kfolds_generator(evaluation,
                                                    list(scores.keys())):
        tmp = {'folds': evaluation_0['folding']['folds'],
                'details': check_1_dataset_known_folds_mos_scores(
                                    scores=scores,
                                    eps=eps,
                                    dataset=evaluation_0['dataset'],
                                    folding=evaluation_0['folding'],
                                    fold_score_bounds=evaluation_0.get('fold_score_bounds'),
                                    solver_name=solver_name,
                                    timeout=timeout,
                                    verbosity=verbosity,
                                    numerical_tolerance=numerical_tolerance),
                'configuration_id': idx}
        results['details'].append(tmp)
        if not tmp['details']['inconsistency']:
            break
        idx += 1

    results['inconsistency'] = all(tmp['details']['inconsistency'] for tmp in results['details'])

    return results

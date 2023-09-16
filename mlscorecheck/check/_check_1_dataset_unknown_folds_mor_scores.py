"""
This module implements consistency testing for scores calculated in a k-fold cross-validation
scenario with unknown fold structures.
"""

import copy

from ..core import NUMERICAL_TOLERANCE, logger
from ..aggregated import generate_evaluations_with_all_kfolds
from ._check_1_dataset_known_folds_mor_scores import check_1_dataset_known_folds_mor_scores

__all__ = ['check_1_dataset_unknown_folds_mor_scores']

def check_1_dataset_unknown_folds_mor_scores(
                                        scores: dict,
                                        eps,
                                        evaluation: dict,
                                        *,
                                        solver_name: str = None,
                                        timeout: int = None,
                                        verbosity: int = 1,
                                        numerical_tolerance: float = NUMERICAL_TOLERANCE) -> dict:
    """
    Checking the consistency of scores calculated in a k-fold cross validation on a single
    dataset, in a mean-of-ratios fashion, without knowing the folding strategy.
    The function generates all possible foldings of k-valid folds and evaluates the
    consistency on each of them. The scores are inconsistent if all the k-fold configurations
    lead to inconsistencies identified.

    Note that depending on the size of the dataset (especially the number of minority instances)
    and the folding configuration, this test might lead to an untractable number of problems to
    be solved.

    Args:
        scores (dict(str,float)): the scores to check
        eps (float|dict(str,float)): the numerical uncertainty(ies) of the scores
        evaluation (dict): the evaluation specification
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
        been found. The details of the mean-of-ratios checks and all fold configurations
        can be found under the ``details`` key.

    Raises:
        ValueError: if the problem is not specified properly

    Examples:
        >>> evaluation = {'dataset': {'p': 126, 'n': 131},
                            'folding': {'n_folds': 2, 'n_repeats': 1}}
        >>> scores = {'acc': 0.573, 'sens': 0.768, 'bacc': 0.662}
        >>> result = check_1_dataset_unknown_folds_mor_scores(evaluation=evaluation,
                                                                scores=scores,
                                                                eps=1e-3)
        >>> result['inconsistency']
        # False

        >>> dataset = {'p': 19, 'n': 97}
        >>> folding = {'n_folds': 3, 'n_repeats': 1}
        >>> evaluation = {'dataset': dataset, 'folding': folding}
        >>> scores = {'acc': 0.9, 'spec': 0.9, 'sens': 0.6}
        >>> result = check_1_dataset_unknown_folds_mor_scores(evaluation=evaluation,
                                                            scores=scores,
                                                            eps=1e-4)
        >>> result['inconsistency']
        # True
    """
    if evaluation.get('aggregation', 'mor') != 'mor':
        raise ValueError("either don't specify the aggregation or set it to 'mor'")

    evaluation = copy.deepcopy(evaluation) | {'aggregation': 'mor'}

    evaluations = generate_evaluations_with_all_kfolds(evaluation)

    logger.info('The total number of fold combinations: %d', len(evaluations))

    results = {'details': []}

    for evaluation_0 in evaluations:
        tmp = {'folds': evaluation_0['folding']['folds'],
                'details': check_1_dataset_known_folds_mor_scores(scores=scores,
                                                        eps=eps,
                                                        evaluation=evaluation_0,
                                                        solver_name=solver_name,
                                                        timeout=timeout,
                                                        verbosity=verbosity,
                                                        numerical_tolerance=numerical_tolerance)}
        results['details'].append(tmp)
        if not tmp['details']['inconsistency']:
            break

    results['inconsistency'] = all(tmp['details']['inconsistency'] for tmp in results['details'])

    return results

"""
This module implements consistency testing for scores calculated in a k-fold cross-validation
scenario with unknown fold structures.
"""

from ..core import NUMERICAL_TOLERANCE, logger
from ..aggregated import generate_datasets_with_all_kfolds
from ._check_1_dataset_kfold_mor_scores import check_1_dataset_kfold_mor_scores

def check_1_dataset_kfold_mor_unknown_folds_scores(scores,
                                                    eps,
                                                    dataset,
                                                    *,
                                                    solver_name=None,
                                                    timeout=None,
                                                    verbosity=1,
                                                    numerical_tolerance=NUMERICAL_TOLERANCE):
    """
    Checking the consistency of scores calculated in a k-fold cross validation on a single
    dataset, in a mean-of-ratios fashion, without knowing the folding strategy.
    The function generates all possible foldings of k-valid folds and evaluates the
    consistency on each of them. The scores are inconsistent if all the k-fold configurations
    lead to inconsistencies identified.

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
        been found. The details of the mean-of-ratios checks and all fold configurations
        can be found under the ``details`` key.

    Raises:
        ValueError: if the problem is not specified properly
    """

    datasets = generate_datasets_with_all_kfolds(dataset)

    logger.info('The total number of fold combinations: %d', len(datasets))

    results = {'details': []}

    for dataset_0 in datasets:
        tmp = {'dataset': dataset_0,
                'details': check_1_dataset_kfold_mor_scores(scores=scores,
                                                        eps=eps,
                                                        dataset=dataset_0,
                                                        solver_name=solver_name,
                                                        timeout=timeout,
                                                        verbosity=verbosity,
                                                        numerical_tolerance=numerical_tolerance)}
        results['details'].append(tmp)

    results['inconsistency'] = all(tmp['details']['inconsistency'] for tmp in results['details'])

    return results

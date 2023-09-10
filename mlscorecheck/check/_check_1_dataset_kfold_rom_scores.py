"""
This module implements the top level check function for
scores calculated by the ratio-of-means aggregation
in a kfold scenario on one single dataset.
"""

import warnings

from ..core import logger, NUMERICAL_TOLERANCE
from ..individual import check_scores_tptn
from ..aggregated import check_aggregated_scores, Experiment

__all__ = ['check_1_dataset_kfold_rom_scores']

def check_1_dataset_kfold_rom_scores(scores,
                                        eps,
                                        dataset,
                                        *,
                                        solver_name=None,
                                        timeout=None,
                                        verbosity=1,
                                        numerical_tolerance=NUMERICAL_TOLERANCE):
    """
    Checking the consistency of scores calculated by applying k-fold
    cross validation to one single dataset and aggregating the figures
    over the folds in the ratio of means fashion. All pairs of
    the supported individual scores are checked against all other as in
    the 1_dataset_no_kfold case, however, additionally, if score_bounds
    are specified in the folds, the aggregated check is also executed
    on the supported acc, bacc, sens and spec scores.

    Args:
        scores (dict(str,float)): the scores to check
        eps (float|dict(str,float)): the numerical uncertainty(ies) of the scores
        dataset (dict): the dataset specification
        solver_name (None|str): the solver to use
        timeout (None|int): the timeout for the linear programming solver in seconds
        verbosity (int): the verbosity level of the pulp solver (0: silent, non-0: verbose)
        numerical_tolerance (float): in practice, beyond the numerical uncertainty of
                                    the scores, some further tolerance is applied. This is
                                    orders of magnitude smaller than the uncertainty of the
                                    scores. It does ensure that the specificity of the test
                                    is 1, it might slightly decrease the sensitivity.

    Returns:
        dict: the dictionary of the results of the analysis, the
        ``inconsistency`` entry indicates if inconsistencies have
        been found. The aggregated_results entry is empty if
        the execution of the linear programming based check was
        unnecessary. The result contains two more keys. Under the key
        ``individual_results`` one finds a structure similar to that of
        the 1_dataset_no_kfold output, summarizing the findings of
        checking the consistency of each pair of scores against a third
        score. Additionally, the key ``aggregated_results`` is available
        if score_bounds are specified and the aggregated checks are
        executed. In the aggregated_results one finds again the
        ``inconsistency`` flag, the status of the linear programming solver
        (``lp_status``) and under the ``lp_configuration`` key one finds
        the actual configuration of tp and tn variables with all
        scores calculated and the score_bounds checked where they
        were specified. Also, under the ``lp_status`` key one finds the
        status message of the linear solver, and in the as a double check, the
        flag under the key ``lp_configuration_scores_match`` indicates if the
        scores of the final configuration match the specified ones, similarly,
        the ``lp_configuration_bounds_match`` indicates if all bounds are
        satisfied.

    Raises:
        ValueError: if the problem is not specified properly

    Examples:
        >>> dataset = {'folds': [{'p': 16, 'n': 99},
                            {'p': 81, 'n': 69},
                            {'p': 83, 'n': 2},
                            {'p': 52, 'n': 19},
                            {'p': 28, 'n': 14}]}
        >>> scores = {'acc': 0.428, 'npv': 0.392, 'bacc': 0.442, 'f1p': 0.391}
        >>> result = check_1_dataset_kfold_rom_scores(scores=scores,
                                                    eps=1e-3,
                                                    dataset=dataset)
        >>> result['inconsistency']
        # False

        >>> dataset = {'name': 'common_datasets.glass_0_1_6_vs_2',
                        'n_folds': 4,
                        'n_repeats': 2,
                        'folding': 'stratified_sklearn'}
        >>> scores = {'acc': 0.9, 'npv': 0.9, 'sens': 0.6, 'f1p': 0.95}
        >>> result = check_1_dataset_kfold_rom_scores(scores=scores,
                                                    eps=1e-2,
                                                    dataset=dataset)
        >>> result['inconsistency']
        # True

    """
    if dataset.get('aggregation', 'rom') != 'rom':
        raise ValueError(f'the aggregation {dataset.get("aggregation")} specified '\
                        'in the dataset specification is not suitable for this test, '\
                        'consider removing the mode of aggregation or specify "rom".')

    # adjusting the dataset specification to ensure the aggregation is set
    dataset = dataset | {'aggregation' : 'rom'}

    if dataset.get('score_bounds') is not None:
        raise ValueError('it is unnecessary to set score bounds at the dataset level, '\
                            'the scores are implicitly bounded by the numerical uncertainty')

    # creating the experiment consisting of one single dataset, the
    # outer level aggregation can be arbitrary
    experiment = Experiment(datasets=[dataset],
                            aggregation='rom')

    # calculating the raw tp, tn, p and n figures
    figures = experiment.calculate_figures()

    # executing the individual tests
    ind_results = check_scores_tptn(scores=scores,
                                    eps=eps,
                                    p=figures['p'],
                                    n=figures['n'],
                                    numerical_tolerance=numerical_tolerance)

    result = {'inconsistency': ind_results['inconsistency'],
                'individual_results': ind_results}

    if not experiment.has_downstream_bounds():
        logger.info('In the lack of score bounds, the linear programming '\
                    'based aggregated test is not considered.')
        return result

    warnings.warn('It is not a common situation that score bounds '\
                        'can be specified with ratio-of-means aggregation '\
                        'on one single dataset. Please double check the '\
                        'configuration.')

    # for the aggregated figures the original dataset is constructed
    # the difference is that for this step the folding structure is not
    # arbitrary

    agg_results = check_aggregated_scores(experiment=experiment,
                                        scores=scores,
                                        eps=eps,
                                        solver_name=solver_name,
                                        timeout=timeout,
                                        verbosity=verbosity,
                                        numerical_tolerance=numerical_tolerance)

    result['aggregated_results'] = agg_results
    result['inconsistency'] = result['inconsistency'] or agg_results['inconsistency']

    return result

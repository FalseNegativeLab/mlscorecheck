"""
This module implements the top level check function for
scores calculated by the ratio of means aggregation
in a kfold scenarios and mean of ratios aggregation on multiple datastes.
"""

from ..aggregated import check_aggregated_scores, Experiment
from ..core import NUMERICAL_TOLERANCE

__all__ = ['check_n_datasets_mor_kfold_rom_scores']

def check_n_datasets_mor_kfold_rom_scores(scores,
                                            eps,
                                            datasets,
                                            *,
                                            solver_name=None,
                                            timeout=None,
                                            verbosity=1,
                                            numerical_tolerance=NUMERICAL_TOLERANCE):
    """
    Checking the consistency of scores calculated by applying k-fold
    cross validation to multiple datasets and aggregating the figures
    over the folds in the ratio of means fashion and over the datasets
    in the mean of ratios fashion. This aggregated check can be applied
    only if some of the acc, sens, spec and bacc scores are provided.

    Args:
        scores (dict(str,float)): the scores to check
        eps (float/dict(str,float)): the numerical uncertainty(ies) of the scores
        datasets (list): the dataset specification
        solver_name (None/str): the solver to use
        timeout (None/int): the timeout for the linear programming solver in seconds
        verbosity (int): the verbosity of the linear programming solver,
                            0: silent, 1: verbose.
        numerical_tolerance (float): in practice, beyond the numerical uncertainty of
                                    the scores, some further tolerance is applied. This is
                                    orders of magnitude smaller than the uncertainty of the
                                    scores. It does ensure that the specificity of the test
                                    is 1, it might slightly decrease the sensitivity.

    Returns:
        dict: the dictionary of the results of the analysis, the
                'inconsistency' entry indicates if inconsistencies have
                been found. The aggregated_results entry is empty if
                the execution of the linear programming based check was
                unnecessary. The result has four more keys. Under 'lp_status'
                one finds the status of the lp solver, under 'lp_configuration_scores_match'
                one finds a flag indicating if the scores from the lp configuration
                match the scores provided, 'lp_configuration_bounds_match' indicates
                if the specified bounds match the actual figures and finally
                'lp_configuration' contains the actual configuration of the
                linear programming solver.

    Raises:
        ValueError: if the problem is not specified properly

    Examples:
        datasets = [{'p': 39,
                    'n': 822,
                    'n_folds': 8,
                    'n_repeats': 4,
                    'folding': 'stratified_sklearn'},
                    {'name': 'common_datasets.winequality-white-3_vs_7',
                    'n_folds': 3,
                    'n_repeats': 3,
                    'folding': 'stratified_sklearn'}]
        scores = {'acc': 0.548, 'sens': 0.593, 'spec': 0.546, 'bacc': 0.569}

        result = check_n_datasets_mor_kfold_rom_scores(datasets=datasets,
                                                eps=1e-3,
                                                scores=scores)
        result['inconsistency']

        >> False

        datasets = [{'folds': [{'p': 22, 'n': 90},
                                {'p': 51, 'n': 45},
                                {'p': 78, 'n': 34},
                                {'p': 33, 'n': 89}]},
                    {'name': 'common_datasets.yeast-1-2-8-9_vs_7',
                    'n_folds': 8,
                    'n_repeats': 4,
                    'folding': 'stratified_sklearn'}]
        scores = {'acc': 0.552, 'sens': 0.555, 'spec': 0.556, 'bacc': 0.555}

        result = check_n_datasets_mor_kfold_rom_scores(datasets=datasets,
                                                eps=1e-3,
                                                scores=scores)
        result['inconsistency']

        >> False

        datasets = [{'folds': [{'p': 22, 'n': 90},
                        {'p': 51, 'n': 45},
                        {'p': 78, 'n': 34},
                        {'p': 33, 'n': 89}],
                    'fold_score_bounds': {'acc': (0.8, 1.0)},
                    'score_bounds': {'acc': (0.8, 1.0)}
                    },
                    {'name': 'common_datasets.yeast-1-2-8-9_vs_7',
                    'n_folds': 8,
                    'n_repeats': 4,
                    'folding': 'stratified_sklearn',
                    'fold_score_bounds': {'acc': (0.8, 1.0)},
                    'score_bounds': {'acc': (0.8, 1.0)}
                    }]
        scores = {'acc': 0.552, 'sens': 0.555, 'spec': 0.556, 'bacc': 0.555}

        result = check_n_datasets_mor_kfold_rom_scores(datasets=datasets,
                                                eps=1e-3,
                                                scores=scores)
        result['inconsistency']

        >> True
    """
    if any(dataset.get('aggregation', 'rom') != 'rom' for dataset in datasets):
        raise ValueError('the aggregation specified in each dataset must be "rom" or nothing.')

    # adjusting the dataset specification to ensure the aggregation is set
    for dataset in datasets:
        dataset['aggregation'] = 'rom'

    # creating the experiment consisting of one single dataset, the
    # outer level aggregation can be arbitrary
    experiment = Experiment(datasets=datasets,
                            aggregation='mor')

    return check_aggregated_scores(experiment=experiment,
                                        scores=scores,
                                        eps=eps,
                                        solver_name=solver_name,
                                        timeout=timeout,
                                        verbosity=verbosity,
                                        numerical_tolerance=numerical_tolerance)

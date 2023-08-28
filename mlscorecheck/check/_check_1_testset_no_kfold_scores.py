"""
This module implements the top level check function for
scores calculated from raw figures
"""

import warnings

from ..core import logger
from ..individual import check_individual_scores
from ..experiments import dataset_statistics

__all__ = ['check_1_testset_no_kfold_scores']

def check_1_testset_no_kfold_scores(scores,
                                    eps,
                                    testset):
    """
    Use this check if the scores are calculated on one single test set
    with no kfolding or aggregation over multiple datasets.

    Args:
        scores (dict(str,float)): the scores to check ('acc', 'sens', 'spec',
                                    'bacc', 'npv', 'ppv', 'f1', 'fm')
        eps (float/dict(str,float)): the numerical uncertainty (potentially for each score)
        testset (dict): the specification of a testset with p, n or its name

    Returns:
        dict: a dictionary containing the details of the analysis,
                the boolean 'inconsistency' attribute indicates if inconsistency
                was found

    Examples:
        check_1_testset_no_kfold_scores(
            scores={'acc': 0.954, 'sens': 0.934, 'spec': 0.98},
            eps={'acc': 1e-3, 'sens': 1e-3, 'spec': 1e-2},
            testset={'p': 10, 'n': 20}
        )

        check_1_testset_no_kfold_scores(
            scores={'acc': 0.954, 'sens': 0.934, 'spec': 0.985, 'ppv': 0.901},
            eps=1e-3,
            testset={'name': 'common_datasets.ADA'}
        )
    """
    logger.info('Use this function if the scores originate from the '\
                'tp and tn statistics calculated on one test set with '\
                'no aggregation of any kind.')

    if ('p' not in testset or 'n' not in testset) and ('name' not in testset):
        raise ValueError('either "p" and "n" or "name" should be specified')

    if ('n_repeats' in testset) or ('n_folds' in testset) \
        or ('folds' in testset) or ('aggregation' in testset):
        warnings.warn('Additional fields beyond ("p", "n") or "name" present ' \
                        'in the specification, you might want to use another check '\
                        'function specialized to datasets (e.g. check_kfold_mor_scores)')

    p = testset.get('p')
    n = testset.get('n')
    if 'name' in testset:
        p = dataset_statistics[testset['name']]['p']
        n = dataset_statistics[testset['name']]['n']

    logger.info('calling the score check with scores %s, uncertainty %s, p %d and n %d',
                str(scores), str(eps), p, n)

    return check_individual_scores(scores, eps=eps, p=p, n=n)

"""
This module implements the top level check functions
"""

from ..core import check
from ..core import consistency_1, consistency_grouped
from ..datasets import lookup_dataset

def _determine_p_n(p=None, n=None, dataset=None):
    if dataset is not None:
        ds = lookup_dataset(dataset)
        if ds is None:
            raise ValueError(f'dataset {dataset} is not available')
        return ds['p'], ds['n']
    return p, n

def _resolve_p_n(dataset_conf):
    if isinstance(dataset_conf, dict):
        result = {**dataset_conf}
        if result.get('dataset') is not None:
            tmp = lookup_dataset(result['dataset'])
            result['p'] = tmp['p']
            result['n'] = tmp['n']
    elif isinstance(dataset_conf, list):
        result = [_resolve_p_n(dataset) for dataset in dataset_conf]

    return result

def check_scores(scores,
                    eps,
                    dataset,
                    *,
                    return_details=False):
    """
    Use for one dataset - based on the solution formulas
    """
    dataset = _resolve_p_n(dataset)

    results = check(scores, eps=eps, p=dataset['p'], n=dataset['n'])

    if return_details:
        return results['overall_consistency'], results

    return results['overall_consistency']

def check_kfold_rom_scores(scores,
                            eps,
                            *,
                            dataset_setup,
                            return_details=False):
    """
    Use when scores are aggregated in the RoM manner
    """
    dataset_resolved = _resolve_p_n(dataset_setup)
    dataset_resolved['p'] = dataset_resolved['p'] * dataset_resolved['n_repeats']
    dataset_resolved['n'] = dataset_resolved['n'] * dataset_resolved['n_repeats']

    return check_scores(scores,
                        eps,
                        dataset=dataset_resolved,
                        return_details=return_details)

def check_multiple_datasets_rom_scores(scores,
                                        eps,
                                        dataset_setup,
                                        *,
                                        return_details=False):
    datasets = _resolve_p_n(dataset_setup)
    tmp = {'p': sum(dataset['p'] for dataset in datasets),
            'n': sum(dataset['n'] for dataset in datasets)}

    return check_scores(scores,
                        eps,
                        tmp,
                        return_details=return_details)

def check_multiple_datasets_rom_kfold_rom_scores(scores,
                                                eps,
                                                setup,
                                                *,
                                                return_details=False):
    """
    Use when scores are aggregated in the RoM manner
    """
    datasets = _resolve_p_n(setup)
    tmp = {'p': sum(dataset['p']*dataset['n_repeats'] for dataset in datasets),
            'n': sum(dataset['n']*dataset['n_repeats'] for dataset in datasets)}

    return check_scores(scores,
                            eps,
                            tmp,
                            return_details=return_details)

def check_kfold_mor_scores(scores,
                            eps,
                            fold_setup,
                            *,
                            return_details=False):
    """
    Use when a scores on a dataset are aggregated in the MoR manner

    By default, assumes stratified k-fold. If fold_setup is provided,
    it does overwrite all other arguments.
    """

    return consistency_1(scores=scores,
                            eps=eps,
                            problem=fold_setup,
                            return_details=return_details)

def check_multiple_dataset_mor_scores(scores,
                                        eps,
                                        dataset_setup,
                                        *,
                                        return_details=False):
    return check_kfold_mor_scores(scores=scores,
                                    eps=eps,
                                    fold_setup={'fold_configuration': dataset_setup},
                                    return_details=return_details)

def check_multiple_dataset_mor_kfold_rom_scores(scores,
                                                eps,
                                                dataset_setup,
                                                *,
                                                return_details=False):
    fold_configuration = []

    for dataset in dataset_setup:
        dataset = _resolve_p_n(dataset)
        n_repeats = dataset['n_repeats']
        fold_configuration.append({'p': dataset['p']*n_repeats,
                                    'n': dataset['n']*n_repeats,
                                    'score_bounds': dataset.get('score_bounds'),
                                    'tptn_bounds': dataset.get('tptn_bounds')})

    return check_kfold_mor_scores(scores=scores,
                                    eps=eps,
                                    fold_setup={'fold_configuration': fold_configuration},
                                    return_details=return_details)

def check_multiple_dataset_mor_kfold_mor_scores(scores,
                                                eps,
                                                dataset_setup,
                                                *,
                                                return_details=False):

    return consistency_grouped(scores=scores,
                                eps=eps,
                                problems=dataset_setup,
                                return_details=return_details)


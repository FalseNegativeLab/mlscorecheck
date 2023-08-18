"""
This module implements the top level check functions
"""

from ..core import check
from ..core import consistency_1, consistency_grouped
from ..datasets import lookup_dataset

__all__ = ['_resolve_p_n',
            'check_scores',
            'check_kfold_rom_scores',
            'check_multiple_datasets_rom_scores',
            'check_multiple_datasets_rom_kfold_rom_scores',
            'check_kfold_mor_scores',
            'check_multiple_datasets_mor_scores',
            'check_multiple_datasets_mor_kfold_rom_scores',
            'check_multiple_datasets_mor_kfold_mor_scores',
            '_check_problem_specification',
            'validate_problem_specification']

def _resolve_p_n(dataset_conf):
    """
    Resolve the dataset configuration from the integrated statistics

    Args:
        dataset_conf (dict/list(dict)): one or multiple dataset specification(s)
                                with 'dataset' field(s) containing the name of
                                the dataset(s)

    Returns:
        dict: the dataset configuration extended by the 'p' and 'n' figures
    """
    if isinstance(dataset_conf, dict):
        result = {**dataset_conf}
        if result.get('dataset') is not None:
            tmp = lookup_dataset(result['dataset'])
            result['p'] = tmp['p']
            result['n'] = tmp['n']
    elif isinstance(dataset_conf, list):
        result = [_resolve_p_n(dataset) for dataset in dataset_conf]

    return result

def _check_problem_specification(problem):
    """
    Check the problem specification
    """
    if isinstance(problem, dict):
        if not (('p' in problem and 'n' in problem) or ('dataset' in problem) or ('fold_configuration' in problem)):
            return False
        if 'fold_configuration' in problem:
            if not all('p' in fold and 'n' in fold for fold in problem['fold_configuration']):
                return False
        return True

    return all(_check_problem_specification(prob) for prob in problem)

def validate_problem_specification(problem):
    if not _check_problem_specification(problem):
        raise ValueError('the problem specification is incorrect')

def check_scores(scores,
                    eps,
                    dataset,
                    *,
                    return_details=False):
    """
    Use for one dataset - based on the solution formulas
    """
    validate_problem_specification(dataset)

    dataset = _resolve_p_n(dataset)

    results = check(scores, eps=eps, p=dataset['p'], n=dataset['n'])

    if return_details:
        return results['overall_consistency'], results

    return results['overall_consistency']

def check_kfold_rom_scores(scores,
                            eps,
                            dataset,
                            *,
                            return_details=False):
    """
    Use when scores are aggregated in the RoM manner
    """
    validate_problem_specification(dataset)

    dataset_resolved = _resolve_p_n(dataset)
    dataset_resolved['p'] = dataset_resolved['p'] * dataset_resolved['n_repeats']
    dataset_resolved['n'] = dataset_resolved['n'] * dataset_resolved['n_repeats']

    return check_scores(scores,
                        eps,
                        dataset=dataset_resolved,
                        return_details=return_details)

def check_multiple_datasets_rom_scores(scores,
                                        eps,
                                        datasets,
                                        *,
                                        return_details=False):
    validate_problem_specification(datasets)

    datasets = _resolve_p_n(datasets)
    tmp = {'p': sum(dataset['p'] for dataset in datasets),
            'n': sum(dataset['n'] for dataset in datasets)}

    return check_scores(scores,
                        eps,
                        tmp,
                        return_details=return_details)

def check_multiple_datasets_rom_kfold_rom_scores(scores,
                                                eps,
                                                datasets,
                                                *,
                                                return_details=False):
    """
    Use when scores are aggregated in the RoM manner
    """
    validate_problem_specification(datasets)

    datasets = _resolve_p_n(datasets)
    tmp = {'p': sum(dataset['p']*dataset['n_repeats'] for dataset in datasets),
            'n': sum(dataset['n']*dataset['n_repeats'] for dataset in datasets)}

    return check_scores(scores,
                            eps,
                            tmp,
                            return_details=return_details)

def check_kfold_mor_scores(scores,
                            eps,
                            dataset,
                            *,
                            return_details=False):
    """
    Use when a scores on a dataset are aggregated in the MoR manner

    By default, assumes stratified k-fold. If fold_setup is provided,
    it does overwrite all other arguments.
    """
    validate_problem_specification(dataset)

    dataset = _resolve_p_n(dataset)
    return consistency_1(scores=scores,
                            eps=eps,
                            problem=dataset,
                            return_details=return_details)

def check_multiple_datasets_mor_scores(scores,
                                        eps,
                                        datasets,
                                        *,
                                        return_details=False):
    validate_problem_specification(datasets)

    datasets = _resolve_p_n(datasets)
    return check_kfold_mor_scores(scores=scores,
                                    eps=eps,
                                    dataset={'fold_configuration': datasets},
                                    return_details=return_details)

def check_multiple_datasets_mor_kfold_rom_scores(scores,
                                                eps,
                                                datasets,
                                                *,
                                                return_details=False):
    validate_problem_specification(datasets)

    fold_configuration = []

    for dataset in datasets:
        dataset = _resolve_p_n(dataset)

        if 'fold_configuration' in dataset:
            fold_configuration.append({'p': sum(fold['p'] for fold in dataset['fold_configuration']),
                                        'n': sum(fold['n'] for fold in dataset['fold_configuration']),
                                        'score_bounds': dataset.get('score_bounds'),
                                        'tptn_bounds': dataset.get('tptn_bounds')})
        else:
            n_repeats = dataset['n_repeats']
            fold_configuration.append({'p': dataset['p']*n_repeats,
                                        'n': dataset['n']*n_repeats,
                                        'score_bounds': dataset.get('score_bounds'),
                                        'tptn_bounds': dataset.get('tptn_bounds')})

    return check_kfold_mor_scores(scores=scores,
                                    eps=eps,
                                    dataset={'fold_configuration': fold_configuration},
                                    return_details=return_details)

def check_multiple_datasets_mor_kfold_mor_scores(scores,
                                                eps,
                                                datasets,
                                                *,
                                                return_details=False):
    validate_problem_specification(datasets)

    datasets = _resolve_p_n(datasets)

    return consistency_grouped(scores=scores,
                                eps=eps,
                                problems=datasets,
                                return_details=return_details)


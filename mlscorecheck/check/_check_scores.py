"""
This module implements the top level check functions
"""

from ..individual import check
from ..aggregated import (_expand_datasets, check_aggregated_scores)

__all__ = ['check_scores',
            'check_kfold_rom_scores',
            'check_datasets_rom_scores',
            'check_datasets_rom_kfold_rom_scores',
            'check_kfold_mor_scores',
            'check_datasets_mor_scores',
            'check_datasets_mor_kfold_rom_scores',
            'check_datasets_mor_kfold_mor_scores']

def check_scores(scores,
                    eps,
                    dataset,
                    *,
                    return_details=False):
    """
    Use for one dataset - based on the solution formulas
    """
    validate_problem_specification(dataset)

    dataset = _expand_datasets(dataset)

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
    flag_ind, details_ind = check_scores(scores=scores,
                                eps=eps,
                                dataset=dataset,
                                return_details=True)
    flag_agg, details_agg = check_aggregated_scores(scores=scores,
                                            eps=eps,
                                            datasets=[dataset],
                                            strategy=('rom', 'rom'),
                                            return_details=True)

    details_agg['configuration'] = details_agg['configuration'][0]

    flag = flag_ind and flag_agg
    details = {'consistency': flag,
                'individual': details_ind,
                'aggregated': details_agg}

    return (flag, details) if return_details else flag

def check_multiple_datasets_rom_scores(scores,
                                        eps,
                                        datasets,
                                        *,
                                        return_details=False):
    datasets = _expand_datasets(datasets)

    total_p = sum(dataset['p'] for dataset in datasets)
    total_n = sum(dataset['n'] for dataset in datasets)

    flag_ind, details_ind = check_scores(scores=scores,
                                eps=eps,
                                dataset={'p': total_p, 'n': total_n},
                                return_details=True)
    flag_agg, details_agg = check_aggregated_scores(scores=scores,
                                            eps=eps,
                                            datasets=datasets,
                                            strategy=('rom', 'rom'),
                                            return_details=True)

    flag = flag_ind and flag_agg
    details = {'consistency': flag,
                'individual': details_ind,
                'aggregated': details_agg}

    return (flag, details) if return_details else flag

def check_multiple_datasets_rom_kfold_rom_scores(scores,
                                                eps,
                                                datasets,
                                                *,
                                                return_details=False):
    """
    Use when scores are aggregated in the RoM manner
    """

    datasets = _expand_datasets(datasets)

    total_p = sum(dataset['p'] for dataset in datasets)
    total_n = sum(dataset['n'] for dataset in datasets)

    flag_ind, details_ind = check_scores(scores=scores,
                                eps=eps,
                                dataset={'p': total_p, 'n': total_n},
                                return_details=True)
    flag_agg, details_agg = check_aggregated_scores(scores=scores,
                                            eps=eps,
                                            datasets=datasets,
                                            strategy=('rom', 'rom'),
                                            return_details=True)

    flag = flag_ind and flag_agg
    details = {'consistency': flag,
                'individual': details_ind,
                'aggregated': details_agg}

    return (flag, details) if return_details else flag

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

    flag_agg, details_agg = check_aggregated_scores(scores=scores,
                                            eps=eps,
                                            datasets=[dataset],
                                            strategy=('mor', 'mor'),
                                            return_details=True)
    details_agg['configuration'] = details_agg['configuration'][0]

    return (flag_agg, details_agg) if return_details else flag_agg

def check_multiple_datasets_mor_scores(scores,
                                        eps,
                                        datasets,
                                        *,
                                        return_details=False):
    validate_problem_specification(datasets)

    flag_agg, details_agg = check_aggregated_scores(scores=scores,
                                            eps=eps,
                                            datasets=datasets,
                                            strategy=('mor', 'mor'),
                                            return_details=True)

    return (flag_agg, details_agg) if return_details else flag_agg

def check_multiple_datasets_mor_kfold_rom_scores(scores,
                                                eps,
                                                datasets,
                                                *,
                                                return_details=False):
    validate_problem_specification(datasets)

    flag_agg, details_agg = check_aggregated_scores(scores=scores,
                                            eps=eps,
                                            datasets=datasets,
                                            strategy=('mor', 'rom'),
                                            return_details=True)

    return (flag_agg, details_agg) if return_details else flag_agg

def check_multiple_datasets_mor_kfold_mor_scores(scores,
                                                eps,
                                                datasets,
                                                *,
                                                return_details=False):
    validate_problem_specification(datasets)

    flag_agg, details_agg = check_aggregated_scores(scores=scores,
                                            eps=eps,
                                            datasets=datasets,
                                            strategy=('mor', 'mor'),
                                            return_details=True)

    return (flag_agg, details_agg) if return_details else flag_agg

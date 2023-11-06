"""
This module implements the multiclass scores
"""

import math

import numpy as np

from ..core import safe_call

from ._scores_standardized import (accuracy_standardized,
                                    sensitivity_standardized,
                                    specificity_standardized,
                                    balanced_accuracy_standardized,
                                    positive_predictive_value_standardized,
                                    negative_predictive_value_standardized,
                                    f_beta_positive_standardized,
                                    f_beta_negative_standardized,
                                    f1_positive_standardized,
                                    f1_negative_standardized,
                                    unified_performance_measure_standardized,
                                    geometric_mean_standardized,
                                    fowlkes_mallows_index_standardized,
                                    markedness_standardized,
                                    positive_likelihood_ratio_standardized,
                                    negative_likelihood_ratio_standardized,
                                    matthews_correlation_coefficient_standardized,
                                    bookmaker_informedness_standardized,
                                    prevalence_threshold_standardized,
                                    diagnostic_odds_ratio_standardized,
                                    jaccard_index_standardized,
                                    cohens_kappa_standardized)
from ._score_bundles import score_specifications

__all__ = ['multiclass_score_macro',
            'multiclass_score_micro',
            'multiclass_score_weighted',
            'multiclass_score',
            'multiclass_accuracy',
            'multiclass_sensitivity',
            'multiclass_specificity',
            'multiclass_balanced_accuracy',
            'multiclass_positive_predictive_value',
            'multiclass_negative_predictive_value',
            'multiclass_f_beta_positive',
            'multiclass_f_beta_negative',
            'multiclass_f1_positive',
            'multiclass_f1_negative',
            'multiclass_unified_performance_measure',
            'multiclass_geometric_mean',
            'multiclass_fowlkes_mallows_index',
            'multiclass_markedness',
            'multiclass_positive_likelihood_ratio',
            'multiclass_negative_likelihood_ratio',
            'multiclass_matthews_correlation_coefficient',
            'multiclass_bookmaker_informedness',
            'multiclass_prevalence_threshold',
            'multiclass_diagnostic_odds_ratio',
            'multiclass_jaccard_index',
            'multiclass_cohens_kappa',
            'multiclass_score_map']

def multiclass_score_macro(confusion_matrix,
                            score_function,
                            additional_params=None,
                            name=None):
    """
    Calculates the multiclass macro average score

    Args:
        confusion_matrix (np.array): the confusion matrix (true labels in rows)
        score_function (callable): the score function to use
        additional_params (None|dict): the additional parameters to use
        name (None|str): the name of the score

    Returns:
        float: the score
    """
    additional_params = {} if additional_params is None else additional_params
    additional_params = additional_params | {'sqrt': math.sqrt}

    counts = np.sum(confusion_matrix, axis=1)
    n_total = np.sum(counts)

    scores = [safe_call(score_function, {'p': count,
                                    'n': n_total - count,
                                    'tp': confusion_matrix[idx, idx],
                                    'tn': np.sum(confusion_matrix[0:idx, 0:idx]) \
                                        + np.sum(confusion_matrix[idx+1:, idx+1:]) \
                                        + np.sum(confusion_matrix[:idx, idx+1:]) \
                                        + np.sum(confusion_matrix[idx+1:, :idx])}
                        | additional_params,
                        score_specifications[name].get('nans'))
                for idx, count in enumerate(counts)]

    return np.mean(scores)

def multiclass_score_micro(confusion_matrix,
                            score_function,
                            additional_params=None,
                            name=None):
    """
    Calculates the multiclass micro average score

    Args:
        confusion_matrix (np.array): the confusion matrix (true labels in rows)
        score_function (callable): the score function to use
        additional_params (None|dict): the additional parameters to use
        name (None|str): the name of the score

    Returns:
        float: the score
    """
    additional_params = {} if additional_params is None else additional_params
    additional_params = additional_params | {'sqrt': math.sqrt}

    counts = np.sum(confusion_matrix, axis=1)
    n_total = np.sum(counts)

    params = {'tp': 0,
                'tn': 0,
                'p': 0,
                'n': 0}

    for idx, count in enumerate(counts):
        params['p'] += count
        params['n'] += n_total - count
        params['tp'] += confusion_matrix[idx, idx]
        params['tn'] += np.sum(confusion_matrix[0:idx, 0:idx]) \
                        + np.sum(confusion_matrix[idx+1:, idx+1:]) \
                        + np.sum(confusion_matrix[:idx, idx+1:]) \
                        + np.sum(confusion_matrix[idx+1:, :idx])

    return safe_call(score_function,
                        params | additional_params,
                        score_specifications[name].get('nans'))

def multiclass_score_weighted(confusion_matrix,
                                score_function,
                                additional_params=None,
                                name=None):
    """
    Calculates the multiclass weighted macro average score

    Args:
        confusion_matrix (np.array): the confusion matrix (true labels in rows)
        score_function (callable): the score function to use
        additional_params (None|dict): the additional parameters to use
        name (None|str): the name of the score

    Returns:
        float: the score
    """
    additional_params = {} if additional_params is None else additional_params
    additional_params = additional_params | {'sqrt': math.sqrt}

    counts = np.sum(confusion_matrix, axis=1)
    n_total = np.sum(counts)

    scores = [safe_call(score_function, {'p': count,
                                    'n': n_total - count,
                                    'tp': confusion_matrix[idx, idx],
                                    'tn': np.sum(confusion_matrix[0:idx, 0:idx]) \
                                        + np.sum(confusion_matrix[idx+1:, idx+1:]) \
                                        + np.sum(confusion_matrix[:idx, idx+1:]) \
                                        + np.sum(confusion_matrix[idx+1:, :idx])}
                        | additional_params,
                        score_specifications[name].get('nans'))\
                * count / n_total
                for idx, count in enumerate(counts)]

    return np.sum(scores)

def multiclass_score(confusion_matrix,
                        score_function,
                        average,
                        additional_params=None,
                        name=None):
    """
    Calculates the multiclass average score

    Args:
        confusion_matrix (np.array): the confusion matrix (true labels in rows)
        score_function (callable): the score function to use
        average (str): the averaging to be used ('macro'/'micro'/'weighted')
        additional_params (None|dict): the additional parameters to use
        name (None|str): the name of the score

    Returns:
        float: the score
    """
    if average == 'micro':
        return multiclass_score_micro(confusion_matrix,
                                        score_function,
                                        additional_params,
                                        name)
    if average == 'macro':
        return multiclass_score_macro(confusion_matrix,
                                        score_function,
                                        additional_params,
                                        name)
    if average == 'weighted':
        return multiclass_score_weighted(confusion_matrix,
                                            score_function,
                                            additional_params,
                                            name)

    raise ValueError(f'averaging {average} is not supported')

def multiclass_accuracy(*, confusion_matrix, average):
    """
    Calculates the multiclass score

    Args:
        confusion_matrix (np.array): the confusion matrix (true labels in rows)
        average (str): the averaging to be used ('macro'/'micro'/'weighted')

    Returns:
        float: the score
    """
    return multiclass_score(confusion_matrix,
                            accuracy_standardized,
                            average,
                            name='acc')

def multiclass_sensitivity(*, confusion_matrix, average):
    """
    Calculates the multiclass score

    Args:
        confusion_matrix (np.array): the confusion matrix (true labels in rows)
        average (str): the averaging to be used ('macro'/'micro'/'weighted')

    Returns:
        float: the score
    """
    return multiclass_score(confusion_matrix,
                            sensitivity_standardized,
                            average,
                            name='sens')

def multiclass_specificity(*, confusion_matrix, average):
    """
    Calculates the multiclass score

    Args:
        confusion_matrix (np.array): the confusion matrix (true labels in rows)
        average (str): the averaging to be used ('macro'/'micro'/'weighted')

    Returns:
        float: the score
    """
    return multiclass_score(confusion_matrix,
                            specificity_standardized,
                            average,
                            name='spec')

def multiclass_balanced_accuracy(*, confusion_matrix, average):
    """
    Calculates the multiclass score

    Args:
        confusion_matrix (np.array): the confusion matrix (true labels in rows)
        average (str): the averaging to be used ('macro'/'micro'/'weighted')

    Returns:
        float: the score
    """
    return multiclass_score(confusion_matrix,
                            balanced_accuracy_standardized,
                            average,
                            name='bacc')

def multiclass_positive_predictive_value(*, confusion_matrix, average):
    """
    Calculates the multiclass score

    Args:
        confusion_matrix (np.array): the confusion matrix (true labels in rows)
        average (str): the averaging to be used ('macro'/'micro'/'weighted')

    Returns:
        float: the score
    """
    return multiclass_score(confusion_matrix,
                            positive_predictive_value_standardized,
                            average,
                            name='ppv')

def multiclass_negative_predictive_value(*, confusion_matrix, average):
    """
    Calculates the multiclass score

    Args:
        confusion_matrix (np.array): the confusion matrix (true labels in rows)
        average (str): the averaging to be used ('macro'/'micro'/'weighted')

    Returns:
        float: the score
    """
    return multiclass_score(confusion_matrix,
                            negative_predictive_value_standardized,
                            average,
                            name='npv')

def multiclass_f_beta_positive(*, beta_positive, confusion_matrix, average):
    """
    Calculates the multiclass score

    Args:
        beta_positive (float): the beta value to be used
        confusion_matrix (np.array): the confusion matrix (true labels in rows)
        average (str): the averaging to be used ('macro'/'micro'/'weighted')

    Returns:
        float: the score
    """
    return multiclass_score(confusion_matrix,
                            f_beta_positive_standardized,
                            average,
                            additional_params={'beta_positive': beta_positive},
                            name='fbp')

def multiclass_f_beta_negative(*, beta_negative, confusion_matrix, average):
    """
    Calculates the multiclass score

    Args:
        beta_negative (float): the beta value to be used
        confusion_matrix (np.array): the confusion matrix (true labels in rows)
        average (str): the averaging to be used ('macro'/'micro'/'weighted')

    Returns:
        float: the score
    """
    return multiclass_score(confusion_matrix,
                            f_beta_negative_standardized,
                            average,
                            additional_params={'beta_negative': beta_negative},
                            name='fbn')

def multiclass_f1_positive(*, confusion_matrix, average):
    """
    Calculates the multiclass score

    Args:
        confusion_matrix (np.array): the confusion matrix (true labels in rows)
        average (str): the averaging to be used ('macro'/'micro'/'weighted')

    Returns:
        float: the score
    """
    return multiclass_score(confusion_matrix,
                            f1_positive_standardized,
                            average,
                            name='f1p')

def multiclass_f1_negative(*, confusion_matrix, average):
    """
    Calculates the multiclass score

    Args:
        confusion_matrix (np.array): the confusion matrix (true labels in rows)
        average (str): the averaging to be used ('macro'/'micro'/'weighted')

    Returns:
        float: the score
    """
    return multiclass_score(confusion_matrix,
                            f1_negative_standardized,
                            average,
                            name='f1n')

def multiclass_unified_performance_measure(*, confusion_matrix, average):
    """
    Calculates the multiclass score

    Args:
        confusion_matrix (np.array): the confusion matrix (true labels in rows)
        average (str): the averaging to be used ('macro'/'micro'/'weighted')

    Returns:
        float: the score
    """
    return multiclass_score(confusion_matrix,
                            unified_performance_measure_standardized,
                            average,
                            name='upm')

def multiclass_geometric_mean(*, confusion_matrix, average):
    """
    Calculates the multiclass score

    Args:
        confusion_matrix (np.array): the confusion matrix (true labels in rows)
        average (str): the averaging to be used ('macro'/'micro'/'weighted')

    Returns:
        float: the score
    """
    return multiclass_score(confusion_matrix,
                            geometric_mean_standardized,
                            average,
                            name='gm')

def multiclass_fowlkes_mallows_index(*, confusion_matrix, average):
    """
    Calculates the multiclass score

    Args:
        confusion_matrix (np.array): the confusion matrix (true labels in rows)
        average (str): the averaging to be used ('macro'/'micro'/'weighted')

    Returns:
        float: the score
    """
    return multiclass_score(confusion_matrix,
                            fowlkes_mallows_index_standardized,
                            average,
                            name='fm')

def multiclass_markedness(*, confusion_matrix, average):
    """
    Calculates the multiclass score

    Args:
        confusion_matrix (np.array): the confusion matrix (true labels in rows)
        average (str): the averaging to be used ('macro'/'micro'/'weighted')

    Returns:
        float: the score
    """
    return multiclass_score(confusion_matrix,
                            markedness_standardized,
                            average,
                            name='mk')

def multiclass_positive_likelihood_ratio(*, confusion_matrix, average):
    """
    Calculates the multiclass score

    Args:
        confusion_matrix (np.array): the confusion matrix (true labels in rows)
        average (str): the averaging to be used ('macro'/'micro'/'weighted')

    Returns:
        float: the score
    """
    return multiclass_score(confusion_matrix,
                            positive_likelihood_ratio_standardized,
                            average,
                            name='lrp')

def multiclass_negative_likelihood_ratio(*, confusion_matrix, average):
    """
    Calculates the multiclass score

    Args:
        confusion_matrix (np.array): the confusion matrix (true labels in rows)
        average (str): the averaging to be used ('macro'/'micro'/'weighted')

    Returns:
        float: the score
    """
    return multiclass_score(confusion_matrix,
                            negative_likelihood_ratio_standardized,
                            average,
                            name='lrn')

def multiclass_matthews_correlation_coefficient(*, confusion_matrix, average):
    """
    Calculates the multiclass score

    Args:
        confusion_matrix (np.array): the confusion matrix (true labels in rows)
        average (str): the averaging to be used ('macro'/'micro'/'weighted')

    Returns:
        float: the score
    """
    return multiclass_score(confusion_matrix,
                            matthews_correlation_coefficient_standardized,
                            average,
                            name='mcc')

def multiclass_bookmaker_informedness(*, confusion_matrix, average):
    """
    Calculates the multiclass score

    Args:
        confusion_matrix (np.array): the confusion matrix (true labels in rows)
        average (str): the averaging to be used ('macro'/'micro'/'weighted')

    Returns:
        float: the score
    """
    return multiclass_score(confusion_matrix,
                            bookmaker_informedness_standardized,
                            average,
                            name='bm')

def multiclass_prevalence_threshold(*, confusion_matrix, average):
    """
    Calculates the multiclass score

    Args:
        confusion_matrix (np.array): the confusion matrix (true labels in rows)
        average (str): the averaging to be used ('macro'/'micro'/'weighted')

    Returns:
        float: the score
    """
    return multiclass_score(confusion_matrix,
                            prevalence_threshold_standardized,
                            average,
                            name='pt')

def multiclass_diagnostic_odds_ratio(*, confusion_matrix, average):
    """
    Calculates the multiclass score

    Args:
        confusion_matrix (np.array): the confusion matrix (true labels in rows)
        average (str): the averaging to be used ('macro'/'micro'/'weighted')

    Returns:
        float: the score
    """
    return multiclass_score(confusion_matrix,
                            diagnostic_odds_ratio_standardized,
                            average,
                            name='dor')

def multiclass_jaccard_index(*, confusion_matrix, average):
    """
    Calculates the multiclass score

    Args:
        confusion_matrix (np.array): the confusion matrix (true labels in rows)
        average (str): the averaging to be used ('macro'/'micro'/'weighted')

    Returns:
        float: the score
    """
    return multiclass_score(confusion_matrix,
                            jaccard_index_standardized,
                            average,
                            name='ji')

def multiclass_cohens_kappa(*, confusion_matrix, average):
    """
    Calculates the multiclass score

    Args:
        confusion_matrix (np.array): the confusion matrix (true labels in rows)
        average (str): the averaging to be used ('macro'/'micro'/'weighted')

    Returns:
        float: the score
    """
    return multiclass_score(confusion_matrix,
                            cohens_kappa_standardized,
                            average,
                            name='kappa')

multiclass_score_map = {'acc': multiclass_accuracy,
                        'sens': multiclass_sensitivity,
                        'spec': multiclass_specificity,
                        'bacc': multiclass_balanced_accuracy,
                        'ppv': multiclass_positive_predictive_value,
                        'npv': multiclass_negative_predictive_value,
                        'fbp': multiclass_f_beta_positive,
                        'fbn': multiclass_f_beta_negative,
                        'f1p': multiclass_f1_positive,
                        'f1n': multiclass_f1_negative,
                        'upm': multiclass_unified_performance_measure,
                        'gm': multiclass_geometric_mean,
                        'fm': multiclass_fowlkes_mallows_index,
                        'mk': multiclass_markedness,
                        'lrp': multiclass_positive_likelihood_ratio,
                        'lrn': multiclass_negative_likelihood_ratio,
                        'mcc': multiclass_matthews_correlation_coefficient,
                        'bm': multiclass_bookmaker_informedness,
                        'pt': multiclass_prevalence_threshold,
                        'dor': multiclass_diagnostic_odds_ratio,
                        'ji': multiclass_jaccard_index,
                        'kappa': multiclass_cohens_kappa}

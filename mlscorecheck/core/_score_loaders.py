"""
This module implements some loader functions for the scores
"""

from ._scores import *
from ._scores_standardized import *
from ._solutions import load_scores

__all__ = ['score_functions',
            'score_functions_standardized',
            'score_function_aliases',
            'score_function_complementers']

def score_functions(complements=False):
    """
    Return a set of scores with no aliases

    Returns:
        dict: the scores with no aliases
    """
    if complements:
        comp = {'err': error_rate,
                'fdr': false_discovery_rate,
                'for_': false_omission_rate,
                'fnr': false_negative_rate,
                'fpr': false_positive_rate}
    else:
        comp = {}

    return {'acc': accuracy,
            'sens': sensitivity,
            'spec': specificity,
            'npv': negative_predictive_value,
            'ppv': positive_predictive_value,
            'f1p': f1_plus,
            'f1m': f1_minus,
            'fbp': f_beta_plus,
            'fbm': f_beta_minus,
            'bacc': balanced_accuracy,
            'fm': fowlkes_mallows_index,
            'bm': bookmaker_informedness,
            'pt': prevalence_threshold,
            'lrp': positive_likelihood_ratio,
            'lrn': negative_likelihood_ratio,
            'mk': markedness,
            'dor': diagnostic_odds_ratio,
            'mcc': matthews_correlation_coefficient,
            'ji': jaccard_index,
            'kappa': cohens_kappa,
            'gm': geometric_mean,
            'upm': unified_performance_measure,
            'p4': p4,
            **comp}

def score_functions_standardized(complements=False):
    """
    Return a set of scores with no aliases

    Returns:
        dict: the scores with no aliases
    """
    if complements:
        comp = {'err': error_rate_standardized,
                'fdr': false_discovery_rate_standardized,
                'for_': false_omission_rate_standardized,
                'fnr': false_negative_rate_standardized,
                'fpr': false_positive_rate_standardized}
    else:
        comp = {}

    return {'acc': accuracy_standardized,
            'sens': sensitivity_standardized,
            'spec': specificity_standardized,
            'npv': negative_predictive_value_standardized,
            'ppv': positive_predictive_value_standardized,
            'f1p': f1_plus_standardized,
            'f1m': f1_minus_standardized,
            'fbp': f_beta_plus_standardized,
            'fbm': f_beta_minus_standardized,
            'bacc': balanced_accuracy_standardized,
            'fm': fowlkes_mallows_index_standardized,
            'bm': bookmaker_informedness_standardized,
            'pt': prevalence_threshold_standardized,
            'lrp': positive_likelihood_ratio_standardized,
            'lrn': negative_likelihood_ratio_standardized,
            'mk': markedness_standardized,
            'dor': diagnostic_odds_ratio_standardized,
            'mcc': matthews_correlation_coefficient_standardized,
            'ji': jaccard_index_standardized,
            'kappa': cohens_kappa_standardized,
            'gm': geometric_mean_standardized,
            'upm': unified_performance_measure_standardized,
            'p4': p4_standardized,
            **comp}

def score_function_aliases():
    """
    Returns the alias mapping

    Returns:
        dict: the alias mapping
    """
    return {'tpr': 'sens',
            'tnr': 'spec',
            'prec': 'ppv',
            'rec': 'sens'}

def score_function_complementers():
    """
    Returns the complementer mapping

    Returns:
        dict: the complementer mapping
    """
    return {'fdr': 'ppv',
            'for_': 'npv',
            'fnr': 'tpr',
            'fpr': 'tnr',
            'err': 'acc'}

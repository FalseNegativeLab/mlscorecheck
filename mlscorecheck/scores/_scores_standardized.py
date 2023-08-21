"""
This module implements the scores with standardized parameterizations
This is a generated file, do not modify it.
"""

import numpy as np

__all__ = [
'accuracy_standardized',
'error_rate_standardized',
'sensitivity_standardized',
'false_negative_rate_standardized',
'false_positive_rate_standardized',
'specificity_standardized',
'positive_predictive_value_standardized',
'false_discovery_rate_standardized',
'false_omission_rate_standardized',
'negative_predictive_value_standardized',
'f_beta_plus_standardized',
'f_beta_minus_standardized',
'f1_plus_standardized',
'f1_minus_standardized',
'unified_performance_measure_standardized',
'geometric_mean_standardized',
'fowlkes_mallows_index_standardized',
'markedness_standardized',
'positive_likelihood_ratio_standardized',
'negative_likelihood_ratio_standardized',
'matthews_correlation_coefficient_standardized',
'bookmaker_informedness_standardized',
'prevalence_threshold_standardized',
'diagnostic_odds_ratio_standardized',
'jaccard_index_standardized',
'balanced_accuracy_standardized',
'cohens_kappa_standardized',
'p4_standardized']

def accuracy_standardized(*, tp, tn, p, n):
    """
    The standardized accuracy score

    Args:
        tp (int/float/np.array/Interval): The number of true positives
        tn (int/float/np.array/Interval): The number of true negatives
        p (int/float/np.array/Interval): The number of positives
        n (int/float/np.array/Interval): The number of negatives

    Returns:
        int/float/np.array/Interval: the score
    """
    return (tp + tn)/(p + n)

def error_rate_standardized(*, tp, tn, p, n):
    """
    The standardized error_rate score

    Args:
        tp (int/float/np.array/Interval): The number of true positives
        tn (int/float/np.array/Interval): The number of true negatives
        p (int/float/np.array/Interval): The number of positives
        n (int/float/np.array/Interval): The number of negatives

    Returns:
        int/float/np.array/Interval: the score
    """
    return 1 - (tp + tn)/(p + n)

def sensitivity_standardized(*, tp, p):
    """
    The standardized sensitivity score

    Args:
        tp (int/float/np.array/Interval): The number of true positives
        p (int/float/np.array/Interval): The number of positives

    Returns:
        int/float/np.array/Interval: the score
    """
    return tp/p

def false_negative_rate_standardized(*, tp, p):
    """
    The standardized false_negative_rate score

    Args:
        tp (int/float/np.array/Interval): The number of true positives
        p (int/float/np.array/Interval): The number of positives

    Returns:
        int/float/np.array/Interval: the score
    """
    return 1 - tp/p

def false_positive_rate_standardized(*, tn, n):
    """
    The standardized false_positive_rate score

    Args:
        tn (int/float/np.array/Interval): The number of true negatives
        n (int/float/np.array/Interval): The number of negatives

    Returns:
        int/float/np.array/Interval: the score
    """
    return 1 - tn/n

def specificity_standardized(*, tn, n):
    """
    The standardized specificity score

    Args:
        tn (int/float/np.array/Interval): The number of true negatives
        n (int/float/np.array/Interval): The number of negatives

    Returns:
        int/float/np.array/Interval: the score
    """
    return tn/n

def positive_predictive_value_standardized(*, tp, tn, n):
    """
    The standardized positive_predictive_value score

    Args:
        tp (int/float/np.array/Interval): The number of true positives
        tn (int/float/np.array/Interval): The number of true negatives
        n (int/float/np.array/Interval): The number of negatives

    Returns:
        int/float/np.array/Interval: the score
    """
    return tp/(tp + n - tn)

def false_discovery_rate_standardized(*, tp, tn, n):
    """
    The standardized false_discovery_rate score

    Args:
        tp (int/float/np.array/Interval): The number of true positives
        tn (int/float/np.array/Interval): The number of true negatives
        n (int/float/np.array/Interval): The number of negatives

    Returns:
        int/float/np.array/Interval: the score
    """
    return 1 - tp/(tp + n - tn)

def false_omission_rate_standardized(*, tp, tn, p):
    """
    The standardized false_omission_rate score

    Args:
        tp (int/float/np.array/Interval): The number of true positives
        tn (int/float/np.array/Interval): The number of true negatives
        p (int/float/np.array/Interval): The number of positives

    Returns:
        int/float/np.array/Interval: the score
    """
    return 1 - tn/(tn + p - tp)

def negative_predictive_value_standardized(*, tp, tn, p):
    """
    The standardized negative_predictive_value score

    Args:
        tp (int/float/np.array/Interval): The number of true positives
        tn (int/float/np.array/Interval): The number of true negatives
        p (int/float/np.array/Interval): The number of positives

    Returns:
        int/float/np.array/Interval: the score
    """
    return tn/(tn + p - tp)

def f_beta_plus_standardized(*, tp, tn, p, n, beta_plus):
    """
    The standardized f_beta_plus score

    Args:
        tp (int/float/np.array/Interval): The number of true positives
        tn (int/float/np.array/Interval): The number of true negatives
        p (int/float/np.array/Interval): The number of positives
        n (int/float/np.array/Interval): The number of negatives
        beta_plus (int/float/np.array/Interval): the beta parameter

    Returns:
        int/float/np.array/Interval: the score
    """
    return ((1 + beta_plus**2)*tp) / (tp + beta_plus**2*p + n - tn)

def f_beta_minus_standardized(*, tp, tn, p, n, beta_minus):
    """
    The standardized f_beta_minus score

    Args:
        tp (int/float/np.array/Interval): The number of true positives
        tn (int/float/np.array/Interval): The number of true negatives
        p (int/float/np.array/Interval): The number of positives
        n (int/float/np.array/Interval): The number of negatives
        beta_minus (int/float/np.array/Interval): the beta parameter

    Returns:
        int/float/np.array/Interval: the score
    """
    return ((1 + beta_minus**2)*tn) / (tn + beta_minus**2*n + p - tp)

def f1_plus_standardized(*, tp, tn, p, n):
    """
    The standardized f1_plus score

    Args:
        tp (int/float/np.array/Interval): The number of true positives
        tn (int/float/np.array/Interval): The number of true negatives
        p (int/float/np.array/Interval): The number of positives
        n (int/float/np.array/Interval): The number of negatives

    Returns:
        int/float/np.array/Interval: the score
    """
    return (2*tp) / (tp + p + n - tn)

def f1_minus_standardized(*, tp, tn, p, n):
    """
    The standardized f1_minus score

    Args:
        tp (int/float/np.array/Interval): The number of true positives
        tn (int/float/np.array/Interval): The number of true negatives
        p (int/float/np.array/Interval): The number of positives
        n (int/float/np.array/Interval): The number of negatives

    Returns:
        int/float/np.array/Interval: the score
    """
    return (2*tn) / (tn + n + p - tp)

def unified_performance_measure_standardized(*, tp, tn, p, n):
    """
    The standardized unified_performance_measure score

    Args:
        tp (int/float/np.array/Interval): The number of true positives
        tn (int/float/np.array/Interval): The number of true negatives
        p (int/float/np.array/Interval): The number of positives
        n (int/float/np.array/Interval): The number of negatives

    Returns:
        int/float/np.array/Interval: the score
    """
    return 4*tn*tp/(tn*(n + p - tn + tp) + tp*(n + p + tn - tp))

def geometric_mean_standardized(*, tp, tn, p, n, sqrt=np.sqrt):
    """
    The standardized geometric_mean score

    Args:
        tp (int/float/np.array/Interval): The number of true positives
        tn (int/float/np.array/Interval): The number of true negatives
        p (int/float/np.array/Interval): The number of positives
        n (int/float/np.array/Interval): The number of negatives

    Returns:
        int/float/np.array/Interval: the score
    """
    return sqrt(tp)*sqrt(tn)/(sqrt(p)*sqrt(n))

def fowlkes_mallows_index_standardized(*, tp, tn, p, n, sqrt=np.sqrt):
    """
    The standardized fowlkes_mallows_index score

    Args:
        tp (int/float/np.array/Interval): The number of true positives
        tn (int/float/np.array/Interval): The number of true negatives
        p (int/float/np.array/Interval): The number of positives
        n (int/float/np.array/Interval): The number of negatives

    Returns:
        int/float/np.array/Interval: the score
    """
    return tp/sqrt(p*(n - tn + tp))

def markedness_standardized(*, tp, tn, p, n):
    """
    The standardized markedness score

    Args:
        tp (int/float/np.array/Interval): The number of true positives
        tn (int/float/np.array/Interval): The number of true negatives
        p (int/float/np.array/Interval): The number of positives
        n (int/float/np.array/Interval): The number of negatives

    Returns:
        int/float/np.array/Interval: the score
    """
    return tp/(tp + n - tn) + tn/(tn + p - tp) - 1

def positive_likelihood_ratio_standardized(*, tp, tn, p, n):
    """
    The standardized positive_likelihood_ratio score

    Args:
        tp (int/float/np.array/Interval): The number of true positives
        tn (int/float/np.array/Interval): The number of true negatives
        p (int/float/np.array/Interval): The number of positives
        n (int/float/np.array/Interval): The number of negatives

    Returns:
        int/float/np.array/Interval: the score
    """
    return n*tp/((n-tn)*p)

def negative_likelihood_ratio_standardized(*, tp, tn, p, n):
    """
    The standardized negative_likelihood_ratio score

    Args:
        tp (int/float/np.array/Interval): The number of true positives
        tn (int/float/np.array/Interval): The number of true negatives
        p (int/float/np.array/Interval): The number of positives
        n (int/float/np.array/Interval): The number of negatives

    Returns:
        int/float/np.array/Interval: the score
    """
    return n*(p - tp)/(tn*p)

def matthews_correlation_coefficient_standardized(*, tp, tn, p, n, sqrt=np.sqrt):
    """
    The standardized matthews_correlation_coefficient score

    Args:
        tp (int/float/np.array/Interval): The number of true positives
        tn (int/float/np.array/Interval): The number of true negatives
        p (int/float/np.array/Interval): The number of positives
        n (int/float/np.array/Interval): The number of negatives

    Returns:
        int/float/np.array/Interval: the score
    """
    return (tn*tp - (n - tn)*(p - tp))/sqrt(n*p*(n - tn + tp)*(p + tn - tp))

def bookmaker_informedness_standardized(*, tp, tn, p, n):
    """
    The standardized bookmaker_informedness score

    Args:
        tp (int/float/np.array/Interval): The number of true positives
        tn (int/float/np.array/Interval): The number of true negatives
        p (int/float/np.array/Interval): The number of positives
        n (int/float/np.array/Interval): The number of negatives

    Returns:
        int/float/np.array/Interval: the score
    """
    return tp/p + tn/n - 1

def prevalence_threshold_standardized(*, tp, tn, p, n, sqrt=np.sqrt):
    """
    The standardized prevalence_threshold score

    Args:
        tp (int/float/np.array/Interval): The number of true positives
        tn (int/float/np.array/Interval): The number of true negatives
        p (int/float/np.array/Interval): The number of positives
        n (int/float/np.array/Interval): The number of negatives

    Returns:
        int/float/np.array/Interval: the score
    """
    return p*((n-tn) - n*sqrt((n-tn)*tp/(n*p)))/((n-tn)*p - n*tp)

def diagnostic_odds_ratio_standardized(*, tp, tn, p, n):
    """
    The standardized diagnostic_odds_ratio score

    Args:
        tp (int/float/np.array/Interval): The number of true positives
        tn (int/float/np.array/Interval): The number of true negatives
        p (int/float/np.array/Interval): The number of positives
        n (int/float/np.array/Interval): The number of negatives

    Returns:
        int/float/np.array/Interval: the score
    """
    return tn*tp/((n-tn)*(p-tp))

def jaccard_index_standardized(*, tp, tn, p, n):
    """
    The standardized jaccard_index score

    Args:
        tp (int/float/np.array/Interval): The number of true positives
        tn (int/float/np.array/Interval): The number of true negatives
        p (int/float/np.array/Interval): The number of positives
        n (int/float/np.array/Interval): The number of negatives

    Returns:
        int/float/np.array/Interval: the score
    """
    return tp / (n - tn + p)

def balanced_accuracy_standardized(*, tp, tn, p, n):
    """
    The standardized balanced_accuracy score

    Args:
        tp (int/float/np.array/Interval): The number of true positives
        tn (int/float/np.array/Interval): The number of true negatives
        p (int/float/np.array/Interval): The number of positives
        n (int/float/np.array/Interval): The number of negatives

    Returns:
        int/float/np.array/Interval: the score
    """
    return tp/(2*p) + tn/(2*n)

def cohens_kappa_standardized(*, tp, tn, p, n):
    """
    The standardized cohens_kappa score

    Args:
        tp (int/float/np.array/Interval): The number of true positives
        tn (int/float/np.array/Interval): The number of true negatives
        p (int/float/np.array/Interval): The number of positives
        n (int/float/np.array/Interval): The number of negatives

    Returns:
        int/float/np.array/Interval: the score
    """
    return -2*(n*p - n*tp - p*tn)/(n**2 - n*tn + n*tp + p**2 + p*tn - p*tp)

def p4_standardized(*, tp, tn, p, n):
    """
    The standardized p4 score

    Args:
        tp (int/float/np.array/Interval): The number of true positives
        tn (int/float/np.array/Interval): The number of true negatives
        p (int/float/np.array/Interval): The number of positives
        n (int/float/np.array/Interval): The number of negatives

    Returns:
        int/float/np.array/Interval: the score
    """
    return 4*tp*tn / (4*tp*tn + (tp + tn)*(p - tp + n - tn))

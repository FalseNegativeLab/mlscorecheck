"""
This module implements the scores with normal parameterizations
This is a generated file, do not modify it.
"""

import math

__all__ = [
'accuracy',
'error_rate',
'sensitivity',
'false_negative_rate',
'false_positive_rate',
'specificity',
'positive_predictive_value',
'false_discovery_rate',
'false_omission_rate',
'negative_predictive_value',
'f_beta_positive',
'f_beta_negative',
'f1_positive',
'f1_negative',
'unified_performance_measure',
'geometric_mean',
'fowlkes_mallows_index',
'markedness',
'positive_likelihood_ratio',
'negative_likelihood_ratio',
'matthews_correlation_coefficient',
'bookmaker_informedness',
'prevalence_threshold',
'diagnostic_odds_ratio',
'jaccard_index',
'balanced_accuracy',
'cohens_kappa']

def accuracy(*, tp, tn, p, n):
    """
    The accuracy score

    Args:
        tp (int|float|Interval|IntervalUnion): The number of true positives
        tn (int|float|Interval|IntervalUnion): The number of true negatives
        p (int|float|Interval|IntervalUnion): The number of positives
        n (int|float|Interval|IntervalUnion): The number of negatives

    Returns:
        int|float|Interval|IntervalUnion: the score
    """
    return (tp + tn)/(p + n)

def error_rate(*, fp, fn, p, n):
    """
    The error_rate score

    Args:
        fp (int|float|Interval|IntervalUnion): The number of false positives
        fn (int|float|Interval|IntervalUnion): The number of false negatives
        p (int|float|Interval|IntervalUnion): The number of positives
        n (int|float|Interval|IntervalUnion): The number of negatives

    Returns:
        int|float|Interval|IntervalUnion: the score
    """
    return (fp + fn)/(p + n)

def sensitivity(*, tp, p):
    """
    The sensitivity score

    Args:
        tp (int|float|Interval|IntervalUnion): The number of true positives
        p (int|float|Interval|IntervalUnion): The number of positives

    Returns:
        int|float|Interval|IntervalUnion: the score
    """
    return tp/p

def false_negative_rate(*, fn, p):
    """
    The false_negative_rate score

    Args:
        fn (int|float|Interval|IntervalUnion): The number of false negatives
        p (int|float|Interval|IntervalUnion): The number of positives

    Returns:
        int|float|Interval|IntervalUnion: the score
    """
    return fn/p

def false_positive_rate(*, fp, n):
    """
    The false_positive_rate score

    Args:
        fp (int|float|Interval|IntervalUnion): The number of false positives
        n (int|float|Interval|IntervalUnion): The number of negatives

    Returns:
        int|float|Interval|IntervalUnion: the score
    """
    return fp/n

def specificity(*, tn, n):
    """
    The specificity score

    Args:
        tn (int|float|Interval|IntervalUnion): The number of true negatives
        n (int|float|Interval|IntervalUnion): The number of negatives

    Returns:
        int|float|Interval|IntervalUnion: the score
    """
    return tn/n

def positive_predictive_value(*, tp, fp):
    """
    The positive_predictive_value score

    Args:
        tp (int|float|Interval|IntervalUnion): The number of true positives
        fp (int|float|Interval|IntervalUnion): The number of false positives

    Returns:
        int|float|Interval|IntervalUnion: the score
    """
    return tp/(tp + fp)

def false_discovery_rate(*, tp, fp):
    """
    The false_discovery_rate score

    Args:
        tp (int|float|Interval|IntervalUnion): The number of true positives
        fp (int|float|Interval|IntervalUnion): The number of false positives

    Returns:
        int|float|Interval|IntervalUnion: the score
    """
    return fp/(tp + fp)

def false_omission_rate(*, tn, fn):
    """
    The false_omission_rate score

    Args:
        tn (int|float|Interval|IntervalUnion): The number of true negatives
        fn (int|float|Interval|IntervalUnion): The number of false negatives

    Returns:
        int|float|Interval|IntervalUnion: the score
    """
    return fn/(tn + fn)

def negative_predictive_value(*, tn, fn):
    """
    The negative_predictive_value score

    Args:
        tn (int|float|Interval|IntervalUnion): The number of true negatives
        fn (int|float|Interval|IntervalUnion): The number of false negatives

    Returns:
        int|float|Interval|IntervalUnion: the score
    """
    return tn/(tn + fn)

def f_beta_positive(*, tp, fp, p, beta_positive):
    """
    The f_beta_positive score

    Args:
        tp (int|float|Interval|IntervalUnion): The number of true positives
        fp (int|float|Interval|IntervalUnion): The number of false positives
        p (int|float|Interval|IntervalUnion): The number of positives
        beta_positive (int|float|Interval|IntervalUnion): the beta parameter

    Returns:
        int|float|Interval|IntervalUnion: the score
    """
    return ((1 + beta_positive**2)*tp) / (tp + beta_positive**2*p + fp)

def f_beta_negative(*, tn, fn, n, beta_negative):
    """
    The f_beta_negative score

    Args:
        tn (int|float|Interval|IntervalUnion): The number of true negatives
        fn (int|float|Interval|IntervalUnion): The number of false negatives
        n (int|float|Interval|IntervalUnion): The number of negatives
        beta_negative (int|float|Interval|IntervalUnion): the beta parameter

    Returns:
        int|float|Interval|IntervalUnion: the score
    """
    return ((1 + beta_negative**2)*tn) / (tn + beta_negative**2*n + fn)

def f1_positive(*, tp, fp, p):
    """
    The f1_positive score

    Args:
        tp (int|float|Interval|IntervalUnion): The number of true positives
        fp (int|float|Interval|IntervalUnion): The number of false positives
        p (int|float|Interval|IntervalUnion): The number of positives

    Returns:
        int|float|Interval|IntervalUnion: the score
    """
    return (2*tp) / (tp + p + fp)

def f1_negative(*, tn, fn, n):
    """
    The f1_negative score

    Args:
        tn (int|float|Interval|IntervalUnion): The number of true negatives
        fn (int|float|Interval|IntervalUnion): The number of false negatives
        n (int|float|Interval|IntervalUnion): The number of negatives

    Returns:
        int|float|Interval|IntervalUnion: the score
    """
    return (2*tn) / (tn + n + fn)

def unified_performance_measure(*, tp, tn, p, n):
    """
    The unified_performance_measure score

    Args:
        tp (int|float|Interval|IntervalUnion): The number of true positives
        tn (int|float|Interval|IntervalUnion): The number of true negatives
        p (int|float|Interval|IntervalUnion): The number of positives
        n (int|float|Interval|IntervalUnion): The number of negatives

    Returns:
        int|float|Interval|IntervalUnion: the score
    """
    return 4*tn*tp/(tn*(n + p - tn + tp) + tp*(n + p + tn - tp))

def geometric_mean(*, tp, tn, p, n, sqrt=math.sqrt):
    """
    The geometric_mean score

    Args:
        tp (int|float|Interval|IntervalUnion): The number of true positives
        tn (int|float|Interval|IntervalUnion): The number of true negatives
        p (int|float|Interval|IntervalUnion): The number of positives
        n (int|float|Interval|IntervalUnion): The number of negatives

    Returns:
        int|float|Interval|IntervalUnion: the score
    """
    return sqrt(tp)*sqrt(tn)/(sqrt(p)*sqrt(n))

def fowlkes_mallows_index(*, tp, fp, p, sqrt=math.sqrt):
    """
    The fowlkes_mallows_index score

    Args:
        tp (int|float|Interval|IntervalUnion): The number of true positives
        fp (int|float|Interval|IntervalUnion): The number of false positives
        p (int|float|Interval|IntervalUnion): The number of positives

    Returns:
        int|float|Interval|IntervalUnion: the score
    """
    return tp/sqrt(p*(fp + tp))

def markedness(*, tp, tn, p, n):
    """
    The markedness score

    Args:
        tp (int|float|Interval|IntervalUnion): The number of true positives
        tn (int|float|Interval|IntervalUnion): The number of true negatives
        p (int|float|Interval|IntervalUnion): The number of positives
        n (int|float|Interval|IntervalUnion): The number of negatives

    Returns:
        int|float|Interval|IntervalUnion: the score
    """
    return tp/(tp + n - tn) + tn/(tn + p - tp) - 1

def positive_likelihood_ratio(*, tp, fp, p, n):
    """
    The positive_likelihood_ratio score

    Args:
        tp (int|float|Interval|IntervalUnion): The number of true positives
        fp (int|float|Interval|IntervalUnion): The number of false positives
        p (int|float|Interval|IntervalUnion): The number of positives
        n (int|float|Interval|IntervalUnion): The number of negatives

    Returns:
        int|float|Interval|IntervalUnion: the score
    """
    return n*tp/(fp*p)

def negative_likelihood_ratio(*, tn, fn, p, n):
    """
    The negative_likelihood_ratio score

    Args:
        tn (int|float|Interval|IntervalUnion): The number of true negatives
        fn (int|float|Interval|IntervalUnion): The number of false negatives
        p (int|float|Interval|IntervalUnion): The number of positives
        n (int|float|Interval|IntervalUnion): The number of negatives

    Returns:
        int|float|Interval|IntervalUnion: the score
    """
    return n*fn/(tn*p)

def matthews_correlation_coefficient(*, tp, tn, p, n, sqrt=math.sqrt):
    """
    The matthews_correlation_coefficient score

    Args:
        tp (int|float|Interval|IntervalUnion): The number of true positives
        tn (int|float|Interval|IntervalUnion): The number of true negatives
        p (int|float|Interval|IntervalUnion): The number of positives
        n (int|float|Interval|IntervalUnion): The number of negatives

    Returns:
        int|float|Interval|IntervalUnion: the score
    """
    return (tn*tp - (n - tn)*(p - tp))/sqrt(n*p*(n - tn + tp)*(p + tn - tp))

def bookmaker_informedness(*, tp, tn, p, n):
    """
    The bookmaker_informedness score

    Args:
        tp (int|float|Interval|IntervalUnion): The number of true positives
        tn (int|float|Interval|IntervalUnion): The number of true negatives
        p (int|float|Interval|IntervalUnion): The number of positives
        n (int|float|Interval|IntervalUnion): The number of negatives

    Returns:
        int|float|Interval|IntervalUnion: the score
    """
    return tp/p + tn/n - 1

def prevalence_threshold(*, tp, fp, p, n, sqrt=math.sqrt):
    """
    The prevalence_threshold score

    Args:
        tp (int|float|Interval|IntervalUnion): The number of true positives
        fp (int|float|Interval|IntervalUnion): The number of false positives
        p (int|float|Interval|IntervalUnion): The number of positives
        n (int|float|Interval|IntervalUnion): The number of negatives

    Returns:
        int|float|Interval|IntervalUnion: the score
    """
    return -p*(-fp + n*sqrt(fp*tp/(n*p)))/(fp*p - n*tp)

def diagnostic_odds_ratio(*, tp, tn, p, n):
    """
    The diagnostic_odds_ratio score

    Args:
        tp (int|float|Interval|IntervalUnion): The number of true positives
        tn (int|float|Interval|IntervalUnion): The number of true negatives
        p (int|float|Interval|IntervalUnion): The number of positives
        n (int|float|Interval|IntervalUnion): The number of negatives

    Returns:
        int|float|Interval|IntervalUnion: the score
    """
    return tn*tp/((n-tn)*(p-tp))

def jaccard_index(*, tp, fp, p):
    """
    The jaccard_index score

    Args:
        tp (int|float|Interval|IntervalUnion): The number of true positives
        fp (int|float|Interval|IntervalUnion): The number of false positives
        p (int|float|Interval|IntervalUnion): The number of positives

    Returns:
        int|float|Interval|IntervalUnion: the score
    """
    return tp/(fp + p)

def balanced_accuracy(*, tp, tn, p, n):
    """
    The balanced_accuracy score

    Args:
        tp (int|float|Interval|IntervalUnion): The number of true positives
        tn (int|float|Interval|IntervalUnion): The number of true negatives
        p (int|float|Interval|IntervalUnion): The number of positives
        n (int|float|Interval|IntervalUnion): The number of negatives

    Returns:
        int|float|Interval|IntervalUnion: the score
    """
    return tp/(2*p) + tn/(2*n)

def cohens_kappa(*, tp, tn, p, n):
    """
    The cohens_kappa score

    Args:
        tp (int|float|Interval|IntervalUnion): The number of true positives
        tn (int|float|Interval|IntervalUnion): The number of true negatives
        p (int|float|Interval|IntervalUnion): The number of positives
        n (int|float|Interval|IntervalUnion): The number of negatives

    Returns:
        int|float|Interval|IntervalUnion: the score
    """
    return (2*tn*tp - 2*(n - tn)*(p - tp))/(n*(n - tn + tp) + p*(p + tn - tp))

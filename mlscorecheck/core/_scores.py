"""
This module implements the calculation of scores
"""

import sympy
import numpy as np

__all__ = ['accuracy',
           'error_rate',
           'sensitivity',
           'specificity',
           'positive_predictive_value',
           'precision',
           'negative_predictive_value',
           'f_beta_plus',
           'f_beta_minus',
           'f1_plus',
           'f1_minus',
           'unified_performance_measure',
           'true_positive_rate',
           'recall',
           'false_positive_rate',
           'false_negative_rate',
           'true_negative_rate',
           'false_discovery_rate',
           'false_omission_rate',
           'geometric_mean',
           'fowlkes_mallows_index',
           'markedness',
           'positive_likelihood_ratio',
           'negative_likelihood_ratio',
           'matthews_correlation_coefficient',
           'informedness',
           'prevalence_threshold',
           'diagnostic_odds_ratio',
           'threat_score',
           'jaccard_index',
           'balanced_accuracy',
           'cohens_kappa',
           'score_function_set',
           'aliases',
           'complementers',
           'sqrt']

def accuracy(*, tp, tn, p, n):
    """
    The accuracy

    Args:
        tp (int/float/np.array/Interval): the number of true positives
        tn (int/float/np.array/Interval): the number of true negatives
        p (int/float/np.array/Interval): the number of positives
        n (int/float/np.array/Interval): the number of negatives

    Returns:
        float/np.array/Interval: the score
    """
    return (tp + tn)/(p + n)

def error_rate(*, fp, fn, p, n):
    """
    The error rate

    Args:
        fp (int/float/np.array/Interval): the number of false positives
        fn (int/float/np.array/Interval): the number of false negatives
        p (int/float/np.array/Interval): the number of positives
        n (int/float/np.array/Interval): the number of negatives

    Returns:
        float/np.array/Interval: the score
    """
    return (fp + fn)/(p + n)

def sensitivity(*, tp, p):
    """
    The sensitivity

    Args:
        tp (int/float/np.array/Interval): the number of true positives
        p (int/float/np.array/Interval): the number of positive items

    Returns:
        float/np.array/Interval: the score
    """
    return tp/p

def specificity(*, tn, n):
    """
    The specificity

    Args:
        tn (int/float/np.array/Interval): the number of true negatives
        n (int/float/np.array/Interval): the number of negative items

    Returns:
        float/np.array/Interval: the score
    """
    return tn/n

def positive_predictive_value(*, tp, fp):
    """
    The positive predictive value

    Args:
        tp (int/float/np.array/Interval): the number of true positives
        fp (int/float/np.array/Interval): the number of false positives

    Returns:
        float/np.array/Interval: the score
    """
    return tp/(tp + fp)

def precision(*, tp, fp):
    """
    The precision - alias to the positive predictive value

    Args:
        tp (int/float/np.array/Interval): the number of true positives
        fp (int/float/np.array/Interval): the number of false positives

    Returns:
        float/np.array/Interval: the score
    """
    return positive_predictive_value(tp=tp, fp=fp)

def negative_predictive_value(*, tn, fn):
    """
    The negative predictive value

    Args:
        tn (int/float/np.array/Interval): the number of true negatives
        fn (int/float/np.array/Interval): the number of false negatives

    Returns:
        float/np.array/Interval: the score
    """
    return tn/(tn + fn)

def f_beta_plus(*, tp, fp, p, beta_plus=1):
    """
    The f_beta_plus score

    Args:
        tp (int/float/np.array/Interval): the number of true positives
        fp (int/float/np.array/Interval): the number of false positives
        p (int/float/np.array/Interval): the number of positives
        beta_plus (int/float/np.array/Interval): the beta value

    Returns:
        float/np.array/Interval: the score
    """
    return ((1 + beta_plus**2)*tp) / (tp + beta_plus**2*p + fp)

def f_beta_minus(*, tn, fn, n, beta_minus=1):
    """
    The f_beta_minus score

    Args:
        tn (int/float/np.array/Interval): the number of true negatives
        fn (int/float/np.array/Interval): the number of false negatives
        n (int/float/np.array/Interval): the number of negatives
        beta_minus (int/float/np.array/Interval): the beta value

    Returns:
        float/np.array/Interval: the score
    """
    return ((1 + beta_minus**2)*tn) / (tn + beta_minus**2*n + fn)

def f1_plus(*, tp, fp, p):
    """
    The f_1 plus score

    Args:
        tp (int/float/np.array/Interval): the number of true positives
        fp (int/float/np.array/Interval): the number of false positives
        p (int/float/np.array/Interval): the number of positives

    Returns:
        float/np.array/Interval: the score
    """
    return f_beta_plus(tp=tp, fp=fp, p=p, beta_plus=1)

def f1_minus(*, tn, fn, n):
    """
    The f_1 minus score

    Args:
        tn (int/float/np.array/Interval): the number of true negatives
        fn (int/float/np.array/Interval): the number of false negatives
        n (int/float/np.array/Interval): the number of negatives

    Returns:
        float/np.array/Interval: the score
    """
    return f_beta_minus(tn=tn, fn=fn, n=n, beta_minus=1)

def unified_performance_measure(*, tp, tn, p, n):
    """
    The unified performance measure score

    Args:
        tp (int/float/np.array/Interval): the number of true positives
        tn (int/float/np.array/Interval): the number of true negatives
        p (int/float/np.array/Interval): the number of positives
        n (int/float/np.array/Interval): the number of negatives

    Returns:
        float/np.array/Interval: the score
    """

    fp = n - tn
    fn = p - tp

    f1p = f1_plus(tp=tp, fp=fp, p=p)
    f1n = f1_minus(tn=tn, fn=fn, n=n)

    return 2 * (f1p * f1n) / (f1p + f1n)

def sqrt(object):
    """
    Square root, numeric or symbolic

    Args:
        object: to take the square root of

    Returns:
        object: the square root
    """
    if isinstance(object, sympy.Basic):
        return sympy.sqrt(object)

    return np.sqrt(object)

def true_positive_rate(*, tp, p):
    """
    The true negative rate: alias for the sensitivity

    Args:
        tp (int/float/np.array/Interval): the number of true positives
        p (int/float/np.array/Interval): the total number of positives

    Returns:
        float/np.array/Interval: the true positive rate
    """
    return sensitivity(tp=tp, p=p)

def recall(*, tp, p):
    """
    The recall: alias for the sensitivity

    Args:
        tp (int/float/np.array/Interval): the number of true positives
        p (int/float/np.array/Interval): the total number of positives

    Returns:
        float/np.array/Interval: the true positive rate
    """
    return sensitivity(tp=tp, p=p)

def false_positive_rate(*, fp, n):
    """
    The false positive rate

    Args:
        fp (int/float/np.array/Interval): the number of false positives
        n (int/float/np.array/Interval): the total number of negatives

    Returns:
        float/np.array/Interval: the false positive rate
    """
    return fp/n

def false_negative_rate(*, fn, p):
    """
    The false negative rate

    Args:
        fn (int/float/np.array/Interval): the number of false negatives
        p (int/float/np.array/Interval): the total number of positives

    Returns:
        float/np.array/Interval: the false negative rate
    """
    return fn/p

def true_negative_rate(*, tn, n):
    """
    The true negative rate: alias for specificity

    Args:
        tn (int/float/np.array/Interval): the number of true negatives
        n (int/float/np.array/Interval): the total number of negatives

    Returns:
        float/np.array/Interval: the true negative rate
    """
    return specificity(tn=tn, n=n)

def false_discovery_rate(*, tp, fp):
    """
    The false discover rate (1 - ppv)

    Args:
        tp (int/float/np.array/Interval): the number of true positives
        fp (int/float/np.array/Interval): the number of false positives

    Returns:
        float/np.array/Interval: the false discovery rate
    """
    return 1 - positive_predictive_value(tp=tp, fp=fp)

def false_omission_rate(*, tn, fn):
    """
    The false omission rate (1 - npv)

    Args:
        tn (int/float/np.array/Interval): the number of true negatives
        fn (int/float/np.array/Interval): the number of false negatives

    Returns:
        float/np.array/Interval: the false omission rate
    """
    return 1 - negative_predictive_value(tn=tn, fn=fn)

def geometric_mean(*, tp, tn, p, n):
    """
    The geometric mean score

    Args:
        tp (int/float/np.array/Interval): the number of true positives
        tn (int/float/np.array/Interval): the number of true negatives
        p (int/float/np.array/Interval): the number of positives
        n (int/float/np.array/Interval): the number of negatives

    Returns:
        float/np.array/Interval: the geometric mean
    """

    tpr = true_positive_rate(tp=tp, p=p)
    tnr = true_negative_rate(tn=tn, n=n)

    return sqrt(tpr*tnr)

def fowlkes_mallows_index(*, tp, fp, p):
    """
    The Fowlkes-Mallows index

    Args:
        tp (int/float/np.array/Interval): the number of true positives
        fp (int/float/np.array/Interval): the number of false positives
        p (int/float/np.array/Interval): the total number of positives

    Returns:
        float/np.array/Interval: the Fowlkes-Mallows index
    """
    ppv = positive_predictive_value(tp=tp, fp=fp)
    tpr = true_positive_rate(tp=tp, p=p)

    return sqrt(ppv * tpr)

def markedness(*, tp, tn, fp, fn):
    """
    The markedness

    Args:
        tp (int/float/np.array/Interval): the number of true positives
        tn (int/float/np.array/Interval): the number of true negatives
        fp (int/float/np.array/Interval): the number of false positives
        fn (int/float/np.array/Interval): the number of false negatives

    Returns:
        float/np.array/Interval: the markedness score
    """
    ppv = positive_predictive_value(tp=tp, fp=fp)
    npv = negative_predictive_value(tn=tn, fn=fn)
    return ppv + npv - 1

def positive_likelihood_ratio(*, tp, fp, p, n):
    """
    The positive likelihood ratio

    Args:
        tp (int/float/np.array/Interval): the number of true positives
        fp (int/float/np.array/Interval): the number of false positives
        p (int/float/np.array/Interval): the number of positives
        n (int/float/np.array/Interval): the number of negatives

    Returns:
        float/np.array/Interval: the postiive likelihood score
    """
    tpr = true_positive_rate(tp=tp, p=p)
    fpr = false_positive_rate(fp=fp, n=n)
    return tpr/fpr

def negative_likelihood_ratio(*, tn, fn, p, n):
    """
    The negative likelihood ratio

    Args:
        tn (int/float/np.array/Interval): the number of true negatives
        fn (int/float/np.array/Interval): the number of false negatives
        p (int/float/np.array/Interval): the number of positives
        n (int/float/np.array/Interval): the number of negatives

    Returns:
        float/np.array/Interval: the negative likelihood score
    """
    fnr = false_negative_rate(fn=fn, p=p)
    tnr = true_negative_rate(tn=tn, n=n)

    return fnr/tnr

def matthews_correlation_coefficient(*, tp, tn, fp, fn, p, n):
    """
    The Matthew's correlation coefficient

    Args:
        tp (int/float/np.array/Interval): the number of true positives
        tn (int/float/np.array/Interval): the number of true negatives
        fp (int/float/np.array/Interval): the number of false positives
        fn (int/float/np.array/Interval): the number of false negatives
        p (int/float/np.array/Interval): the number of positives
        n (int/float/np.array/Interval): the number of negatives

    Returns:
        float/np.array/Interval: the Matthew's correlation coefficient
    """
    numerator = (tp * tn) - (fp * fn)
    denominator = sqrt((tp + fp)*p*n*(tn + fn))
    return numerator / denominator

def informedness(*, tp, tn, p, n):
    """
    The informedness score

    Args:
        tp (int/float/np.array/Interval): the number of true positives
        tn (int/float/np.array/Interval): the number of true negatives
        p (int/float/np.array/Interval): the number of positives
        n (int/float/np.array/Interval): the number of negatives

    Returns:
        float/np.array/Interval: the informedness score
    """
    tpr = true_positive_rate(tp=tp, p=p)
    tnr = true_negative_rate(tn=tn, n=n)
    return tpr + tnr - 1

def prevalence_threshold(*, tp, fp, p, n):
    """
    The prevalence score

    Args:
        tp (int/float/np.array/Interval): the number of true positives
        fp (int/float/np.array/Interval): the number of false positives
        p (int/float/np.array/Interval): the number of positives
        n (int/float/np.array/Interval): the number of negatives

    Returns:
        float/np.array/Interval: the prevalence threshold score
    """
    tpr = true_positive_rate(tp=tp, p=p)
    fpr = false_positive_rate(fp=fp, n=n)

    return (sqrt(tpr * fpr) - fpr)/(tpr - fpr)

def diagnostic_odds_ratio(*, tp, tn, fp, fn, p, n):
    """
    The diagnostic odds ratio

    Args:
        tp (int/float/np.array/Interval): the number of true positives
        tn (int/float/np.array/Interval): the number of true negatives
        fp (int/float/np.array/Interval): the number of false positives
        fn (int/float/np.array/Interval): the number of false negatives
        p (int/float/np.array/Interval): the number of positives
        n (int/float/np.array/Interval): the number of negatives

    Returns:
        float/np.array/Interval: the diagnostic odds ratio
    """
    plr = positive_likelihood_ratio(tp=tp, fp=fp, p=p, n=n)
    nlr = negative_likelihood_ratio(tn=tn, fn=fn, p=p, n=n)
    return plr/nlr

def threat_score(*, tp, fp, p):
    """
    The threat score

    Args:
        tp (int/float/np.array/Interval): the number of true positives
        fp (int/float/np.array/Interval): the number of false positives
        p (int/float/np.array/Interval): the number of positives

    Returns:
        float/np.array/Interval: the threat score
    """
    return tp/(fp + p)

def jaccard_index(*, tp, fp, p):
    """
    The Jaccrad index - alias to the threat score

    Args:
        tp (int/float/np.array/Interval): the number of true positives
        fp (int/float/np.array/Interval): the number of false positives
        p (int/float/np.array/Interval): the number of positives

    Returns:
        float/np.array/Interval: the Jaccard index
    """
    return threat_score(tp=tp, fp=fp, p=p)

def balanced_accuracy(*, tp, tn, p, n):
    """
    The balanced accuracy

    Args:
        tp (int/float/np.array/Interval): the number of true positives
        tn (int/float/np.array/Interval): the number of true negatives
        p (int/float/np.array/Interval): the number of positives
        n (int/float/np.array/Interval): the number of negatives

    Returns:
        float/np.array/Interval: the balanced accuracy
    """
    tpr = true_positive_rate(tp=tp, p=p)
    tnr = true_negative_rate(tn=tn, n=n)

    return (tpr + tnr)/2.0

def cohens_kappa(*, tp, tn, p, n):
    """
    Cohen's kappa score

    Args:
        tp (int/float/np.array/Interval): the number of true positives
        tn (int/float/np.array/Interval): the number of true negatives
        p (int/float/np.array/Interval): the number of positives
        n (int/float/np.array/Interval): the number of negatives

    Returns:
        float/np.array/Interval: the Cohen's kappa score
    """
    acc = accuracy(tp=tp, tn=tn, p=p, n=n)
    fp = p - tn
    fn = n - tp

    term = ((tp + tn)*(fp + fn)) / (2*(tp*tn - fp*fn))

    return acc/(acc + term)

def score_function_set():
    """
    Return a set of scores with no aliases

    Returns:
        dict: the scores with no aliases
    """
    return {'acc': accuracy,
            'sens': sensitivity,
            'spec': specificity,
            'npv': negative_predictive_value,
            'ppv': positive_predictive_value,
            'f1_plus': f1_plus,
            'f1_minus': f1_minus,
            'fm': fowlkes_mallows_index,
            'bm': informedness,
            'pt': prevalence_threshold,
            'lrp': positive_likelihood_ratio,
            'lrn': negative_likelihood_ratio,
            'mk': markedness,
            'dor': diagnostic_odds_ratio,
            'mcc': matthews_correlation_coefficient,
            'ji': jaccard_index,
            'ba': balanced_accuracy,
            'kappa': cohens_kappa,
            'gm': geometric_mean,
            'upm': unified_performance_measure
            }

def aliases():
    return {'tpr': 'sens',
            'tnr': 'spec',
            'prec': 'ppv',
            'rec': 'sens'}

def complementers():
    return {'fdr': 'ppv',
            'for': 'npv',
            'fnr': 'tpr',
            'fpr': 'tnr',
            'acc': 'err'}

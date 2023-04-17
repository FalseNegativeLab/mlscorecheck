"""
This module implements the calculation of scores
"""

__all__ = ['accuracy',
           'sensitivity',
           'specificity',
           'positive_predictive_value',
           'negative_predictive_value']

def accuracy(*, tp, tn, total):
    """
    The accuracy

    Args:
        tp (int/float): the number of true positives
        tn (int/float): the number of true negatives
        total (int/float): the total number of items

    Returns:
        float: the score
    """
    return (tp + tn)/total

def sensitivity(*, tp, p):
    """
    The sensitivity

    Args:
        tp (int/float): the number of true positives
        p (int/float): the number of positive items

    Returns:
        float: the score
    """
    return tp/p

def specificity(*, tn, n):
    """
    The specificity

    Args:
        tn (int/float): the number of true negatives
        n (int/float): the number of negative items

    Returns:
        float: the score
    """
    return tn/n

def positive_predictive_value(*, tp, fp):
    """
    The positive predictive value

    Args:
        tp (int/float): the number of true positives
        fp (int/float): the number of false positives

    Returns:
        float: the score
    """
    return tp/(tp + fp)

def negative_predictive_value(*, tn, fn):
    """
    The negative predictive value

    Args:
        tn (int/float): the number of true negatives
        fn (int/float): the number of false negatives

    Returns:
        float: the score
    """
    return tn/(tn + fn)

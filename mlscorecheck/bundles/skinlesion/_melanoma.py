"""
This module implements the test bundles related to melanoma classification
"""

from ...check import check_1_testset_no_kfold_scores
from ...experiments import (load_isic2016,
                            load_isic2017)

__all__ = ['check_isic2016',
            'check_isic2017m',
            'check_isic2017sk']

def check_isic2016(*, scores, eps):
    """
    Tests if the scores are consistent with the test set of the ISIC2016
    melanoma classification dataset
    """

    return check_1_testset_no_kfold_scores(scores=scores,
                                            testset=load_isic2016(),
                                            eps=eps)

def check_isic2017m(*, scores, eps):
    """
    Tests if the scores are consistent with the test set of the ISIC2017
    melanoma classification dataset regarding the task of binary classification
    of melanoma images
    """

    return check_1_testset_no_kfold_scores(scores=scores,
                                            testset=load_isic2017()['m'],
                                            eps=eps)

def check_isic2017sk(*, scores, eps):
    """
    Tests if the scores are consistent with the test set of the ISIC2017
    melanoma classification dataset regarding the task of binary classification
    of seborrheic keratosis images
    """

    return check_1_testset_no_kfold_scores(scores=scores,
                                            testset=load_isic2017()['sk'],
                                            eps=eps)

"""
This module tests the AUC estimation related functionalities
"""

import pytest

import numpy as np

from mlscorecheck.auc import (
    prepare_intervals_for_auc_estimation,
    auc_from_sens_spec,
    acc_from_auc
)

def test_prepare_intervals_for_auc_estimation():
    """
    Testing the perparation of intervals
    """

    p = 20
    n = 80
    eps = 1e-4

    scores = {
        'acc': 0.6,
        'sens': 0.55,
        'spec': 0.7,
        'asdf': 'dummy'
    }

    intervals = prepare_intervals_for_auc_estimation(scores, eps=eps, p=p, n=n)
    assert len(intervals) == 3

    scores = {
        'sens': 0.55,
        'spec': 0.7
    }

    intervals = prepare_intervals_for_auc_estimation(scores, eps=eps, p=p, n=n)
    assert len(intervals) == 3
    acc = (p*scores['sens'] + n*scores['spec']) / (p + n)
    assert abs(np.mean(intervals['acc']) - acc) < 1e-5

    scores = {
        'acc': 0.6,
        'spec': 0.7
    }

    intervals = prepare_intervals_for_auc_estimation(scores, eps=eps, p=p, n=n)
    assert len(intervals) == 3
    sens = ((p + n)*scores['acc'] - n*scores['spec']) / p
    print(intervals, sens)
    assert abs(np.mean(intervals['sens']) - sens) < 1e-5

    scores = {
        'acc': 0.6,
        'sens': 0.7
    }

    intervals = prepare_intervals_for_auc_estimation(scores, eps=eps, p=p, n=n)
    assert len(intervals) == 3
    spec = ((p + n)*scores['acc'] - p*scores['sens']) / n
    assert abs(np.mean(intervals['spec']) - spec) < 1e-5

def test_auc_from():
    """
    Testing the auc estimation functionalities
    """

    p = 20
    n = 80
    eps = 1e-4

    scores = {
        'acc': 0.6
    }

    with pytest.raises(ValueError):
        auc_from_sens_spec(
            scores=scores,
            eps=eps,
            p=p,
            n=n,
            lower='min',
            upper='max'
        )

    scores = {
        'acc': 0.6,
        'sens': 0.55,
        'spec': 0.7,
        'asdf': 'dummy'
    }

    with pytest.raises(ValueError):
        auc_from_sens_spec(
            scores=scores,
            eps=eps,
            p=p,
            n=n,
            lower='cmin',
            upper='max-acc'
        )

    scores = {
        'acc': 0.9,
        'sens': 0.92,
        'spec': 0.95,
        'asdf': 'dummy'
    }

    auc0 = auc_from_sens_spec(
            scores=scores,
            eps=eps,
            p=p,
            n=n,
            lower='min',
            upper='max'
        )

    auc1 = auc_from_sens_spec(
            scores=scores,
            eps=eps,
            p=p,
            n=n,
            lower='cmin',
            upper='max-acc'
        )

    assert auc0[0] < auc1[0]
    assert auc0[1] > auc1[1]

    with pytest.raises(ValueError):
        auc_from_sens_spec(
            scores=scores | {'sens': 0.1, 'spec': 0.8},
            eps=eps,
            p=p,
            n=n,
            lower='cmin',
            upper='max-acc'
        )

    with pytest.raises(ValueError):
        auc_from_sens_spec(
            scores=scores,
            eps=eps,
            p=p,
            n=n,
            lower='dummy',
            upper='max'
        )

    with pytest.raises(ValueError):
        auc_from_sens_spec(
            scores=scores,
            eps=eps,
            p=p,
            n=n,
            lower='min',
            upper='dummy'
        )

def test_acc_from_auc():
    """
    Testing the accuracy estimation
    """

    scores = {'auc': 0.9}
    p = 20
    n = 80
    eps = 1e-4

    acc = acc_from_auc(scores=scores, eps=eps, p=p, n=n)

    assert acc[1] > acc[0]

    scores = {'auc': 1 - min(p, n)/(2*max(p, n)) - 0.1}

    with pytest.raises(ValueError):
        acc_from_auc(scores=scores, eps=eps, p=p, n=n)

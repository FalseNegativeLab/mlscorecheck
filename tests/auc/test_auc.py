"""
This module tests the AUC estimation related functionalities
"""

import pytest

import numpy as np

from mlscorecheck.auc import (
    prepare_intervals_for_auc_estimation,
    auc_from_sens_spec,
    acc_from_auc,
    generate_kfold_auc_fix_problem,
    generate_kfold_sens_spec_fix_problem,
    auc_from_sens_spec_kfold,
    generate_average
)

EPS = 0.05
ITERATIONS = 5000

@pytest.mark.parametrize('sens_spec', [(0.7, 0.8), (0.6, 0.8)])
@pytest.mark.parametrize('k', [5, 10])
def test_kfold_min(sens_spec, k):
    # min
    aucs = []
    senss = []
    specs = []
    p = 100
    n = 100
    for _ in range(ITERATIONS):
        problem = generate_kfold_sens_spec_fix_problem(sens=sens_spec[0], 
                                                       spec=sens_spec[1], 
                                                       k=k)
        aucs.append(np.mean(problem['spec'] * problem['sens']))
        senss.append(problem['sens'])
        specs.append(problem['spec'])

    interval = auc_from_sens_spec_kfold(scores={'sens': sens_spec[0], 'spec': sens_spec[1]},
                         eps=1e-4,
                         p=p,
                         n=n,
                         k=k,
                         lower='min')
    assert np.abs(np.min(aucs) - interval[0]) < EPS

@pytest.mark.parametrize('sens_spec', [(0.7, 0.8), (0.6, 0.8)])
@pytest.mark.parametrize('k', [5, 10])
def test_kfold_cmin(sens_spec, k):
    aucs = []
    senss = []
    specs = []
    p = 100
    n = 100
    for _ in range(ITERATIONS):
        problem = generate_kfold_sens_spec_fix_problem(sens=sens_spec[0], 
                                                       spec=sens_spec[1], 
                                                       k=k)
        if np.any(problem['sens'] < 1 - problem['spec']):
            continue
        aucs.append(np.mean(0.5 + (1 - problem['spec'] - problem['sens'])**2/2.0))
        senss.append(problem['sens'])
        specs.append(problem['spec'])

    interval = auc_from_sens_spec_kfold(scores={'sens': sens_spec[0], 'spec': sens_spec[1]},
                         eps=1e-4,
                         p=p,
                         n=n,
                         k=k,
                         lower='cmin')
    assert np.abs(np.min(aucs) - interval[0]) < EPS

@pytest.mark.parametrize('sens_spec', [(0.7, 0.8), (0.6, 0.8)])
@pytest.mark.parametrize('k', [5, 10])
def test_kfold_max(sens_spec, k):
    aucs = []
    senss = []
    specs = []
    p = 100
    n = 100
    for _ in range(ITERATIONS):
        problem = generate_kfold_sens_spec_fix_problem(sens=sens_spec[0], 
                                                       spec=sens_spec[1], 
                                                       k=k)
        aucs.append(1 - np.mean((1 - problem['spec']) * (1 - problem['sens'])))
        senss.append(problem['sens'])
        specs.append(problem['spec'])
    np.max(aucs)

    interval = auc_from_sens_spec_kfold(scores={'sens': sens_spec[0], 
                                                'spec': sens_spec[1]},
                         eps=1e-4,
                         p=p,
                         n=n,
                         k=k,
                         upper='max')
    assert np.abs(np.max(aucs) - interval[1]) < EPS

@pytest.mark.parametrize('sens_spec', [(0.7, 0.8), (0.6, 0.8)])
@pytest.mark.parametrize('k', [5, 10])
def test_kfold_amax(sens_spec, k):
    aucs = []
    senss = []
    specs = []
    accs0 = []
    p = 100
    n = 100
    for _ in range(ITERATIONS):
        problem = generate_kfold_sens_spec_fix_problem(sens=sens_spec[0],
                                                       spec=sens_spec[1], 
                                                       k=k)
        accs = (problem['sens'] + problem['spec']) / 2.0
        aucs.append(1 - np.mean(((1 - accs) * (p + n))**2 / (2*p*n)))
        senss.append(problem['sens'])
        specs.append(problem['spec'])
        accs0.append(accs)

    interval = auc_from_sens_spec_kfold(scores={'sens': sens_spec[0], 
                                                'spec': sens_spec[1]},
                         eps=1e-4,
                         p=p,
                         n=n,
                         k=k,
                         upper='amax')
    assert np.abs(np.max(aucs) - interval[1]) < EPS

@pytest.mark.parametrize('auc', [0.7, 0.8])
@pytest.mark.parametrize('k', [5, 10])
def test_kfold_acc_min_max(auc, k):
    accs = []
    p = 100
    n = 100
    for _ in range(ITERATIONS):
        aucs = generate_average(auc, k)
        tmp = []
        for a in aucs:
            if a < 0.51:
                break
            acc_int = acc_from_auc(scores={'auc': a}, eps=1e-4, p=p, n=n, upper='max')
            tmp.append(np.random.random()*(acc_int[1] - acc_int[0]) + acc_int[0])
        if len(tmp) < k:
            continue
        
        accs.append(np.mean(tmp))

    interval = acc_from_auc(scores={'auc': auc}, eps=1e-4, p=p, n=n, upper='max')
    assert np.abs(np.max(accs) - interval[1]) < EPS
    assert np.abs(np.min(accs) - interval[0]) < EPS

@pytest.mark.parametrize('auc', [0.7, 0.8])
@pytest.mark.parametrize('k', [5, 10])
def test_kfold_acc_min_cmax(auc, k):
    accs = []
    p = 100
    n = 100
    for _ in range(ITERATIONS):
        aucs = generate_average(auc, k)
        tmp = []
        for a in aucs:
            if a < 0.51:
                break
            acc_int = acc_from_auc(scores={'auc': a}, eps=1e-4, p=p, n=n, upper='cmax')
            tmp.append(np.random.random()*(acc_int[1] - acc_int[0]) + acc_int[0])
        
        if len(tmp) < k:
            continue

        accs.append(np.mean(tmp))

    interval = acc_from_auc(scores={'auc': auc}, eps=1e-4, p=p, n=n, upper='cmax')
    assert np.abs(np.max(accs) - interval[1]) < EPS
    assert np.abs(np.min(accs) - interval[0]) < EPS

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
            upper='amax'
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
            upper='amax'
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
            upper='amax'
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

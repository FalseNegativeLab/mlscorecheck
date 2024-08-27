"""
This module tests the AUC functionalities related to single test sets
"""

import pytest

import numpy as np

from mlscorecheck.auc import (
    roc_min,
    roc_max,
    roc_rmin,
    roc_rmin_grid,
    roc_maxa,
    auc_min,
    auc_max,
    auc_rmin,
    auc_rmin_grid,
    auc_maxa,
    auc_amin,
    auc_armin,
    auc_amax,
    acc_min,
    acc_rmin,
    acc_max,
    acc_rmax,
    macc_min,
    integrate_roc_curve,
    translate_scores,
    prepare_intervals,
    augment_intervals,
    auc_from,
    acc_from,
    max_acc_from
)

scenarios = [
    {
        'fpr': 0.1,
        'tpr': 0.8,
        'p': 21,
        'n': 80
    },
    {
        'fpr': 0.13,
        'tpr': 0.82,
        'p': 49,
        'n': 50
    }
]

auc_scenarios = [
    {
        'auc': 0.8,
        'p': 20,
        'n': 60
    },
    {
        'auc': 0.4,
        'p': 50,
        'n': 51
    }
]

@pytest.mark.parametrize('scenario', scenarios)
def test_auc_min(scenario):
    """
    Testing the minimum ROC curves

    Args:
        scenario (dict): the scenario to test
    """

    np.testing.assert_almost_equal(
        integrate_roc_curve(*roc_min(scenario['fpr'], scenario['tpr'])),
        auc_min(scenario['fpr'], scenario['tpr'])
    )

@pytest.mark.parametrize('scenario', scenarios)
def test_auc_max(scenario):
    """
    Testing the maximum ROC curves

    Args:
        scenario (dict): the scenario to test
    """

    np.testing.assert_almost_equal(
        integrate_roc_curve(*roc_max(scenario['fpr'], scenario['tpr'])),
        auc_max(scenario['fpr'], scenario['tpr'])
    )

@pytest.mark.parametrize('scenario', scenarios)
def test_auc_rmin(scenario):
    """
    Testing the regulated minimum ROC curves

    Args:
        scenario (dict): the scenario to test
    """

    np.testing.assert_almost_equal(
        integrate_roc_curve(*roc_rmin(scenario['fpr'], scenario['tpr'])),
        auc_rmin(scenario['fpr'], scenario['tpr'])
    )

@pytest.mark.parametrize('scenario', scenarios)
def test_auc_rmin_grid(scenario):
    """
    Testing the regulated minimum ROC curves with grid correction

    Args:
        scenario (dict): the scenario to test
    """

    np.testing.assert_almost_equal(
        integrate_roc_curve(*roc_rmin_grid(
            scenario['fpr'], scenario['tpr'], scenario['p'], scenario['n']
        )),
        auc_rmin_grid(
            scenario['fpr'], scenario['tpr'], scenario['p'], scenario['n']
        )
    )

@pytest.mark.parametrize('scenario', scenarios)
def test_auc_maxa(scenario):
    """
    Testing the maximum accuracy ROC curves

    Args:
        scenario (dict): the scenario to test
    """

    fpr = scenario['fpr']
    tpr = scenario['tpr']
    p = scenario['p']
    n = scenario['n']

    acc = (tpr*p + (1 - fpr)*n) / (p + n)

    np.testing.assert_almost_equal(
        integrate_roc_curve(*roc_maxa(acc, p, n)),
        auc_maxa(acc, p, n)
    )

def test_rmin_error():
    """
    Testing the existential conditions in regulated minimum curves
    """

    with pytest.raises(ValueError):
        roc_rmin(0.8, 0.3)

    with pytest.raises(ValueError):
        auc_rmin(0.8, 0.3)

    with pytest.raises(ValueError):
        roc_rmin_grid(0.8, 0.3, 20, 100)

    with pytest.raises(ValueError):
        auc_rmin_grid(0.8, 0.3, 20, 100)

def test_maxa_error():
    """
    Testing the existential conditions in max-accuracy curves
    """

    with pytest.raises(ValueError):
        roc_maxa(0.3, 50, 50)

    with pytest.raises(ValueError):
        auc_maxa(0.3, 50, 50)

def test_auc_amin():
    """
    Testing the accuracy based minimum AUC estimation
    """

    assert auc_amin(0.1, 50, 50) == 0.0

    tpr = scenarios[0]['tpr']
    fpr = scenarios[0]['fpr']
    p = scenarios[0]['p']
    n = scenarios[0]['n']

    acc = (tpr * p + (1 - fpr) * n) / (p + n)

    tpr_at_fpr_0 = (acc * (p + n) - n) / p
    fpr_at_tpr_1 = 1 - (acc * (p + n) - p) / n

    auc_a = auc_min(fpr_at_tpr_1, 1)
    auc_b = auc_min(0, tpr_at_fpr_0)

    np.testing.assert_almost_equal(auc_amin(acc, p, n), min(auc_a, auc_b))

def test_auc_amax():
    """
    Testing the accuracy based maximum AUC estimation
    """

    assert auc_amax(0.9, 50, 50) == 1.0

    tpr = 0.3
    fpr = 0.7
    p = 43
    n = 80

    acc = (tpr * p + (1 - fpr) * n) / (p + n)

    tpr_at_fpr_1 = (acc * (p + n)) / p
    fpr_at_tpr_0 = 1 - (acc * (p + n)) / n

    auc_a = auc_max(fpr_at_tpr_0, 0)
    auc_b = auc_max(1, tpr_at_fpr_1)

    print(acc, tpr_at_fpr_1, fpr_at_tpr_0, auc_a, auc_b)

    np.testing.assert_almost_equal(auc_amax(acc, p, n), max(auc_a, auc_b))

def test_auc_armin():
    """
    Testing the accuracy based minimum AUC under a regulated minimum curve
    """

    with pytest.raises(ValueError):
        auc_armin(0.1, 50, 50)

    tpr = scenarios[0]['tpr']
    fpr = scenarios[0]['fpr']
    p = scenarios[0]['p']
    n = scenarios[0]['n']

    acc = (tpr * p + (1 - fpr) * n) / (p + n)

    tpr_at_fpr_0 = (acc * (p + n) - n) / p
    fpr_at_tpr_1 = 1 - (acc * (p + n) - p) / n

    auc_a = auc_rmin(fpr_at_tpr_1, 1)
    auc_b = auc_rmin(0, tpr_at_fpr_0)

    np.testing.assert_almost_equal(auc_armin(acc, p, n), min(auc_a, auc_b))

def test_acc_min():
    """
    Testing the minimum accuracy estimation
    """
    np.testing.assert_almost_equal(acc_min(0.8, 10, 20), 10 / 30 * 0.8)

def test_acc_rmin():
    """
    Testing the minimum accuracy assuming the curve does not go below
    the random classification line
    """
    np.testing.assert_almost_equal(acc_rmin(0.6, 10, 20), 10 / 30)

    with pytest.raises(ValueError):
        acc_rmin(0.2, 10, 20)

@pytest.mark.parametrize('scenario', auc_scenarios)
def test_acc_max(scenario):
    """
    Testing the maximum accuracy estimation
    """
    auc = scenario['auc']
    p = scenario['p']
    n = scenario['n']

    tpr_at_fpr_0 = auc
    fpr_at_tpr_1 = 1 - auc

    acc_a = (tpr_at_fpr_0 * p + n) / (p + n)
    acc_b = ((1 - fpr_at_tpr_1) * n + p) / (p + n)

    np.testing.assert_almost_equal(acc_max(auc, p, n), max(acc_a, acc_b))

@pytest.mark.parametrize('scenario', auc_scenarios)
def test_acc_rmax(scenario):
    """
    Testing the maximum accuracy estimation assuming the curve does
    not go below the random classification line
    """
    auc = scenario['auc']
    p = scenario['p']
    n = scenario['n']

    if auc < 0.5:
        with pytest.raises(ValueError):
            acc_rmax(auc, p, n)
        return

    tpr_at_fpr_0 = np.sqrt((auc - 0.5) * 2)
    fpr_at_tpr_1 = 1 - np.sqrt((auc - 0.5) * 2)

    acc_a = (tpr_at_fpr_0 * p + n) / (p + n)
    acc_b = ((1 - fpr_at_tpr_1) * n + p) / (p + n)

    np.testing.assert_almost_equal(acc_rmax(auc, p, n), max(acc_a, acc_b))

def test_macc_min():
    """
    Testing the estimation of the minimum of the maximum accuracy
    """
    auc = 0.9
    p = 50
    n = 60

    np.testing.assert_almost_equal(
        macc_min(auc, p, n),
        1 - (np.sqrt(2 * p * n - 2 * auc * p * n)) / (p + n)
    )

    np.testing.assert_almost_equal(
        macc_min(0.1, p, n),
        max(p, n) / (p + n)
    )

    np.testing.assert_almost_equal(
        macc_min(1 - min(p, n)/(2*max(p, n)), p, n),
        max(p, n)/(p + n)
    )

def test_translate_scores():
    """
    Testing the translation of scores
    """

    with pytest.raises(ValueError):
        translate_scores({'sens': 0.6, 'tpr': 0.7})

    with pytest.raises(ValueError):
        translate_scores({'sens': 0.6, 'fnr': 0.3})

    with pytest.raises(ValueError):
        translate_scores({'spec': 0.6, 'tnr': 0.7})

    with pytest.raises(ValueError):
        translate_scores({'spec': 0.6, 'fpr': 0.3})

    tmp = translate_scores({'spec': 0.7})
    np.testing.assert_almost_equal(tmp['fpr'], 0.3)

    tmp = translate_scores({'tnr': 0.7})
    np.testing.assert_almost_equal(tmp['fpr'], 0.3)

    tmp = translate_scores({'sens': 0.7})
    np.testing.assert_almost_equal(tmp['tpr'], 0.7)

    tmp = translate_scores({'fnr': 0.7})
    np.testing.assert_almost_equal(tmp['tpr'], 0.3)

def test_prepare_intervals():
    """
    Testing the preparation of intervals
    """

    intervals = prepare_intervals({'tpr': 0.8, 'f1': 0.7}, eps=0.1)

    np.testing.assert_almost_equal(intervals['tpr'][0], 0.7)
    np.testing.assert_almost_equal(intervals['tpr'][1], 0.9)

    assert len(intervals) == 1

def test_augment_intervals():
    """
    Testing the augmentation of intervals
    """

    intervals = augment_intervals(
        {'tpr': (0.8, 0.81), 'fpr': (0.89, 0.9)},
        p=10,
        n=100
    )
    np.testing.assert_almost_equal(intervals['acc'][0], 18/110)

    intervals = augment_intervals(
        {'tpr': (0.8, 0.81), 'acc': (0.9, 0.91)},
        p=10,
        n=100
    )
    np.testing.assert_almost_equal(intervals['fpr'][0], 1 - (0.91*110 - 0.8*10)/100)

    intervals = augment_intervals(
        {'acc': (0.8, 0.81), 'fpr': (0.1, 0.11)},
        p=40,
        n=100
    )
    np.testing.assert_almost_equal(intervals['tpr'][0], (0.8*140 - (1 - 0.1)*100)/40)


def test_auc_from():
    """
    Testing the auc_from function
    """

    with pytest.raises(ValueError):
        auc_from(scores={'tpr': 0.9}, eps=0.01)

    with pytest.raises(ValueError):
        auc_from(scores={'tpr': 0.9, 'fpr': 0.1}, eps=0.01, lower='grmin')

    with pytest.raises(ValueError):
        auc_from(scores={'tpr': 0.9}, eps=0.01, lower='amin', upper='amax', p=20, n=40)

    for lower in ['min', 'rmin', 'grmin', 'amin', 'armin']:
        for upper in ['max', 'amax', 'maxa']:
            tmp = auc_from(
                scores={'tpr': 0.9, 'fpr': 0.1},
                eps=1e-4,
                p=60,
                n=100,
                lower=lower,
                upper=upper
            )
            assert tmp[0] <= tmp[1]

    with pytest.raises(ValueError):
        auc_from(
            scores={'tpr': 0.9, 'fpr': 0.1},
            eps=1e-4,
            p=10,
            n=100,
            lower='dummy'
        )

    with pytest.raises(ValueError):
        auc_from(
            scores={'tpr': 0.9, 'fpr': 0.1},
            eps=1e-4,
            p=10,
            n=100,
            upper='dummy'
        )

def test_acc_from():
    """
    Testing the acc_from functionality
    """

    with pytest.raises(ValueError):
        acc_from(scores={}, eps=1e-4, p=50, n=100)

    for lower in ['min', 'rmin']:
        for upper in ['max', 'rmax']:
            tmp = acc_from(
                scores={'auc': 0.9},
                eps=1e-4,
                p=20,
                n=50,
                lower=lower,
                upper=upper
            )
            assert tmp[0] <= tmp[1]

    with pytest.raises(ValueError):
        acc_from(
            scores={'auc': 0.9},
            eps=1e-4,
            p=20,
            n=50,
            lower='dummy'
        )

    with pytest.raises(ValueError):
        acc_from(
            scores={'auc': 0.9},
            eps=1e-4,
            p=20,
            n=50,
            upper='dummy'
        )


def test_max_acc_from():
    """
    Testing the max_acc_from functionality
    """

    with pytest.raises(ValueError):
        max_acc_from(scores={}, eps=1e-4, p=50, n=100)

    for lower in ['min']:
        for upper in ['max', 'rmax']:
            tmp = max_acc_from(
                scores={'auc': 0.9},
                eps=1e-4,
                p=20,
                n=50,
                lower=lower,
                upper=upper
            )
            assert tmp[0] <= tmp[1]

    with pytest.raises(ValueError):
        max_acc_from(
            scores={'auc': 0.9},
            eps=1e-4,
            p=20,
            n=50,
            lower='dummy'
        )

    with pytest.raises(ValueError):
        max_acc_from(
            scores={'auc': 0.9},
            eps=1e-4,
            p=20,
            n=50,
            upper='dummy'
        )

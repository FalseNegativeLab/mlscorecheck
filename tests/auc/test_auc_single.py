"""
This module tests the AUC functionalities related to single test sets
"""

import pytest

import numpy as np

from mlscorecheck.auc import (
    roc_min,
    roc_max,
    roc_rmin,
    roc_rmin,
    roc_rmin_grid,
    roc_rmin_grid_correction,
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
    acc_max,
    acc_rmax,
    integrate_roc_curve
)

sens = 0.8
spec = 0.9
fpr = 1 - spec
tpr = sens
p = 21
n = 80
acc = (sens*p + spec*n) / (p + n)

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
    pass

def test_acc_max():
    pass

def test_acc_rmax():
    pass

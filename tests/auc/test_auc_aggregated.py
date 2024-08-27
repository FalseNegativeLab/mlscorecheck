"""
This module tests the aggregated AUC related functionalities
"""

import pytest

import numpy as np

from mlscorecheck.auc import (
    auc_min,
    auc_max,
    auc_rmin,
    auc_maxa,
    auc_amin,
    auc_amax,
    auc_armin,
    acc_min,
    acc_max,
    acc_rmin,
    acc_rmax,
    auc_min_aggregated,
    auc_max_aggregated,
    auc_rmin_aggregated,
    auc_maxa_aggregated,
    auc_amin_aggregated,
    auc_amax_aggregated,
    auc_armin_aggregated,
    acc_min_aggregated,
    acc_rmin_aggregated,
    acc_max_aggregated,
    acc_rmax_aggregated,
    perturbe_solutions
)

random_seeds = list(range(10))

auc_confs = [
    {
     'tpr': 0.8,
     'fpr': 0.1,
     'k': 5
     },
     {
     'tpr': 0.4,
     'fpr': 0.3,
     'k': 5
     },
     {
     'tpr': 0.8,
     'fpr': 0.1,
     'k': 10
     },
     {
     'tpr': 0.4,
     'fpr': 0.3,
     'k': 10
     },
]

auc_acc_confs = [
    {
     'acc': 0.8,
     'ps': [10, 10, 10],
     'ns': [19, 19, 20]
     },
     {
     'acc': 0.4,
     'ps': [10, 10, 10],
     'ns': [19, 19, 20]
     },
     {
     'acc': 0.8,
     'ps': [10, 11, 12, 13, 14],
     'ns': [19, 18, 17, 16, 15]
     },
     {
     'acc': 0.4,
     'ps': [10, 11, 12, 13, 14],
     'ns': [19, 18, 17, 16, 15]
     },
]

acc_auc_confs = [
    {
     'auc': 0.8,
     'ps': [10, 10, 10],
     'ns': [19, 19, 20]
     },
     {
     'auc': 0.6,
     'ps': [10, 10, 10],
     'ns': [19, 19, 20]
     },
     {
     'auc': 0.8,
     'ps': [10, 11, 12, 13, 14],
     'ns': [19, 18, 17, 16, 15]
     },
     {
     'auc': 0.6,
     'ps': [10, 11, 12, 13, 14],
     'ns': [19, 18, 17, 16, 15]
     },
]

@pytest.mark.parametrize('conf', auc_confs)
@pytest.mark.parametrize('random_seed', random_seeds)
def test_auc_min_aggregated(conf, random_seed):
    """
    Testing if perturbation cant find smaller objective

    Args:
        conf (dict): the configuration
        random_seed (int): the random seed
    """

    tpr = conf['tpr']
    fpr = conf['fpr']
    k = conf['k']

    try:
        auc, (fprs, tprs, lower, upper) = auc_min_aggregated(fpr, tpr, k, True)
    except ValueError as exc:
        return
    auc = np.round(auc, 8)

    tmp_fprs = perturbe_solutions(fprs, lower, upper, random_seed)
    tmp_tprs = perturbe_solutions(tprs, lower, upper, random_seed+1)
    auc_tmp = np.mean([auc_min(fpr, tpr) for fpr, tpr in zip(tmp_fprs, tmp_tprs)])
    auc_tmp = np.round(auc_tmp, 8)

    assert auc <= auc_tmp

@pytest.mark.parametrize('conf', auc_confs)
@pytest.mark.parametrize('random_seed', random_seeds)
def test_auc_max_aggregated(conf, random_seed):
    """
    Testing if perturbation cant find greater objective

    Args:
        conf (dict): the configuration
        random_seed (int): the random seed
    """

    tpr = conf['tpr']
    fpr = conf['fpr']
    k = conf['k']

    try:
        auc, (fprs, tprs, lower, upper) = auc_max_aggregated(fpr, tpr, k, True)
    except ValueError as exc:
        return
    auc = np.round(auc, 8)

    tmp_fprs = perturbe_solutions(fprs, lower, upper, random_seed)
    tmp_tprs = perturbe_solutions(tprs, lower, upper, random_seed+1)
    auc_tmp = np.mean([auc_max(fpr, tpr) for fpr, tpr in zip(tmp_fprs, tmp_tprs)])
    auc_tmp = np.round(auc_tmp, 8)

    assert auc >= auc_tmp

@pytest.mark.parametrize('conf', auc_confs)
@pytest.mark.parametrize('random_seed', random_seeds)
def test_auc_rmin_aggregated(conf, random_seed):
    """
    Testing if perturbation cant find smaller objective

    Args:
        conf (dict): the configuration
        random_seed (int): the random seed
    """

    tpr = conf['tpr']
    fpr = conf['fpr']
    k = conf['k']

    try:
        auc, (fprs, tprs, lower, upper) = auc_rmin_aggregated(fpr, tpr, k, True)
    except ValueError as exc:
        return
    auc = np.round(auc, 8)

    tmp_fprs = perturbe_solutions(fprs, lower, upper, random_seed)
    tmp_tprs = perturbe_solutions(tprs, lower, upper, random_seed+1)
    auc_tmp = np.mean([auc_rmin(fpr, tpr) for fpr, tpr in zip(tmp_fprs, tmp_tprs)])
    auc_tmp = np.round(auc_tmp, 8)

    assert auc <= auc_tmp

@pytest.mark.parametrize('conf', auc_acc_confs)
@pytest.mark.parametrize('random_seed', random_seeds)
def test_auc_maxa_aggregated(conf, random_seed):
    """
    Testing if perturbation cant find greater objective

    Args:
        conf (dict): the configuration
        random_seed (int): the random seed
    """

    acc = conf['acc']
    ps = conf['ps']
    ns = conf['ns']

    try:
        auc, (accs, lower, upper) = auc_maxa_aggregated(acc, ps, ns, True)
    except ValueError as exc:
        return
    auc = np.round(auc, 8)

    print(accs, lower, upper)

    tmp = perturbe_solutions(accs, lower, upper, random_seed)
    auc_tmp = np.mean([auc_maxa(acc, p, n) for acc, p, n in zip(tmp, ps, ns)])
    auc_tmp = np.round(auc_tmp, 8)

    assert auc >= auc_tmp

@pytest.mark.parametrize('conf', auc_acc_confs)
@pytest.mark.parametrize('random_seed', random_seeds)
def test_auc_amin_aggregated(conf, random_seed):
    """
    Testing if perturbation cant find smaller objective

    Args:
        conf (dict): the configuration
        random_seed (int): the random seed
    """

    acc = conf['acc']
    ps = conf['ps']
    ns = conf['ns']

    try:
        auc, (accs, _, _, lower, upper) = auc_amin_aggregated(acc, ps, ns, True)
    except ValueError as exc:
        return
    auc = np.round(auc, 8)

    tmp = perturbe_solutions(accs, lower, upper, random_seed)
    auc_tmp = np.mean([auc_amin(acc, p, n) for acc, p, n in zip(tmp, ps, ns)])
    auc_tmp = np.round(auc_tmp, 8)

    assert auc <= auc_tmp

@pytest.mark.parametrize('conf', auc_acc_confs)
@pytest.mark.parametrize('random_seed', random_seeds)
def test_auc_amax_aggregated(conf, random_seed):
    """
    Testing if perturbation cant find greater objective

    Args:
        conf (dict): the configuration
        random_seed (int): the random seed
    """

    acc = conf['acc']
    ps = conf['ps']
    ns = conf['ns']

    try:
        auc, (accs, _, _, lower, upper) = auc_amax_aggregated(acc, ps, ns, True)
    except ValueError as exc:
        return
    auc = np.round(auc, 8)

    tmp = perturbe_solutions(accs, lower, upper, random_seed)
    auc_tmp = np.mean([auc_amax(acc, p, n) for acc, p, n in zip(tmp, ps, ns)])
    auc_tmp = np.round(auc_tmp, 8)

    assert auc >= auc_tmp

@pytest.mark.parametrize('conf', auc_acc_confs)
@pytest.mark.parametrize('random_seed', random_seeds)
def test_auc_armin_aggregated(conf, random_seed):
    """
    Testing if perturbation cant find smaller objective

    Args:
        conf (dict): the configuration
        random_seed (int): the random seed
    """

    acc = conf['acc']
    ps = conf['ps']
    ns = conf['ns']

    try:
        auc, (accs, _, _, lower, upper) = auc_armin_aggregated(acc, ps, ns, True)
    except ValueError as exc:
        return
    auc = np.round(auc, 8)

    tmp = perturbe_solutions(accs, lower, upper, random_seed)
    auc_tmp = np.mean([auc_armin(acc, p, n) for acc, p, n in zip(tmp, ps, ns)])
    auc_tmp = np.round(auc_tmp, 8)

    assert auc <= auc_tmp

@pytest.mark.parametrize('conf', acc_auc_confs)
@pytest.mark.parametrize('random_seed', random_seeds)
def test_acc_min_aggregated(conf, random_seed):
    """
    Testing if perturbation cant find smaller objective

    Args:
        conf (dict): the configuration
        random_seed (int): the random seed
    """

    auc = conf['auc']
    ps = conf['ps']
    ns = conf['ns']

    try:
        acc, (aucs, ps, ns, lower, upper) = acc_min_aggregated(auc, ps, ns, True)
    except ValueError as exc:
        return
    acc = np.round(acc, 8)

    tmp = perturbe_solutions(aucs, lower, upper, random_seed)
    acc_tmp = np.mean([acc_min(acc, p, n) for acc, p, n in zip(tmp, ps, ns)])
    acc_tmp = np.round(acc_tmp, 8)

    assert acc <= acc_tmp

@pytest.mark.parametrize('conf', acc_auc_confs)
@pytest.mark.parametrize('random_seed', random_seeds)
def test_acc_max_aggregated(conf, random_seed):
    """
    Testing if perturbation cant find greater objective

    Args:
        conf (dict): the configuration
        random_seed (int): the random seed
    """

    auc = conf['auc']
    ps = conf['ps']
    ns = conf['ns']

    try:
        acc, (aucs, ps, ns, lower, upper) = acc_max_aggregated(auc, ps, ns, True)
    except ValueError as exc:
        return
    acc = np.round(acc, 8)

    tmp = perturbe_solutions(aucs, lower, upper, random_seed)
    acc_tmp = np.mean([acc_max(acc, p, n) for acc, p, n in zip(tmp, ps, ns)])
    acc_tmp = np.round(acc_tmp, 8)

    assert acc >= acc_tmp

@pytest.mark.parametrize('conf', acc_auc_confs)
@pytest.mark.parametrize('random_seed', random_seeds)
def test_acc_rmin_aggregated(conf, random_seed):
    """
    Testing if perturbation cant find smaller objective

    Args:
        conf (dict): the configuration
        random_seed (int): the random seed
    """

    auc = conf['auc']
    ps = conf['ps']
    ns = conf['ns']

    try:
        acc, (aucs, lower, upper) = acc_rmin_aggregated(auc, ps, ns, True)
    except ValueError as exc:
        return
    acc = np.round(acc, 8)

    tmp = perturbe_solutions(aucs, lower, upper, random_seed)
    acc_tmp = np.mean([acc_rmin(acc, p, n) for acc, p, n in zip(tmp, ps, ns)])
    acc_tmp = np.round(acc_tmp, 8)

    assert acc <= acc_tmp

@pytest.mark.parametrize('conf', acc_auc_confs)
@pytest.mark.parametrize('random_seed', random_seeds)
def test_acc_rmax_aggregated(conf, random_seed):
    """
    Testing if perturbation cant find greater objective

    Args:
        conf (dict): the configuration
        random_seed (int): the random seed
    """

    auc = conf['auc']
    ps = conf['ps']
    ns = conf['ns']

    try:
        acc, (aucs, ps, ns, lower, upper) = acc_rmax_aggregated(auc, ps, ns, True)
    except ValueError as exc:
        return
    acc = np.round(acc, 8)

    tmp = perturbe_solutions(aucs, lower, upper, random_seed)
    acc_tmp = np.mean([acc_rmax(acc, p, n) for acc, p, n in zip(tmp, ps, ns)])
    acc_tmp = np.round(acc_tmp, 8)

    assert acc >= acc_tmp
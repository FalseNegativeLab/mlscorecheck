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
    macc_min,
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
    macc_min_aggregated,
    perturb_solutions,
    multi_perturb_solutions,
    R,
    F,
    auc_from_aggregated,
    acc_from_aggregated,
    max_acc_from_aggregated
)

random_seeds = list(range(30))

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

def test_r_f():
    """
    Testing the R and F functions
    """

    lower = np.array([0.1, 0.2, 0.3])
    upper = np.array([0.9, 0.8, 0.8])
    x = 0.78

    r = 1 - R(x, lower=lower, upper=upper, k=3)
    f = F(R(1 - x, lower=1 - F(upper), upper=1 - F(lower), k=3))

    np.testing.assert_array_almost_equal(r, f)

def test_multi_perturb():
    """
    Testing the iterated perturbation
    """

    x = np.array([0.2, 0.8, 0.5])
    lower = np.array([0.1, 0.2, 0.3])
    upper = np.array([0.8, 0.9, 0.7])

    x_new = multi_perturb_solutions(5, x, lower, upper, 5)

    np.testing.assert_almost_equal(np.mean(x), np.mean(x_new))

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

    #try:
    auc, (fprs, tprs, lower, upper) = auc_min_aggregated(fpr, tpr, k, True)
    #except ValueError as exc:
    #    return
    auc = np.round(auc, 8)

    tmp_fprs = perturb_solutions(fprs, lower, upper, random_seed)
    tmp_tprs = perturb_solutions(tprs, lower, upper, random_seed+1)
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

    #try:
    auc, (fprs, tprs, lower, upper) = auc_max_aggregated(fpr, tpr, k, True)
    #except ValueError as exc:
    #    return
    auc = np.round(auc, 8)

    tmp_fprs = perturb_solutions(fprs, lower, upper, random_seed)
    tmp_tprs = perturb_solutions(tprs, lower, upper, random_seed+1)
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

    if tpr >= fpr:
        auc, (fprs, tprs, lower, upper) = auc_rmin_aggregated(fpr, tpr, k, True)
    else:
        with pytest.raises(ValueError):
            auc, (fprs, tprs, lower, upper) = auc_rmin_aggregated(fpr, tpr, k, True)
        return

    auc = np.round(auc, 8)

    tmp_fprs = perturb_solutions(fprs, lower, tprs, random_seed)
    tmp_tprs = perturb_solutions(tprs, tmp_fprs, upper, random_seed+1)
    auc_tmp = np.mean([auc_rmin(fpr, tpr) for fpr, tpr in zip(tmp_fprs, tmp_tprs)])
    auc_tmp = np.round(auc_tmp, 8)

    assert auc <= auc_tmp

def test_auc_rmin_aggregated_exception():
    """
    Testing the exception for auc_rmin_aggregated
    """

    with pytest.raises(ValueError):
        auc_rmin_aggregated(0.8, 0.1, 5)

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

    if acc < np.mean([max(p, n)/(p + n) for p, n in zip(ps, ns)]):
        with pytest.raises(ValueError):
            auc, (accs, lower, upper) = auc_maxa_aggregated(acc, ps, ns, True)
        return
    else:
        auc, (accs, lower, upper) = auc_maxa_aggregated(acc, ps, ns, True)

    auc = np.round(auc, 8)

    print(accs, lower, upper)

    tmp = perturb_solutions(accs, lower, upper, random_seed)
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

    auc, (accs, _, _, lower, upper) = auc_amin_aggregated(acc, ps, ns, True)
    auc = np.round(auc, 8)

    tmp = perturb_solutions(accs, lower, upper, random_seed)
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

    auc, (accs, _, _, lower, upper) = auc_amax_aggregated(acc, ps, ns, True)
    auc = np.round(auc, 8)

    tmp = perturb_solutions(accs, lower, upper, random_seed)
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

    lower = np.array([min(p, n)/(p + n) for p, n in zip(ps, ns)])

    if acc < np.mean(lower):
        with pytest.raises(ValueError):
            auc, (accs, _, _, lower, upper) = auc_armin_aggregated(acc, ps, ns, True)
        return
    else:
        auc, (accs, _, _, lower, upper) = auc_armin_aggregated(acc, ps, ns, True)
    auc = np.round(auc, 8)

    tmp = perturb_solutions(accs, lower, upper, random_seed)
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

    acc, (aucs, ps, ns, lower, upper) = acc_min_aggregated(auc, ps, ns, True)
    acc = np.round(acc, 8)

    tmp = perturb_solutions(aucs, lower, upper, random_seed)
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

    acc, (aucs, ps, ns, lower, upper) = acc_max_aggregated(auc, ps, ns, True)
    acc = np.round(acc, 8)

    tmp = perturb_solutions(aucs, lower, upper, random_seed)
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

    acc, (aucs, lower, upper) = acc_rmin_aggregated(auc, ps, ns, True)
    acc = np.round(acc, 8)

    tmp = perturb_solutions(aucs, lower, upper, random_seed)
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

    if auc >= 0.5:
        acc, (aucs, ps, ns, lower, upper) = acc_rmax_aggregated(auc, ps, ns, True)
    else:
        with pytest.raises(ValueError):
            acc, (aucs, ps, ns, lower, upper) = acc_rmax_aggregated(auc, ps, ns, True)
        return
    acc = np.round(acc, 8)

    tmp = perturb_solutions(aucs, lower, upper, random_seed)
    acc_tmp = np.mean([acc_rmax(acc, p, n) for acc, p, n in zip(tmp, ps, ns)])
    acc_tmp = np.round(acc_tmp, 8)

    assert acc >= acc_tmp

@pytest.mark.parametrize('conf', acc_auc_confs)
@pytest.mark.parametrize('random_seed', random_seeds)
def test_macc_min_aggregated(conf, random_seed):
    """
    Testing if perturbation cant find smaller objective

    Args:
        conf (dict): the configuration
        random_seed (int): the random seed
    """

    auc = conf['auc']
    ps = conf['ps']
    ns = conf['ns']

    lower_bounds = 1.0 - np.array([min(p, n)/(2*max(p, n)) for p, n in zip(ps, ns)])
    if auc < np.mean(lower_bounds):
        with pytest.raises(ValueError):
            acc, (aucs, lower, upper) = macc_min_aggregated(auc, ps, ns, True)
        return
    else:
        acc, (aucs, lower, upper) = macc_min_aggregated(auc, ps, ns, True)
    acc = np.round(acc, 8)

    tmp = perturb_solutions(aucs, lower, upper, random_seed)

    print('aucs', aucs)
    print('lowe', lower)
    print('tmp', tmp)

    acc_tmp = np.mean([macc_min(acc, p, n) for acc, p, n in zip(tmp, ps, ns)])
    acc_tmp = np.round(acc_tmp, 8)

    assert acc <= acc_tmp

def test_auc_from_aggregated():
    """
    Testing the auc_from_aggregated function
    """

    ps = [60, 70, 80]
    ns = [80, 90, 80]

    with pytest.raises(ValueError):
        auc_from_aggregated(scores={'tpr': 0.9}, eps=0.01, k=5)

    with pytest.raises(ValueError):
        auc_from_aggregated(scores={'tpr': 0.9, 'fpr': 0.1}, eps=0.01, k=5, lower='amin')

    with pytest.raises(ValueError):
        auc_from_aggregated(
            scores={'tpr': 0.9}, 
            eps=0.01, 
            k=len(ps),
            lower='amin', 
            upper='amax', 
            ps=ps, 
            ns=ns
        )

    for lower in ['min', 'rmin', 'amin', 'armin']:
        for upper in ['max', 'amax', 'maxa']:
            print(lower, upper)
            tmp = auc_from_aggregated(
                scores={'tpr': 0.9, 'fpr': 0.1},
                eps=1e-4,
                k=len(ps),
                ps=ps,
                ns=ns,
                lower=lower,
                upper=upper
            )
            assert tmp[0] <= tmp[1]

    with pytest.raises(ValueError):
        auc_from_aggregated(
            scores={'tpr': 0.9, 'fpr': 0.1},
            eps=1e-4,
            k=len(ps),
            ps=ps,
            ns=ns,
            lower='dummy'
        )

    with pytest.raises(ValueError):
        auc_from_aggregated(
            scores={'tpr': 0.9, 'fpr': 0.1},
            eps=1e-4,
            k=len(ps),
            ps=ps,
            ns=ns,
            upper='dummy'
        )

def test_acc_from_aggregated():
    """
    Testing the acc_from_aggregated functionality
    """

    ps = [60, 70, 80]
    ns = [80, 90, 80]

    with pytest.raises(ValueError):
        acc_from_aggregated(scores={}, eps=1e-4, ps=ps, ns=ns)

    for lower in ['min', 'rmin']:
        for upper in ['max', 'rmax']:
            tmp = acc_from_aggregated(
                scores={'auc': 0.9},
                eps=1e-4,
                ps=ps,
                ns=ns,
                lower=lower,
                upper=upper
            )
            assert tmp[0] <= tmp[1]

    with pytest.raises(ValueError):
        acc_from_aggregated(
            scores={'auc': 0.9},
            eps=1e-4,
            ps=ps,
            ns=ns,
            lower='dummy'
        )

    with pytest.raises(ValueError):
        acc_from_aggregated(
            scores={'auc': 0.9},
            eps=1e-4,
            ps=ps,
            ns=ns,
            upper='dummy'
        )


def test_max_acc_from_aggregated():
    """
    Testing the max_acc_from functionality
    """

    ps = [60, 70, 80]
    ns = [80, 90, 80]

    with pytest.raises(ValueError):
        max_acc_from_aggregated(scores={}, eps=1e-4, ps=ps, ns=ns)

    for lower in ['min']:
        for upper in ['max', 'rmax']:
            tmp = max_acc_from_aggregated(
                scores={'auc': 0.9},
                eps=1e-4,
                ps=ps,
                ns=ns,
                lower=lower,
                upper=upper
            )
            assert tmp[0] <= tmp[1]

    with pytest.raises(ValueError):
        max_acc_from_aggregated(
            scores={'auc': 0.9},
            eps=1e-4,
            ps=ps,
            ns=ns,
            lower='dummy'
        )

    with pytest.raises(ValueError):
        max_acc_from_aggregated(
            scores={'auc': 0.9},
            eps=1e-4,
            ps=ps,
            ns=ns,
            upper='dummy'
        )

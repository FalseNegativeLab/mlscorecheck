"""
This module contains the tp and tn solutions.
This is a generated file, do not edit.
"""

# pylint: disable=line-too-long
# pylint: disable=too-many-lines

from ._utils import is_zero, unify_results
from ._interval import sqrt

__all__ = [
"acc_tp",
"acc_tn",
"sens_tp",
"spec_tn",
"ppv_tp",
"ppv_tn",
"npv_tp",
"npv_tn",
"fbp_tp",
"fbp_tn",
"f1p_tp",
"f1p_tn",
"fbn_tp",
"fbn_tn",
"f1n_tp",
"f1n_tn",
"gm_tp",
"gm_tn",
"fm_tp_0",
"fm_tp_1",
"fm_tp",
"fm_tn",
"upm_tp_0",
"upm_tp_1",
"upm_tp",
"upm_tn_0",
"upm_tn_1",
"upm_tn",
"mk_tp_0",
"mk_tp_1",
"mk_tp",
"mk_tn_0",
"mk_tn_1",
"mk_tn",
"lrp_tp",
"lrp_tn",
"lrn_tp",
"lrn_tn",
"bm_tp",
"bm_tn",
"pt_tp_0",
"pt_tp_1",
"pt_tp",
"pt_tn_0",
"pt_tn_1",
"pt_tn",
"dor_tp",
"dor_tn",
"ji_tp",
"ji_tn",
"bacc_tp",
"bacc_tn",
"kappa_tp",
"kappa_tn",
"mcc_tp_0",
"mcc_tp_1",
"mcc_tp",
"mcc_tn_0",
"mcc_tn_1",
"mcc_tn"]

def acc_tp(*, tn, n, acc, p, **kwargs):
    """
    Solves tp from the score acc

    Args:
        tn (int): the tn count
        n (int): the n count
        acc (float|Interval|IntervalUnion): the value or interval for the score acc
        p (int): the p count
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tp
    """
    _ = kwargs
    return acc*n + acc*p - tn

def acc_tn(*, n, acc, p, tp, **kwargs):
    """
    Solves tn from the score acc

    Args:
        n (int): the n count
        acc (float|Interval|IntervalUnion): the value or interval for the score acc
        p (int): the p count
        tp (int): the tp count
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tn
    """
    _ = kwargs
    return acc*n + acc*p - tp

def sens_tp(*, sens, p, **kwargs):
    """
    Solves tp from the score sens

    Args:
        sens (float|Interval|IntervalUnion): the value or interval for the score sens
        p (int): the p count
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tp
    """
    _ = kwargs
    return p*sens

def spec_tn(*, spec, n, **kwargs):
    """
    Solves tn from the score spec

    Args:
        spec (float|Interval|IntervalUnion): the value or interval for the score spec
        n (int): the n count
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tn
    """
    _ = kwargs
    return n*spec

def ppv_tp(*, n, tn, ppv, **kwargs):
    """
    Solves tp from the score ppv

    Args:
        n (int): the n count
        tn (int): the tn count
        ppv (float|Interval|IntervalUnion): the value or interval for the score ppv
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tp
    """
    _ = kwargs
    if is_zero(ppv - 1):
        return None
    return ppv*(-n + tn)/(ppv - 1)

def ppv_tn(*, ppv, n, tp, **kwargs):
    """
    Solves tn from the score ppv

    Args:
        ppv (float|Interval|IntervalUnion): the value or interval for the score ppv
        n (int): the n count
        tp (int): the tp count
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tn
    """
    _ = kwargs
    if is_zero(ppv):
        return None
    return n + tp - tp/ppv

def npv_tp(*, npv, p, tn, **kwargs):
    """
    Solves tp from the score npv

    Args:
        npv (float|Interval|IntervalUnion): the value or interval for the score npv
        p (int): the p count
        tn (int): the tn count
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tp
    """
    _ = kwargs
    if is_zero(npv):
        return None
    return p + tn - tn/npv

def npv_tn(*, p, npv, tp, **kwargs):
    """
    Solves tn from the score npv

    Args:
        p (int): the p count
        npv (float|Interval|IntervalUnion): the value or interval for the score npv
        tp (int): the tp count
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tn
    """
    _ = kwargs
    if is_zero(npv - 1):
        return None
    return npv*(-p + tp)/(npv - 1)

def fbp_tp(*, tn, n, beta_positive, p, fbp, **kwargs):
    """
    Solves tp from the score fbp

    Args:
        tn (int): the tn count
        n (int): the n count
        beta_positive (float): the beta_positive parameter of the score
        p (int): the p count
        fbp (float|Interval|IntervalUnion): the value or interval for the score fbp
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tp
    """
    _ = kwargs
    if is_zero(beta_positive**2 - fbp + 1):
        return None
    return fbp*(beta_positive**2*p + n - tn)/(beta_positive**2 - fbp + 1)

def fbp_tn(*, n, beta_positive, p, fbp, tp, **kwargs):
    """
    Solves tn from the score fbp

    Args:
        n (int): the n count
        beta_positive (float): the beta_positive parameter of the score
        p (int): the p count
        fbp (float|Interval|IntervalUnion): the value or interval for the score fbp
        tp (int): the tp count
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tn
    """
    _ = kwargs
    if is_zero(fbp):
        return None
    return (-beta_positive**2*tp + fbp*(beta_positive**2*p + n + tp) - tp)/fbp

def f1p_tp(*, f1p, p, tn, n, **kwargs):
    """
    Solves tp from the score f1p

    Args:
        f1p (float|Interval|IntervalUnion): the value or interval for the score f1p
        p (int): the p count
        tn (int): the tn count
        n (int): the n count
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tp
    """
    _ = kwargs
    if is_zero(f1p - 2):
        return None
    return f1p*(-n - p + tn)/(f1p - 2)

def f1p_tn(*, f1p, n, p, tp, **kwargs):
    """
    Solves tn from the score f1p

    Args:
        f1p (float|Interval|IntervalUnion): the value or interval for the score f1p
        n (int): the n count
        p (int): the p count
        tp (int): the tp count
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tn
    """
    _ = kwargs
    if is_zero(f1p):
        return None
    return n + p + tp - 2*tp/f1p

def fbn_tp(*, fbn, tn, n, beta_negative, p, **kwargs):
    """
    Solves tp from the score fbn

    Args:
        fbn (float|Interval|IntervalUnion): the value or interval for the score fbn
        tn (int): the tn count
        n (int): the n count
        beta_negative (float): the beta_negative parameter of the score
        p (int): the p count
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tp
    """
    _ = kwargs
    if is_zero(fbn):
        return None
    return (-beta_negative**2*tn + fbn*(beta_negative**2*n + p + tn) - tn)/fbn

def fbn_tn(*, fbn, n, beta_negative, p, tp, **kwargs):
    """
    Solves tn from the score fbn

    Args:
        fbn (float|Interval|IntervalUnion): the value or interval for the score fbn
        n (int): the n count
        beta_negative (float): the beta_negative parameter of the score
        p (int): the p count
        tp (int): the tp count
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tn
    """
    _ = kwargs
    if is_zero(beta_negative**2 - fbn + 1):
        return None
    return fbn*(beta_negative**2*n + p - tp)/(beta_negative**2 - fbn + 1)

def f1n_tp(*, tn, n, f1n, p, **kwargs):
    """
    Solves tp from the score f1n

    Args:
        tn (int): the tn count
        n (int): the n count
        f1n (float|Interval|IntervalUnion): the value or interval for the score f1n
        p (int): the p count
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tp
    """
    _ = kwargs
    if is_zero(f1n):
        return None
    return n + p + tn - 2*tn/f1n

def f1n_tn(*, f1n, p, n, tp, **kwargs):
    """
    Solves tn from the score f1n

    Args:
        f1n (float|Interval|IntervalUnion): the value or interval for the score f1n
        p (int): the p count
        n (int): the n count
        tp (int): the tp count
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tn
    """
    _ = kwargs
    if is_zero(f1n - 2):
        return None
    return f1n*(-n - p + tp)/(f1n - 2)

def gm_tp(*, tn, gm, p, n, **kwargs):
    """
    Solves tp from the score gm

    Args:
        tn (int): the tn count
        gm (float|Interval|IntervalUnion): the value or interval for the score gm
        p (int): the p count
        n (int): the n count
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tp
    """
    _ = kwargs
    if is_zero(tn):
        return None
    return gm**2*n*p/tn

def gm_tn(*, gm, p, n, tp, **kwargs):
    """
    Solves tn from the score gm

    Args:
        gm (float|Interval|IntervalUnion): the value or interval for the score gm
        p (int): the p count
        n (int): the n count
        tp (int): the tp count
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tn
    """
    _ = kwargs
    if is_zero(tp):
        return None
    return gm**2*n*p/tp

def fm_tp_0(*, tn, n, fm, p, **kwargs):
    """
    Solves tp from the score fm

    Args:
        tn (int): the tn count
        n (int): the n count
        fm (float|Interval|IntervalUnion): the value or interval for the score fm
        p (int): the p count
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tp
    """
    _ = kwargs
    return fm*(fm*p - sqrt(p)*sqrt(fm**2*p + 4*n - 4*tn))/2

def fm_tp_1(*, tn, n, fm, p, **kwargs):
    """
    Solves tp from the score fm

    Args:
        tn (int): the tn count
        n (int): the n count
        fm (float|Interval|IntervalUnion): the value or interval for the score fm
        p (int): the p count
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tp
    """
    _ = kwargs
    return fm*(fm*p + sqrt(p)*sqrt(fm**2*p + 4*n - 4*tn))/2

def fm_tp(*, tn, n, fm, p, **kwargs):
    """
    Solves tp from the score fm

    Args:
        tn (int): the tn count
        n (int): the n count
        fm (float|Interval|IntervalUnion): the value or interval for the score fm
        p (int): the p count
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tp
    """
    _ = kwargs
    return unify_results([fm_tp_0(tn=tn, n=n, fm=fm, p=p),
                          fm_tp_1(tn=tn, n=n, fm=fm, p=p)])
def fm_tn(*, n, fm, p, tp, **kwargs):
    """
    Solves tn from the score fm

    Args:
        n (int): the n count
        fm (float|Interval|IntervalUnion): the value or interval for the score fm
        p (int): the p count
        tp (int): the tp count
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tn
    """
    _ = kwargs
    if is_zero(fm**2) or is_zero(p):
        return None
    return n + tp - tp**2/(fm**2*p)

def upm_tp_0(*, tn, n, upm, p, **kwargs):
    """
    Solves tp from the score upm

    Args:
        tn (int): the tn count
        n (int): the n count
        upm (float|Interval|IntervalUnion): the value or interval for the score upm
        p (int): the p count
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tp
    """
    _ = kwargs
    if is_zero(upm):
        return None
    return n/2 + p/2 + tn - 2*tn/upm - sqrt(n**2*upm**2 + 2*n*p*upm**2 + 8*n*tn*upm**2 - 8*n*tn*upm + p**2*upm**2 + 8*p*tn*upm**2 - 8*p*tn*upm - 16*tn**2*upm + 16*tn**2)/(2*upm)

def upm_tp_1(*, tn, n, upm, p, **kwargs):
    """
    Solves tp from the score upm

    Args:
        tn (int): the tn count
        n (int): the n count
        upm (float|Interval|IntervalUnion): the value or interval for the score upm
        p (int): the p count
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tp
    """
    _ = kwargs
    if is_zero(upm):
        return None
    return n/2 + p/2 + tn - 2*tn/upm + sqrt(n**2*upm**2 + 2*n*p*upm**2 + 8*n*tn*upm**2 - 8*n*tn*upm + p**2*upm**2 + 8*p*tn*upm**2 - 8*p*tn*upm - 16*tn**2*upm + 16*tn**2)/(2*upm)

def upm_tp(*, tn, n, upm, p, **kwargs):
    """
    Solves tp from the score upm

    Args:
        tn (int): the tn count
        n (int): the n count
        upm (float|Interval|IntervalUnion): the value or interval for the score upm
        p (int): the p count
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tp
    """
    _ = kwargs
    return unify_results([upm_tp_0(tn=tn, n=n, upm=upm, p=p),
                          upm_tp_1(tn=tn, n=n, upm=upm, p=p)])
def upm_tn_0(*, n, upm, p, tp, **kwargs):
    """
    Solves tn from the score upm

    Args:
        n (int): the n count
        upm (float|Interval|IntervalUnion): the value or interval for the score upm
        p (int): the p count
        tp (int): the tp count
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tn
    """
    _ = kwargs
    if is_zero(upm):
        return None
    return n/2 + p/2 + tp - 2*tp/upm - sqrt(n**2*upm**2 + 2*n*p*upm**2 + 8*n*tp*upm**2 - 8*n*tp*upm + p**2*upm**2 + 8*p*tp*upm**2 - 8*p*tp*upm - 16*tp**2*upm + 16*tp**2)/(2*upm)

def upm_tn_1(*, n, upm, p, tp, **kwargs):
    """
    Solves tn from the score upm

    Args:
        n (int): the n count
        upm (float|Interval|IntervalUnion): the value or interval for the score upm
        p (int): the p count
        tp (int): the tp count
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tn
    """
    _ = kwargs
    if is_zero(upm):
        return None
    return n/2 + p/2 + tp - 2*tp/upm + sqrt(n**2*upm**2 + 2*n*p*upm**2 + 8*n*tp*upm**2 - 8*n*tp*upm + p**2*upm**2 + 8*p*tp*upm**2 - 8*p*tp*upm - 16*tp**2*upm + 16*tp**2)/(2*upm)

def upm_tn(*, n, upm, p, tp, **kwargs):
    """
    Solves tn from the score upm

    Args:
        n (int): the n count
        upm (float|Interval|IntervalUnion): the value or interval for the score upm
        p (int): the p count
        tp (int): the tp count
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tn
    """
    _ = kwargs
    return unify_results([upm_tn_0(n=n, upm=upm, p=p, tp=tp),
                          upm_tn_1(n=n, upm=upm, p=p, tp=tp)])
def mk_tp_0(*, n, tn, mk, p, **kwargs):
    """
    Solves tp from the score mk

    Args:
        n (int): the n count
        tn (int): the tn count
        mk (float|Interval|IntervalUnion): the value or interval for the score mk
        p (int): the p count
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tp
    """
    _ = kwargs
    if is_zero(mk):
        return None
    return (-mk*n + mk*p + 2*mk*tn - n - sqrt(mk**2*n**2 + 2*mk**2*n*p + mk**2*p**2 + 2*mk*n**2 + 2*mk*n*p - 4*mk*n*tn - 4*mk*p*tn + n**2))/(2*mk)

def mk_tp_1(*, tn, n, mk, p, **kwargs):
    """
    Solves tp from the score mk

    Args:
        tn (int): the tn count
        n (int): the n count
        mk (float|Interval|IntervalUnion): the value or interval for the score mk
        p (int): the p count
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tp
    """
    _ = kwargs
    if is_zero(mk):
        return None
    return (-mk*n + mk*p + 2*mk*tn - n + sqrt(mk**2*n**2 + 2*mk**2*n*p + mk**2*p**2 + 2*mk*n**2 + 2*mk*n*p - 4*mk*n*tn - 4*mk*p*tn + n**2))/(2*mk)

def mk_tp(*, tn, n, mk, p, **kwargs):
    """
    Solves tp from the score mk

    Args:
        tn (int): the tn count
        n (int): the n count
        mk (float|Interval|IntervalUnion): the value or interval for the score mk
        p (int): the p count
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tp
    """
    _ = kwargs
    return unify_results([mk_tp_0(tn=tn, n=n, mk=mk, p=p),
                          mk_tp_1(tn=tn, n=n, mk=mk, p=p)])
def mk_tn_0(*, n, mk, p, tp, **kwargs):
    """
    Solves tn from the score mk

    Args:
        n (int): the n count
        mk (float|Interval|IntervalUnion): the value or interval for the score mk
        p (int): the p count
        tp (int): the tp count
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tn
    """
    _ = kwargs
    if is_zero(mk):
        return None
    return (mk*n - mk*p + 2*mk*tp - p - sqrt(mk**2*n**2 + 2*mk**2*n*p + mk**2*p**2 + 2*mk*n*p - 4*mk*n*tp + 2*mk*p**2 - 4*mk*p*tp + p**2))/(2*mk)

def mk_tn_1(*, n, mk, p, tp, **kwargs):
    """
    Solves tn from the score mk

    Args:
        n (int): the n count
        mk (float|Interval|IntervalUnion): the value or interval for the score mk
        p (int): the p count
        tp (int): the tp count
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tn
    """
    _ = kwargs
    if is_zero(mk):
        return None
    return (mk*n - mk*p + 2*mk*tp - p + sqrt(mk**2*n**2 + 2*mk**2*n*p + mk**2*p**2 + 2*mk*n*p - 4*mk*n*tp + 2*mk*p**2 - 4*mk*p*tp + p**2))/(2*mk)

def mk_tn(*, n, mk, p, tp, **kwargs):
    """
    Solves tn from the score mk

    Args:
        n (int): the n count
        mk (float|Interval|IntervalUnion): the value or interval for the score mk
        p (int): the p count
        tp (int): the tp count
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tn
    """
    _ = kwargs
    return unify_results([mk_tn_0(n=n, mk=mk, p=p, tp=tp),
                          mk_tn_1(n=n, mk=mk, p=p, tp=tp)])
def lrp_tp(*, lrp, n, tn, p, **kwargs):
    """
    Solves tp from the score lrp

    Args:
        lrp (float|Interval|IntervalUnion): the value or interval for the score lrp
        n (int): the n count
        tn (int): the tn count
        p (int): the p count
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tp
    """
    _ = kwargs
    if is_zero(n):
        return None
    return lrp*p*(n - tn)/n

def lrp_tn(*, lrp, n, p, tp, **kwargs):
    """
    Solves tn from the score lrp

    Args:
        lrp (float|Interval|IntervalUnion): the value or interval for the score lrp
        n (int): the n count
        p (int): the p count
        tp (int): the tp count
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tn
    """
    _ = kwargs
    if is_zero(lrp) or is_zero(p):
        return None
    return n - n*tp/(lrp*p)

def lrn_tp(*, lrn, n, tn, p, **kwargs):
    """
    Solves tp from the score lrn

    Args:
        lrn (float|Interval|IntervalUnion): the value or interval for the score lrn
        n (int): the n count
        tn (int): the tn count
        p (int): the p count
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tp
    """
    _ = kwargs
    if is_zero(n):
        return None
    return p*(-lrn*tn + n)/n

def lrn_tn(*, lrn, n, p, tp, **kwargs):
    """
    Solves tn from the score lrn

    Args:
        lrn (float|Interval|IntervalUnion): the value or interval for the score lrn
        n (int): the n count
        p (int): the p count
        tp (int): the tp count
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tn
    """
    _ = kwargs
    if is_zero(p) or is_zero(lrn):
        return None
    return n*(p - tp)/(lrn*p)

def bm_tp(*, n, tn, p, bm, **kwargs):
    """
    Solves tp from the score bm

    Args:
        n (int): the n count
        tn (int): the tn count
        p (int): the p count
        bm (float|Interval|IntervalUnion): the value or interval for the score bm
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tp
    """
    _ = kwargs
    if is_zero(n):
        return None
    return p*(n*(bm + 1) - tn)/n

def bm_tn(*, n, p, bm, tp, **kwargs):
    """
    Solves tn from the score bm

    Args:
        n (int): the n count
        p (int): the p count
        bm (float|Interval|IntervalUnion): the value or interval for the score bm
        tp (int): the tp count
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tn
    """
    _ = kwargs
    if is_zero(p):
        return None
    return n*(p*(bm + 1) - tp)/p

def pt_tp_0(*, tn, p, n, **kwargs):
    """
    Solves tp from the score pt

    Args:
        tn (int): the tn count
        p (int): the p count
        n (int): the n count
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tp
    """
    _ = kwargs
    if is_zero(n):
        return None
    return p*(n - tn)/n

def pt_tp_1(*, pt, n, tn, p, **kwargs):
    """
    Solves tp from the score pt

    Args:
        pt (float|Interval|IntervalUnion): the value or interval for the score pt
        n (int): the n count
        tn (int): the tn count
        p (int): the p count
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tp
    """
    _ = kwargs
    if is_zero(pt**2) or is_zero(n):
        return None
    return p*(n*pt**2 - 2*n*pt + n - pt**2*tn + 2*pt*tn - tn)/(n*pt**2)

def pt_tp(*, pt, n, tn, p, **kwargs):
    """
    Solves tp from the score pt

    Args:
        pt (float|Interval|IntervalUnion): the value or interval for the score pt
        n (int): the n count
        tn (int): the tn count
        p (int): the p count
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tp
    """
    _ = kwargs
    return unify_results([pt_tp_0(pt=pt, n=n, tn=tn, p=p),
                          pt_tp_1(pt=pt, n=n, tn=tn, p=p)])
def pt_tn_0(*, p, n, tp, **kwargs):
    """
    Solves tn from the score pt

    Args:
        p (int): the p count
        n (int): the n count
        tp (int): the tp count
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tn
    """
    _ = kwargs
    if is_zero(p):
        return None
    return n*(p - tp)/p

def pt_tn_1(*, pt, n, p, tp, **kwargs):
    """
    Solves tn from the score pt

    Args:
        pt (float|Interval|IntervalUnion): the value or interval for the score pt
        n (int): the n count
        p (int): the p count
        tp (int): the tp count
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tn
    """
    _ = kwargs
    if is_zero(p) or is_zero(pt**2 - 2*pt + 1):
        return None
    return n*(p*pt**2 - 2*p*pt + p - pt**2*tp)/(p*(pt**2 - 2*pt + 1))

def pt_tn(*, pt, n, p, tp, **kwargs):
    """
    Solves tn from the score pt

    Args:
        pt (float|Interval|IntervalUnion): the value or interval for the score pt
        n (int): the n count
        p (int): the p count
        tp (int): the tp count
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tn
    """
    _ = kwargs
    return unify_results([pt_tn_0(pt=pt, n=n, p=p, tp=tp),
                          pt_tn_1(pt=pt, n=n, p=p, tp=tp)])
def dor_tp(*, dor, tn, n, p, **kwargs):
    """
    Solves tp from the score dor

    Args:
        dor (float|Interval|IntervalUnion): the value or interval for the score dor
        tn (int): the tn count
        n (int): the n count
        p (int): the p count
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tp
    """
    _ = kwargs
    if is_zero(dor*n - dor*tn + tn):
        return None
    return dor*p*(n - tn)/(dor*n - dor*tn + tn)

def dor_tn(*, dor, n, p, tp, **kwargs):
    """
    Solves tn from the score dor

    Args:
        dor (float|Interval|IntervalUnion): the value or interval for the score dor
        n (int): the n count
        p (int): the p count
        tp (int): the tp count
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tn
    """
    _ = kwargs
    if is_zero(dor*p - dor*tp + tp):
        return None
    return dor*n*(p - tp)/(dor*p - dor*tp + tp)

def ji_tp(*, ji, p, tn, n, **kwargs):
    """
    Solves tp from the score ji

    Args:
        ji (float|Interval|IntervalUnion): the value or interval for the score ji
        p (int): the p count
        tn (int): the tn count
        n (int): the n count
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tp
    """
    _ = kwargs
    return ji*(n + p - tn)

def ji_tn(*, ji, p, n, tp, **kwargs):
    """
    Solves tn from the score ji

    Args:
        ji (float|Interval|IntervalUnion): the value or interval for the score ji
        p (int): the p count
        n (int): the n count
        tp (int): the tp count
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tn
    """
    _ = kwargs
    if is_zero(ji):
        return None
    return n + p - tp/ji

def bacc_tp(*, n, bacc, tn, p, **kwargs):
    """
    Solves tp from the score bacc

    Args:
        n (int): the n count
        bacc (float|Interval|IntervalUnion): the value or interval for the score bacc
        tn (int): the tn count
        p (int): the p count
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tp
    """
    _ = kwargs
    if is_zero(n):
        return None
    return p*(2*bacc*n - tn)/n

def bacc_tn(*, n, bacc, p, tp, **kwargs):
    """
    Solves tn from the score bacc

    Args:
        n (int): the n count
        bacc (float|Interval|IntervalUnion): the value or interval for the score bacc
        p (int): the p count
        tp (int): the tp count
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tn
    """
    _ = kwargs
    if is_zero(p):
        return None
    return n*(2*bacc*p - tp)/p

def kappa_tp(*, kappa, n, tn, p, **kwargs):
    """
    Solves tp from the score kappa

    Args:
        kappa (float|Interval|IntervalUnion): the value or interval for the score kappa
        n (int): the n count
        tn (int): the tn count
        p (int): the p count
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tp
    """
    _ = kwargs
    if is_zero(-kappa*n + kappa*p + 2*n):
        return None
    return (kappa*n**2 - kappa*n*tn + kappa*p**2 + kappa*p*tn + 2*n*p - 2*p*tn)/(-kappa*n + kappa*p + 2*n)

def kappa_tn(*, kappa, n, p, tp, **kwargs):
    """
    Solves tn from the score kappa

    Args:
        kappa (float|Interval|IntervalUnion): the value or interval for the score kappa
        n (int): the n count
        p (int): the p count
        tp (int): the tp count
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tn
    """
    _ = kwargs
    if is_zero(kappa*n - kappa*p + 2*p):
        return None
    return (kappa*n**2 + kappa*n*tp + kappa*p**2 - kappa*p*tp + 2*n*p - 2*n*tp)/(kappa*n - kappa*p + 2*p)

def mcc_tp_0(*, mcc, n, tn, p, **kwargs):
    """
    Solves tp from the score mcc

    Args:
        mcc (float|Interval|IntervalUnion): the value or interval for the score mcc
        n (int): the n count
        tn (int): the tn count
        p (int): the p count
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tp
    """
    _ = kwargs
    if is_zero(sqrt(n)) or is_zero(mcc**2*p + n):
        return None
    return (-mcc*sqrt(p)*(n + p)*sqrt(mcc**2*n*p + 4*n*tn - 4*tn**2) + sqrt(n)*p*(-mcc**2*n + mcc**2*p + 2*mcc**2*tn + 2*n - 2*tn))/(2*sqrt(n)*(mcc**2*p + n))

def mcc_tp_1(*, mcc, n, tn, p, **kwargs):
    """
    Solves tp from the score mcc

    Args:
        mcc (float|Interval|IntervalUnion): the value or interval for the score mcc
        n (int): the n count
        tn (int): the tn count
        p (int): the p count
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tp
    """
    _ = kwargs
    if is_zero(sqrt(n)) or is_zero(mcc**2*p + n):
        return None
    return (mcc*sqrt(p)*(n + p)*sqrt(mcc**2*n*p + 4*n*tn - 4*tn**2) + sqrt(n)*p*(-mcc**2*n + mcc**2*p + 2*mcc**2*tn + 2*n - 2*tn))/(2*sqrt(n)*(mcc**2*p + n))

def mcc_tp(*, mcc, n, tn, p, **kwargs):
    """
    Solves tp from the score mcc

    Args:
        mcc (float|Interval|IntervalUnion): the value or interval for the score mcc
        n (int): the n count
        tn (int): the tn count
        p (int): the p count
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tp
    """
    _ = kwargs
    return unify_results([mcc_tp_0(mcc=mcc, n=n, tn=tn, p=p),
                          mcc_tp_1(mcc=mcc, n=n, tn=tn, p=p)])
def mcc_tn_0(*, mcc, n, p, tp, **kwargs):
    """
    Solves tn from the score mcc

    Args:
        mcc (float|Interval|IntervalUnion): the value or interval for the score mcc
        n (int): the n count
        p (int): the p count
        tp (int): the tp count
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tn
    """
    _ = kwargs
    if is_zero(sqrt(p)) or is_zero(mcc**2*n + p):
        return None
    return (-mcc*sqrt(n)*(n + p)*sqrt(mcc**2*n*p + 4*p*tp - 4*tp**2) + n*sqrt(p)*(mcc**2*n - mcc**2*p + 2*mcc**2*tp + 2*p - 2*tp))/(2*sqrt(p)*(mcc**2*n + p))

def mcc_tn_1(*, mcc, n, p, tp, **kwargs):
    """
    Solves tn from the score mcc

    Args:
        mcc (float|Interval|IntervalUnion): the value or interval for the score mcc
        n (int): the n count
        p (int): the p count
        tp (int): the tp count
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tn
    """
    _ = kwargs
    if is_zero(sqrt(p)) or is_zero(mcc**2*n + p):
        return None
    return (mcc*sqrt(n)*(n + p)*sqrt(mcc**2*n*p + 4*p*tp - 4*tp**2) + n*sqrt(p)*(mcc**2*n - mcc**2*p + 2*mcc**2*tp + 2*p - 2*tp))/(2*sqrt(p)*(mcc**2*n + p))

def mcc_tn(*, mcc, n, p, tp, **kwargs):
    """
    Solves tn from the score mcc

    Args:
        mcc (float|Interval|IntervalUnion): the value or interval for the score mcc
        n (int): the n count
        p (int): the p count
        tp (int): the tp count
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tn
    """
    _ = kwargs
    return unify_results([mcc_tn_0(mcc=mcc, n=n, p=p, tp=tp),
                          mcc_tn_1(mcc=mcc, n=n, p=p, tp=tp)])

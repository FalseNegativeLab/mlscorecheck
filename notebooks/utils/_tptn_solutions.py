"""
This module contains the tp and tn solutions.
This is a generated file, do not edit.
"""

# pylint: disable=line-too-long
# pylint: disable=too-many-lines

from ._interval import sqrt
from ._utils import is_zero, unify_results

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
"p4_tp_0",
"p4_tp_1",
"p4_tp",
"p4_tn_0",
"p4_tn_1",
"p4_tn",
"mcc_tp_0",
"mcc_tp_1",
"mcc_tp",
"mcc_tn_0",
"mcc_tn_1",
"mcc_tn"]

def acc_tp(*, acc, p, tn, n, **kwargs):
    """
    Solves tp from the score acc

    Args:
        acc (float|Interval|IntervalUnion): the value or interval for the score acc
        p (int): the p count
        tn (int): the tn count
        n (int): the n count
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tp
    """
    _ = kwargs
    return acc*n + acc*p - tn

def acc_tn(*, acc, p, tp, n, **kwargs):
    """
    Solves tn from the score acc

    Args:
        acc (float|Interval|IntervalUnion): the value or interval for the score acc
        p (int): the p count
        tp (int): the tp count
        n (int): the n count
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tn
    """
    _ = kwargs
    return acc*n + acc*p - tp

def sens_tp(*, p, sens, **kwargs):
    """
    Solves tp from the score sens

    Args:
        p (int): the p count
        sens (float|Interval|IntervalUnion): the value or interval for the score sens
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

def ppv_tp(*, tn, ppv, n, **kwargs):
    """
    Solves tp from the score ppv

    Args:
        tn (int): the tn count
        ppv (float|Interval|IntervalUnion): the value or interval for the score ppv
        n (int): the n count
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tp
    """
    _ = kwargs
    if is_zero(ppv - 1):
        return None
    return ppv*(-n + tn)/(ppv - 1)

def ppv_tn(*, tp, ppv, n, **kwargs):
    """
    Solves tn from the score ppv

    Args:
        tp (int): the tp count
        ppv (float|Interval|IntervalUnion): the value or interval for the score ppv
        n (int): the n count
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tn
    """
    _ = kwargs
    if is_zero(ppv):
        return None
    return n + tp - tp/ppv

def npv_tp(*, p, tn, npv, **kwargs):
    """
    Solves tp from the score npv

    Args:
        p (int): the p count
        tn (int): the tn count
        npv (float|Interval|IntervalUnion): the value or interval for the score npv
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tp
    """
    _ = kwargs
    if is_zero(npv):
        return None
    return p + tn - tn/npv

def npv_tn(*, p, tp, npv, **kwargs):
    """
    Solves tn from the score npv

    Args:
        p (int): the p count
        tp (int): the tp count
        npv (float|Interval|IntervalUnion): the value or interval for the score npv
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tn
    """
    _ = kwargs
    if is_zero(npv - 1):
        return None
    return npv*(-p + tp)/(npv - 1)

def fbp_tp(*, p, fbp, tn, beta_positive, n, **kwargs):
    """
    Solves tp from the score fbp

    Args:
        p (int): the p count
        fbp (float|Interval|IntervalUnion): the value or interval for the score fbp
        tn (int): the tn count
        beta_positive (float): the beta_positive parameter of the score
        n (int): the n count
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tp
    """
    _ = kwargs
    if is_zero(beta_positive**2 - fbp + 1):
        return None
    return fbp*(beta_positive**2*p + n - tn)/(beta_positive**2 - fbp + 1)

def fbp_tn(*, p, fbp, tp, beta_positive, n, **kwargs):
    """
    Solves tn from the score fbp

    Args:
        p (int): the p count
        fbp (float|Interval|IntervalUnion): the value or interval for the score fbp
        tp (int): the tp count
        beta_positive (float): the beta_positive parameter of the score
        n (int): the n count
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tn
    """
    _ = kwargs
    if is_zero(fbp):
        return None
    return (-beta_positive**2*tp + fbp*(beta_positive**2*p + n + tp) - tp)/fbp

def f1p_tp(*, p, tn, f1p, n, **kwargs):
    """
    Solves tp from the score f1p

    Args:
        p (int): the p count
        tn (int): the tn count
        f1p (float|Interval|IntervalUnion): the value or interval for the score f1p
        n (int): the n count
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tp
    """
    _ = kwargs
    if is_zero(f1p - 2):
        return None
    return f1p*(-n - p + tn)/(f1p - 2)

def f1p_tn(*, p, tp, f1p, n, **kwargs):
    """
    Solves tn from the score f1p

    Args:
        p (int): the p count
        tp (int): the tp count
        f1p (float|Interval|IntervalUnion): the value or interval for the score f1p
        n (int): the n count
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tn
    """
    _ = kwargs
    if is_zero(f1p):
        return None
    return n + p + tp - 2*tp/f1p

def fbn_tp(*, p, beta_negative, tn, fbn, n, **kwargs):
    """
    Solves tp from the score fbn

    Args:
        p (int): the p count
        beta_negative (float): the beta_negative parameter of the score
        tn (int): the tn count
        fbn (float|Interval|IntervalUnion): the value or interval for the score fbn
        n (int): the n count
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tp
    """
    _ = kwargs
    if is_zero(fbn):
        return None
    return (-beta_negative**2*tn + fbn*(beta_negative**2*n + p + tn) - tn)/fbn

def fbn_tn(*, p, beta_negative, tp, fbn, n, **kwargs):
    """
    Solves tn from the score fbn

    Args:
        p (int): the p count
        beta_negative (float): the beta_negative parameter of the score
        tp (int): the tp count
        fbn (float|Interval|IntervalUnion): the value or interval for the score fbn
        n (int): the n count
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tn
    """
    _ = kwargs
    if is_zero(beta_negative**2 - fbn + 1):
        return None
    return fbn*(beta_negative**2*n + p - tp)/(beta_negative**2 - fbn + 1)

def f1n_tp(*, p, f1n, tn, n, **kwargs):
    """
    Solves tp from the score f1n

    Args:
        p (int): the p count
        f1n (float|Interval|IntervalUnion): the value or interval for the score f1n
        tn (int): the tn count
        n (int): the n count
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tp
    """
    _ = kwargs
    if is_zero(f1n):
        return None
    return n + p + tn - 2*tn/f1n

def f1n_tn(*, p, tp, n, f1n, **kwargs):
    """
    Solves tn from the score f1n

    Args:
        p (int): the p count
        tp (int): the tp count
        n (int): the n count
        f1n (float|Interval|IntervalUnion): the value or interval for the score f1n
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tn
    """
    _ = kwargs
    if is_zero(f1n - 2):
        return None
    return f1n*(-n - p + tp)/(f1n - 2)

def gm_tp(*, p, tn, gm, n, **kwargs):
    """
    Solves tp from the score gm

    Args:
        p (int): the p count
        tn (int): the tn count
        gm (float|Interval|IntervalUnion): the value or interval for the score gm
        n (int): the n count
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tp
    """
    _ = kwargs
    if is_zero(tn):
        return None
    return gm**2*n*p/tn

def gm_tn(*, p, tp, gm, n, **kwargs):
    """
    Solves tn from the score gm

    Args:
        p (int): the p count
        tp (int): the tp count
        gm (float|Interval|IntervalUnion): the value or interval for the score gm
        n (int): the n count
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tn
    """
    _ = kwargs
    if is_zero(tp):
        return None
    return gm**2*n*p/tp

def fm_tp_0(*, p, tn, fm, n, **kwargs):
    """
    Solves tp from the score fm

    Args:
        p (int): the p count
        tn (int): the tn count
        fm (float|Interval|IntervalUnion): the value or interval for the score fm
        n (int): the n count
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tp
    """
    _ = kwargs
    return fm*(fm*p - sqrt(p)*sqrt(fm**2*p + 4*n - 4*tn))/2

def fm_tp_1(*, p, tn, fm, n, **kwargs):
    """
    Solves tp from the score fm

    Args:
        p (int): the p count
        tn (int): the tn count
        fm (float|Interval|IntervalUnion): the value or interval for the score fm
        n (int): the n count
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tp
    """
    _ = kwargs
    return fm*(fm*p + sqrt(p)*sqrt(fm**2*p + 4*n - 4*tn))/2

def fm_tp(*, p, tn, fm, n, **kwargs):
    """
    Solves tp from the score fm

    Args:
        p (int): the p count
        tn (int): the tn count
        fm (float|Interval|IntervalUnion): the value or interval for the score fm
        n (int): the n count
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tp
    """
    _ = kwargs
    return unify_results([fm_tp_0(p=p, tn=tn, fm=fm, n=n),
                          fm_tp_1(p=p, tn=tn, fm=fm, n=n)])
def fm_tn(*, p, fm, tp, n, **kwargs):
    """
    Solves tn from the score fm

    Args:
        p (int): the p count
        fm (float|Interval|IntervalUnion): the value or interval for the score fm
        tp (int): the tp count
        n (int): the n count
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tn
    """
    _ = kwargs
    if is_zero(p) or is_zero(fm**2):
        return None
    return n + tp - tp**2/(fm**2*p)

def upm_tp_0(*, p, upm, tn, n, **kwargs):
    """
    Solves tp from the score upm

    Args:
        p (int): the p count
        upm (float|Interval|IntervalUnion): the value or interval for the score upm
        tn (int): the tn count
        n (int): the n count
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tp
    """
    _ = kwargs
    if is_zero(upm):
        return None
    return n/2 + p/2 + tn - 2*tn/upm - sqrt(n**2*upm**2 + 2*n*p*upm**2 + 8*n*tn*upm**2 - 8*n*tn*upm + p**2*upm**2 + 8*p*tn*upm**2 - 8*p*tn*upm - 16*tn**2*upm + 16*tn**2)/(2*upm)

def upm_tp_1(*, p, upm, tn, n, **kwargs):
    """
    Solves tp from the score upm

    Args:
        p (int): the p count
        upm (float|Interval|IntervalUnion): the value or interval for the score upm
        tn (int): the tn count
        n (int): the n count
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tp
    """
    _ = kwargs
    if is_zero(upm):
        return None
    return n/2 + p/2 + tn - 2*tn/upm + sqrt(n**2*upm**2 + 2*n*p*upm**2 + 8*n*tn*upm**2 - 8*n*tn*upm + p**2*upm**2 + 8*p*tn*upm**2 - 8*p*tn*upm - 16*tn**2*upm + 16*tn**2)/(2*upm)

def upm_tp(*, p, upm, tn, n, **kwargs):
    """
    Solves tp from the score upm

    Args:
        p (int): the p count
        upm (float|Interval|IntervalUnion): the value or interval for the score upm
        tn (int): the tn count
        n (int): the n count
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tp
    """
    _ = kwargs
    return unify_results([upm_tp_0(p=p, upm=upm, tn=tn, n=n),
                          upm_tp_1(p=p, upm=upm, tn=tn, n=n)])
def upm_tn_0(*, p, upm, tp, n, **kwargs):
    """
    Solves tn from the score upm

    Args:
        p (int): the p count
        upm (float|Interval|IntervalUnion): the value or interval for the score upm
        tp (int): the tp count
        n (int): the n count
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tn
    """
    _ = kwargs
    if is_zero(upm):
        return None
    return n/2 + p/2 + tp - 2*tp/upm - sqrt(n**2*upm**2 + 2*n*p*upm**2 + 8*n*tp*upm**2 - 8*n*tp*upm + p**2*upm**2 + 8*p*tp*upm**2 - 8*p*tp*upm - 16*tp**2*upm + 16*tp**2)/(2*upm)

def upm_tn_1(*, p, upm, tp, n, **kwargs):
    """
    Solves tn from the score upm

    Args:
        p (int): the p count
        upm (float|Interval|IntervalUnion): the value or interval for the score upm
        tp (int): the tp count
        n (int): the n count
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tn
    """
    _ = kwargs
    if is_zero(upm):
        return None
    return n/2 + p/2 + tp - 2*tp/upm + sqrt(n**2*upm**2 + 2*n*p*upm**2 + 8*n*tp*upm**2 - 8*n*tp*upm + p**2*upm**2 + 8*p*tp*upm**2 - 8*p*tp*upm - 16*tp**2*upm + 16*tp**2)/(2*upm)

def upm_tn(*, p, upm, tp, n, **kwargs):
    """
    Solves tn from the score upm

    Args:
        p (int): the p count
        upm (float|Interval|IntervalUnion): the value or interval for the score upm
        tp (int): the tp count
        n (int): the n count
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tn
    """
    _ = kwargs
    return unify_results([upm_tn_0(p=p, upm=upm, tp=tp, n=n),
                          upm_tn_1(p=p, upm=upm, tp=tp, n=n)])
def mk_tp_0(*, p, mk, tn, n, **kwargs):
    """
    Solves tp from the score mk

    Args:
        p (int): the p count
        mk (float|Interval|IntervalUnion): the value or interval for the score mk
        tn (int): the tn count
        n (int): the n count
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tp
    """
    _ = kwargs
    if is_zero(mk):
        return None
    return (-mk*n + mk*p + 2*mk*tn - n - sqrt(mk**2*n**2 + 2*mk**2*n*p + mk**2*p**2 + 2*mk*n**2 + 2*mk*n*p - 4*mk*n*tn - 4*mk*p*tn + n**2))/(2*mk)

def mk_tp_1(*, p, mk, tn, n, **kwargs):
    """
    Solves tp from the score mk

    Args:
        p (int): the p count
        mk (float|Interval|IntervalUnion): the value or interval for the score mk
        tn (int): the tn count
        n (int): the n count
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tp
    """
    _ = kwargs
    if is_zero(mk):
        return None
    return (-mk*n + mk*p + 2*mk*tn - n + sqrt(mk**2*n**2 + 2*mk**2*n*p + mk**2*p**2 + 2*mk*n**2 + 2*mk*n*p - 4*mk*n*tn - 4*mk*p*tn + n**2))/(2*mk)

def mk_tp(*, p, mk, tn, n, **kwargs):
    """
    Solves tp from the score mk

    Args:
        p (int): the p count
        mk (float|Interval|IntervalUnion): the value or interval for the score mk
        tn (int): the tn count
        n (int): the n count
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tp
    """
    _ = kwargs
    return unify_results([mk_tp_0(p=p, mk=mk, tn=tn, n=n),
                          mk_tp_1(p=p, mk=mk, tn=tn, n=n)])
def mk_tn_0(*, p, mk, tp, n, **kwargs):
    """
    Solves tn from the score mk

    Args:
        p (int): the p count
        mk (float|Interval|IntervalUnion): the value or interval for the score mk
        tp (int): the tp count
        n (int): the n count
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tn
    """
    _ = kwargs
    if is_zero(mk):
        return None
    return (mk*n - mk*p + 2*mk*tp - p - sqrt(mk**2*n**2 + 2*mk**2*n*p + mk**2*p**2 + 2*mk*n*p - 4*mk*n*tp + 2*mk*p**2 - 4*mk*p*tp + p**2))/(2*mk)

def mk_tn_1(*, p, mk, tp, n, **kwargs):
    """
    Solves tn from the score mk

    Args:
        p (int): the p count
        mk (float|Interval|IntervalUnion): the value or interval for the score mk
        tp (int): the tp count
        n (int): the n count
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tn
    """
    _ = kwargs
    if is_zero(mk):
        return None
    return (mk*n - mk*p + 2*mk*tp - p + sqrt(mk**2*n**2 + 2*mk**2*n*p + mk**2*p**2 + 2*mk*n*p - 4*mk*n*tp + 2*mk*p**2 - 4*mk*p*tp + p**2))/(2*mk)

def mk_tn(*, p, mk, tp, n, **kwargs):
    """
    Solves tn from the score mk

    Args:
        p (int): the p count
        mk (float|Interval|IntervalUnion): the value or interval for the score mk
        tp (int): the tp count
        n (int): the n count
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tn
    """
    _ = kwargs
    return unify_results([mk_tn_0(p=p, mk=mk, tp=tp, n=n),
                          mk_tn_1(p=p, mk=mk, tp=tp, n=n)])
def lrp_tp(*, p, tn, lrp, n, **kwargs):
    """
    Solves tp from the score lrp

    Args:
        p (int): the p count
        tn (int): the tn count
        lrp (float|Interval|IntervalUnion): the value or interval for the score lrp
        n (int): the n count
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tp
    """
    _ = kwargs
    if is_zero(n):
        return None
    return lrp*p*(n - tn)/n

def lrp_tn(*, p, tp, lrp, n, **kwargs):
    """
    Solves tn from the score lrp

    Args:
        p (int): the p count
        tp (int): the tp count
        lrp (float|Interval|IntervalUnion): the value or interval for the score lrp
        n (int): the n count
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tn
    """
    _ = kwargs
    if is_zero(p) or is_zero(lrp):
        return None
    return n - n*tp/(lrp*p)

def lrn_tp(*, p, lrn, tn, n, **kwargs):
    """
    Solves tp from the score lrn

    Args:
        p (int): the p count
        lrn (float|Interval|IntervalUnion): the value or interval for the score lrn
        tn (int): the tn count
        n (int): the n count
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tp
    """
    _ = kwargs
    if is_zero(n):
        return None
    return p*(-lrn*tn + n)/n

def lrn_tn(*, p, lrn, tp, n, **kwargs):
    """
    Solves tn from the score lrn

    Args:
        p (int): the p count
        lrn (float|Interval|IntervalUnion): the value or interval for the score lrn
        tp (int): the tp count
        n (int): the n count
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tn
    """
    _ = kwargs
    if is_zero(p) or is_zero(lrn):
        return None
    return n*(p - tp)/(lrn*p)

def bm_tp(*, p, tn, bm, n, **kwargs):
    """
    Solves tp from the score bm

    Args:
        p (int): the p count
        tn (int): the tn count
        bm (float|Interval|IntervalUnion): the value or interval for the score bm
        n (int): the n count
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tp
    """
    _ = kwargs
    if is_zero(n):
        return None
    return p*(n*(bm + 1) - tn)/n

def bm_tn(*, p, tp, bm, n, **kwargs):
    """
    Solves tn from the score bm

    Args:
        p (int): the p count
        tp (int): the tp count
        bm (float|Interval|IntervalUnion): the value or interval for the score bm
        n (int): the n count
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tn
    """
    _ = kwargs
    if is_zero(p):
        return None
    return n*(p*(bm + 1) - tp)/p

def pt_tp_0(*, p, tn, n, **kwargs):
    """
    Solves tp from the score pt

    Args:
        p (int): the p count
        tn (int): the tn count
        n (int): the n count
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tp
    """
    _ = kwargs
    if is_zero(n):
        return None
    return p*(n - tn)/n

def pt_tp_1(*, p, pt, tn, n, **kwargs):
    """
    Solves tp from the score pt

    Args:
        p (int): the p count
        pt (float|Interval|IntervalUnion): the value or interval for the score pt
        tn (int): the tn count
        n (int): the n count
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tp
    """
    _ = kwargs
    if is_zero(pt**2) or is_zero(n):
        return None
    return p*(n*pt**2 - 2*n*pt + n - pt**2*tn + 2*pt*tn - tn)/(n*pt**2)

def pt_tp(*, p, pt, tn, n, **kwargs):
    """
    Solves tp from the score pt

    Args:
        p (int): the p count
        pt (float|Interval|IntervalUnion): the value or interval for the score pt
        tn (int): the tn count
        n (int): the n count
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tp
    """
    _ = kwargs
    return unify_results([pt_tp_0(p=p, pt=pt, tn=tn, n=n),
                          pt_tp_1(p=p, pt=pt, tn=tn, n=n)])
def pt_tn_0(*, p, tp, n, **kwargs):
    """
    Solves tn from the score pt

    Args:
        p (int): the p count
        tp (int): the tp count
        n (int): the n count
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tn
    """
    _ = kwargs
    if is_zero(p):
        return None
    return n*(p - tp)/p

def pt_tn_1(*, p, pt, tp, n, **kwargs):
    """
    Solves tn from the score pt

    Args:
        p (int): the p count
        pt (float|Interval|IntervalUnion): the value or interval for the score pt
        tp (int): the tp count
        n (int): the n count
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tn
    """
    _ = kwargs
    if is_zero(p) or is_zero(pt**2 - 2*pt + 1):
        return None
    return n*(p*pt**2 - 2*p*pt + p - pt**2*tp)/(p*(pt**2 - 2*pt + 1))

def pt_tn(*, p, pt, tp, n, **kwargs):
    """
    Solves tn from the score pt

    Args:
        p (int): the p count
        pt (float|Interval|IntervalUnion): the value or interval for the score pt
        tp (int): the tp count
        n (int): the n count
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tn
    """
    _ = kwargs
    return unify_results([pt_tn_0(p=p, pt=pt, tp=tp, n=n),
                          pt_tn_1(p=p, pt=pt, tp=tp, n=n)])
def dor_tp(*, p, dor, tn, n, **kwargs):
    """
    Solves tp from the score dor

    Args:
        p (int): the p count
        dor (float|Interval|IntervalUnion): the value or interval for the score dor
        tn (int): the tn count
        n (int): the n count
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tp
    """
    _ = kwargs
    if is_zero(dor*n - dor*tn + tn):
        return None
    return dor*p*(n - tn)/(dor*n - dor*tn + tn)

def dor_tn(*, p, dor, tp, n, **kwargs):
    """
    Solves tn from the score dor

    Args:
        p (int): the p count
        dor (float|Interval|IntervalUnion): the value or interval for the score dor
        tp (int): the tp count
        n (int): the n count
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tn
    """
    _ = kwargs
    if is_zero(dor*p - dor*tp + tp):
        return None
    return dor*n*(p - tp)/(dor*p - dor*tp + tp)

def ji_tp(*, p, tn, n, ji, **kwargs):
    """
    Solves tp from the score ji

    Args:
        p (int): the p count
        tn (int): the tn count
        n (int): the n count
        ji (float|Interval|IntervalUnion): the value or interval for the score ji
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tp
    """
    _ = kwargs
    return ji*(n + p - tn)

def ji_tn(*, p, tp, ji, n, **kwargs):
    """
    Solves tn from the score ji

    Args:
        p (int): the p count
        tp (int): the tp count
        ji (float|Interval|IntervalUnion): the value or interval for the score ji
        n (int): the n count
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tn
    """
    _ = kwargs
    if is_zero(ji):
        return None
    return n + p - tp/ji

def bacc_tp(*, p, bacc, tn, n, **kwargs):
    """
    Solves tp from the score bacc

    Args:
        p (int): the p count
        bacc (float|Interval|IntervalUnion): the value or interval for the score bacc
        tn (int): the tn count
        n (int): the n count
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tp
    """
    _ = kwargs
    if is_zero(n):
        return None
    return p*(2*bacc*n - tn)/n

def bacc_tn(*, p, bacc, tp, n, **kwargs):
    """
    Solves tn from the score bacc

    Args:
        p (int): the p count
        bacc (float|Interval|IntervalUnion): the value or interval for the score bacc
        tp (int): the tp count
        n (int): the n count
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tn
    """
    _ = kwargs
    if is_zero(p):
        return None
    return n*(2*bacc*p - tp)/p

def kappa_tp(*, p, n, tn, kappa, **kwargs):
    """
    Solves tp from the score kappa

    Args:
        p (int): the p count
        n (int): the n count
        tn (int): the tn count
        kappa (float|Interval|IntervalUnion): the value or interval for the score kappa
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tp
    """
    _ = kwargs
    if is_zero(-kappa*n + kappa*p + 2*n):
        return None
    return (kappa*n**2 - kappa*n*tn + kappa*p**2 + kappa*p*tn + 2*n*p - 2*p*tn)/(-kappa*n + kappa*p + 2*n)

def kappa_tn(*, p, n, tp, kappa, **kwargs):
    """
    Solves tn from the score kappa

    Args:
        p (int): the p count
        n (int): the n count
        tp (int): the tp count
        kappa (float|Interval|IntervalUnion): the value or interval for the score kappa
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tn
    """
    _ = kwargs
    if is_zero(kappa*n - kappa*p + 2*p):
        return None
    return (kappa*n**2 + kappa*n*tp + kappa*p**2 - kappa*p*tp + 2*n*p - 2*n*tp)/(kappa*n - kappa*p + 2*p)

def p4_tp_0(*, p, p4, tn, n, **kwargs):
    """
    Solves tp from the score p4

    Args:
        p (int): the p count
        p4 (float|Interval|IntervalUnion): the value or interval for the score p4
        tn (int): the tn count
        n (int): the n count
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tp
    """
    _ = kwargs
    if is_zero(p4):
        return None
    return n/2 + p/2 + tn - 2*tn/p4 - sqrt(n**2*p4**2 + 2*n*p*p4**2 + 8*n*p4**2*tn - 8*n*p4*tn + p**2*p4**2 + 8*p*p4**2*tn - 8*p*p4*tn - 16*p4*tn**2 + 16*tn**2)/(2*p4)

def p4_tp_1(*, p, p4, tn, n, **kwargs):
    """
    Solves tp from the score p4

    Args:
        p (int): the p count
        p4 (float|Interval|IntervalUnion): the value or interval for the score p4
        tn (int): the tn count
        n (int): the n count
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tp
    """
    _ = kwargs
    if is_zero(p4):
        return None
    return n/2 + p/2 + tn - 2*tn/p4 + sqrt(n**2*p4**2 + 2*n*p*p4**2 + 8*n*p4**2*tn - 8*n*p4*tn + p**2*p4**2 + 8*p*p4**2*tn - 8*p*p4*tn - 16*p4*tn**2 + 16*tn**2)/(2*p4)

def p4_tp(*, p, p4, tn, n, **kwargs):
    """
    Solves tp from the score p4

    Args:
        p (int): the p count
        p4 (float|Interval|IntervalUnion): the value or interval for the score p4
        tn (int): the tn count
        n (int): the n count
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tp
    """
    _ = kwargs
    return unify_results([p4_tp_0(p=p, p4=p4, tn=tn, n=n),
                          p4_tp_1(p=p, p4=p4, tn=tn, n=n)])
def p4_tn_0(*, p, p4, tp, n, **kwargs):
    """
    Solves tn from the score p4

    Args:
        p (int): the p count
        p4 (float|Interval|IntervalUnion): the value or interval for the score p4
        tp (int): the tp count
        n (int): the n count
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tn
    """
    _ = kwargs
    if is_zero(p4):
        return None
    return n/2 + p/2 + tp - 2*tp/p4 - sqrt(n**2*p4**2 + 2*n*p*p4**2 + 8*n*p4**2*tp - 8*n*p4*tp + p**2*p4**2 + 8*p*p4**2*tp - 8*p*p4*tp - 16*p4*tp**2 + 16*tp**2)/(2*p4)

def p4_tn_1(*, p, p4, tp, n, **kwargs):
    """
    Solves tn from the score p4

    Args:
        p (int): the p count
        p4 (float|Interval|IntervalUnion): the value or interval for the score p4
        tp (int): the tp count
        n (int): the n count
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tn
    """
    _ = kwargs
    if is_zero(p4):
        return None
    return n/2 + p/2 + tp - 2*tp/p4 + sqrt(n**2*p4**2 + 2*n*p*p4**2 + 8*n*p4**2*tp - 8*n*p4*tp + p**2*p4**2 + 8*p*p4**2*tp - 8*p*p4*tp - 16*p4*tp**2 + 16*tp**2)/(2*p4)

def p4_tn(*, p, p4, tp, n, **kwargs):
    """
    Solves tn from the score p4

    Args:
        p (int): the p count
        p4 (float|Interval|IntervalUnion): the value or interval for the score p4
        tp (int): the tp count
        n (int): the n count
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tn
    """
    _ = kwargs
    return unify_results([p4_tn_0(p=p, p4=p4, tp=tp, n=n),
                          p4_tn_1(p=p, p4=p4, tp=tp, n=n)])
def mcc_tp_0(*, p, mcc, tn, n, **kwargs):
    """
    Solves tp from the score mcc

    Args:
        p (int): the p count
        mcc (float|Interval|IntervalUnion): the value or interval for the score mcc
        tn (int): the tn count
        n (int): the n count
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tp
    """
    _ = kwargs
    if is_zero(mcc**2*p + n) or is_zero(sqrt(n)):
        return None
    return (-mcc*sqrt(p)*(n + p)*sqrt(mcc**2*n*p + 4*n*tn - 4*tn**2) + sqrt(n)*p*(-mcc**2*n + mcc**2*p + 2*mcc**2*tn + 2*n - 2*tn))/(2*sqrt(n)*(mcc**2*p + n))

def mcc_tp_1(*, p, mcc, tn, n, **kwargs):
    """
    Solves tp from the score mcc

    Args:
        p (int): the p count
        mcc (float|Interval|IntervalUnion): the value or interval for the score mcc
        tn (int): the tn count
        n (int): the n count
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tp
    """
    _ = kwargs
    if is_zero(mcc**2*p + n) or is_zero(sqrt(n)):
        return None
    return (mcc*sqrt(p)*(n + p)*sqrt(mcc**2*n*p + 4*n*tn - 4*tn**2) + sqrt(n)*p*(-mcc**2*n + mcc**2*p + 2*mcc**2*tn + 2*n - 2*tn))/(2*sqrt(n)*(mcc**2*p + n))

def mcc_tp(*, p, mcc, tn, n, **kwargs):
    """
    Solves tp from the score mcc

    Args:
        p (int): the p count
        mcc (float|Interval|IntervalUnion): the value or interval for the score mcc
        tn (int): the tn count
        n (int): the n count
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tp
    """
    _ = kwargs
    return unify_results([mcc_tp_0(p=p, mcc=mcc, tn=tn, n=n),
                          mcc_tp_1(p=p, mcc=mcc, tn=tn, n=n)])
def mcc_tn_0(*, p, mcc, tp, n, **kwargs):
    """
    Solves tn from the score mcc

    Args:
        p (int): the p count
        mcc (float|Interval|IntervalUnion): the value or interval for the score mcc
        tp (int): the tp count
        n (int): the n count
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tn
    """
    _ = kwargs
    if is_zero(sqrt(p)) or is_zero(mcc**2*n + p):
        return None
    return (-mcc*sqrt(n)*(n + p)*sqrt(mcc**2*n*p + 4*p*tp - 4*tp**2) + n*sqrt(p)*(mcc**2*n - mcc**2*p + 2*mcc**2*tp + 2*p - 2*tp))/(2*sqrt(p)*(mcc**2*n + p))

def mcc_tn_1(*, p, mcc, tp, n, **kwargs):
    """
    Solves tn from the score mcc

    Args:
        p (int): the p count
        mcc (float|Interval|IntervalUnion): the value or interval for the score mcc
        tp (int): the tp count
        n (int): the n count
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tn
    """
    _ = kwargs
    if is_zero(sqrt(p)) or is_zero(mcc**2*n + p):
        return None
    return (mcc*sqrt(n)*(n + p)*sqrt(mcc**2*n*p + 4*p*tp - 4*tp**2) + n*sqrt(p)*(mcc**2*n - mcc**2*p + 2*mcc**2*tp + 2*p - 2*tp))/(2*sqrt(p)*(mcc**2*n + p))

def mcc_tn(*, p, mcc, tp, n, **kwargs):
    """
    Solves tn from the score mcc

    Args:
        p (int): the p count
        mcc (float|Interval|IntervalUnion): the value or interval for the score mcc
        tp (int): the tp count
        n (int): the n count
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tn
    """
    _ = kwargs
    return unify_results([mcc_tn_0(p=p, mcc=mcc, tp=tp, n=n),
                          mcc_tn_1(p=p, mcc=mcc, tp=tp, n=n)])

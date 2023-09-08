"""
This module contains the tp and tn solutions.
This is a generated file, do not edit.
"""

from ._helper import is_less_than_zero, is_zero, unify_results
from ._interval import sqrt

__all__ = [
"mcc_tp_0",
"mcc_tp_1",
"mcc_tp",
"mcc_tn_0",
"mcc_tn_1",
"mcc_tn",
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
"fbm_tp",
"fbm_tn",
"f1m_tp",
"f1m_tn",
"upm_tp_0",
"upm_tp_1",
"upm_tp",
"upm_tn_0",
"upm_tn_1",
"upm_tn",
"gm_tp",
"gm_tn",
"fm_tp_0",
"fm_tp_1",
"fm_tp",
"fm_tn",
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
"p4_tn"]

def mcc_tp_0(*, p, tn, n, mcc, **kwargs):
    """
    Solves tp from the score mcc

    Args:
        p (int): the p count
        tn (int): the tn count
        n (int): the n count
        mcc (float|Interval|IntervalUnion): the value or interval for the score mcc
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tp
    """
    if is_less_than_zero(n) or is_less_than_zero(p) or is_less_than_zero(mcc**2*n*p + 4*n*tn - 4*tn**2):
        return None
    if is_zero(sqrt(n)) or is_zero(mcc**2*p + n):
        return None
    return (-mcc*sqrt(p)*(n + p)*sqrt(mcc**2*n*p + 4*n*tn - 4*tn**2) + sqrt(n)*p*(-mcc**2*n + mcc**2*p + 2*mcc**2*tn + 2*n - 2*tn))/(2*sqrt(n)*(mcc**2*p + n))

def mcc_tp_1(*, p, tn, n, mcc, **kwargs):
    """
    Solves tp from the score mcc

    Args:
        p (int): the p count
        tn (int): the tn count
        n (int): the n count
        mcc (float|Interval|IntervalUnion): the value or interval for the score mcc
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tp
    """
    if is_less_than_zero(n) or is_less_than_zero(p) or is_less_than_zero(mcc**2*n*p + 4*n*tn - 4*tn**2):
        return None
    if is_zero(sqrt(n)) or is_zero(mcc**2*p + n):
        return None
    return (mcc*sqrt(p)*(n + p)*sqrt(mcc**2*n*p + 4*n*tn - 4*tn**2) + sqrt(n)*p*(-mcc**2*n + mcc**2*p + 2*mcc**2*tn + 2*n - 2*tn))/(2*sqrt(n)*(mcc**2*p + n))

def mcc_tp(*, p, tn, n, mcc, **kwargs):
    """
    Solves tp from the score mcc

    Args:
        p (int): the p count
        tn (int): the tn count
        n (int): the n count
        mcc (float|Interval|IntervalUnion): the value or interval for the score mcc
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tp
    """
    return unify_results([mcc_tp_0(p=p, tn=tn, n=n, mcc=mcc),
                          mcc_tp_1(p=p, tn=tn, n=n, mcc=mcc)])
def mcc_tn_0(*, p, n, tp, mcc, **kwargs):
    """
    Solves tn from the score mcc

    Args:
        p (int): the p count
        n (int): the n count
        tp (int): the tp count
        mcc (float|Interval|IntervalUnion): the value or interval for the score mcc
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tn
    """
    if is_less_than_zero(n) or is_less_than_zero(mcc**2*n*p + 4*p*tp - 4*tp**2) or is_less_than_zero(p):
        return None
    if is_zero(mcc**2*n + p) or is_zero(sqrt(p)):
        return None
    return (-mcc*sqrt(n)*(n + p)*sqrt(mcc**2*n*p + 4*p*tp - 4*tp**2) + n*sqrt(p)*(mcc**2*n - mcc**2*p + 2*mcc**2*tp + 2*p - 2*tp))/(2*sqrt(p)*(mcc**2*n + p))

def mcc_tn_1(*, p, n, tp, mcc, **kwargs):
    """
    Solves tn from the score mcc

    Args:
        p (int): the p count
        n (int): the n count
        tp (int): the tp count
        mcc (float|Interval|IntervalUnion): the value or interval for the score mcc
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tn
    """
    if is_less_than_zero(n) or is_less_than_zero(mcc**2*n*p + 4*p*tp - 4*tp**2) or is_less_than_zero(p):
        return None
    if is_zero(mcc**2*n + p) or is_zero(sqrt(p)):
        return None
    return (mcc*sqrt(n)*(n + p)*sqrt(mcc**2*n*p + 4*p*tp - 4*tp**2) + n*sqrt(p)*(mcc**2*n - mcc**2*p + 2*mcc**2*tp + 2*p - 2*tp))/(2*sqrt(p)*(mcc**2*n + p))

def mcc_tn(*, p, n, tp, mcc, **kwargs):
    """
    Solves tn from the score mcc

    Args:
        p (int): the p count
        n (int): the n count
        tp (int): the tp count
        mcc (float|Interval|IntervalUnion): the value or interval for the score mcc
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tn
    """
    return unify_results([mcc_tn_0(p=p, n=n, tp=tp, mcc=mcc),
                          mcc_tn_1(p=p, n=n, tp=tp, mcc=mcc)])
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
    return acc*n + acc*p - tn

def acc_tn(*, acc, p, n, tp, **kwargs):
    """
    Solves tn from the score acc

    Args:
        acc (float|Interval|IntervalUnion): the value or interval for the score acc
        p (int): the p count
        n (int): the n count
        tp (int): the tp count
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tn
    """
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
    return p*sens

def spec_tn(*, n, spec, **kwargs):
    """
    Solves tn from the score spec

    Args:
        n (int): the n count
        spec (float|Interval|IntervalUnion): the value or interval for the score spec
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tn
    """
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
    if is_zero(ppv - 1):
        return None
    return ppv*(-n + tn)/(ppv - 1)

def ppv_tn(*, n, ppv, tp, **kwargs):
    """
    Solves tn from the score ppv

    Args:
        n (int): the n count
        ppv (float|Interval|IntervalUnion): the value or interval for the score ppv
        tp (int): the tp count
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tn
    """
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
    if is_zero(npv):
        return None
    return p + tn - tn/npv

def npv_tn(*, npv, p, tp, **kwargs):
    """
    Solves tn from the score npv

    Args:
        npv (float|Interval|IntervalUnion): the value or interval for the score npv
        p (int): the p count
        tp (int): the tp count
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tn
    """
    if is_zero(npv - 1):
        return None
    return npv*(-p + tp)/(npv - 1)

def fbp_tp(*, beta_plus, p, fbp, tn, n, **kwargs):
    """
    Solves tp from the score fbp

    Args:
        beta_plus (float): the beta_plus parameter of the score
        p (int): the p count
        fbp (float|Interval|IntervalUnion): the value or interval for the score fbp
        tn (int): the tn count
        n (int): the n count
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tp
    """
    if is_zero(beta_plus**2 - fbp + 1):
        return None
    return fbp*(beta_plus**2*p + n - tn)/(beta_plus**2 - fbp + 1)

def fbp_tn(*, p, beta_plus, fbp, n, tp, **kwargs):
    """
    Solves tn from the score fbp

    Args:
        p (int): the p count
        beta_plus (float): the beta_plus parameter of the score
        fbp (float|Interval|IntervalUnion): the value or interval for the score fbp
        n (int): the n count
        tp (int): the tp count
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tn
    """
    if is_zero(fbp):
        return None
    return (-beta_plus**2*tp + fbp*(beta_plus**2*p + n + tp) - tp)/fbp

def f1p_tp(*, n, p, tn, f1p, **kwargs):
    """
    Solves tp from the score f1p

    Args:
        n (int): the n count
        p (int): the p count
        tn (int): the tn count
        f1p (float|Interval|IntervalUnion): the value or interval for the score f1p
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tp
    """
    if is_zero(f1p - 2):
        return None
    return f1p*(-n - p + tn)/(f1p - 2)

def f1p_tn(*, p, n, f1p, tp, **kwargs):
    """
    Solves tn from the score f1p

    Args:
        p (int): the p count
        n (int): the n count
        f1p (float|Interval|IntervalUnion): the value or interval for the score f1p
        tp (int): the tp count
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tn
    """
    if is_zero(f1p):
        return None
    return n + p + tp - 2*tp/f1p

def fbm_tp(*, beta_minus, p, tn, n, fbm, **kwargs):
    """
    Solves tp from the score fbm

    Args:
        beta_minus (float): the beta_minus parameter of the score
        p (int): the p count
        tn (int): the tn count
        n (int): the n count
        fbm (float|Interval|IntervalUnion): the value or interval for the score fbm
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tp
    """
    if is_zero(fbm):
        return None
    return (-beta_minus**2*tn + fbm*(beta_minus**2*n + p + tn) - tn)/fbm

def fbm_tn(*, beta_minus, p, n, tp, fbm, **kwargs):
    """
    Solves tn from the score fbm

    Args:
        beta_minus (float): the beta_minus parameter of the score
        p (int): the p count
        n (int): the n count
        tp (int): the tp count
        fbm (float|Interval|IntervalUnion): the value or interval for the score fbm
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tn
    """
    if is_zero(beta_minus**2 - fbm + 1):
        return None
    return fbm*(beta_minus**2*n + p - tp)/(beta_minus**2 - fbm + 1)

def f1m_tp(*, p, tn, n, f1m, **kwargs):
    """
    Solves tp from the score f1m

    Args:
        p (int): the p count
        tn (int): the tn count
        n (int): the n count
        f1m (float|Interval|IntervalUnion): the value or interval for the score f1m
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tp
    """
    if is_zero(f1m):
        return None
    return n + p + tn - 2*tn/f1m

def f1m_tn(*, n, p, f1m, tp, **kwargs):
    """
    Solves tn from the score f1m

    Args:
        n (int): the n count
        p (int): the p count
        f1m (float|Interval|IntervalUnion): the value or interval for the score f1m
        tp (int): the tp count
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tn
    """
    if is_zero(f1m - 2):
        return None
    return f1m*(-n - p + tp)/(f1m - 2)

def upm_tp_0(*, p, tn, upm, n, **kwargs):
    """
    Solves tp from the score upm

    Args:
        p (int): the p count
        tn (int): the tn count
        upm (float|Interval|IntervalUnion): the value or interval for the score upm
        n (int): the n count
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tp
    """
    if is_less_than_zero(n**2*upm**2 + 2*n*p*upm**2 + 8*n*tn*upm**2 - 8*n*tn*upm + p**2*upm**2 + 8*p*tn*upm**2 - 8*p*tn*upm - 16*tn**2*upm + 16*tn**2):
        return None
    if is_zero(upm):
        return None
    return n/2 + p/2 + tn - 2*tn/upm - sqrt(n**2*upm**2 + 2*n*p*upm**2 + 8*n*tn*upm**2 - 8*n*tn*upm + p**2*upm**2 + 8*p*tn*upm**2 - 8*p*tn*upm - 16*tn**2*upm + 16*tn**2)/(2*upm)

def upm_tp_1(*, p, tn, upm, n, **kwargs):
    """
    Solves tp from the score upm

    Args:
        p (int): the p count
        tn (int): the tn count
        upm (float|Interval|IntervalUnion): the value or interval for the score upm
        n (int): the n count
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tp
    """
    if is_less_than_zero(n**2*upm**2 + 2*n*p*upm**2 + 8*n*tn*upm**2 - 8*n*tn*upm + p**2*upm**2 + 8*p*tn*upm**2 - 8*p*tn*upm - 16*tn**2*upm + 16*tn**2):
        return None
    if is_zero(upm):
        return None
    return n/2 + p/2 + tn - 2*tn/upm + sqrt(n**2*upm**2 + 2*n*p*upm**2 + 8*n*tn*upm**2 - 8*n*tn*upm + p**2*upm**2 + 8*p*tn*upm**2 - 8*p*tn*upm - 16*tn**2*upm + 16*tn**2)/(2*upm)

def upm_tp(*, p, tn, upm, n, **kwargs):
    """
    Solves tp from the score upm

    Args:
        p (int): the p count
        tn (int): the tn count
        upm (float|Interval|IntervalUnion): the value or interval for the score upm
        n (int): the n count
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tp
    """
    return unify_results([upm_tp_0(p=p, tn=tn, upm=upm, n=n),
                          upm_tp_1(p=p, tn=tn, upm=upm, n=n)])
def upm_tn_0(*, p, upm, n, tp, **kwargs):
    """
    Solves tn from the score upm

    Args:
        p (int): the p count
        upm (float|Interval|IntervalUnion): the value or interval for the score upm
        n (int): the n count
        tp (int): the tp count
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tn
    """
    if is_less_than_zero(n**2*upm**2 + 2*n*p*upm**2 + 8*n*tp*upm**2 - 8*n*tp*upm + p**2*upm**2 + 8*p*tp*upm**2 - 8*p*tp*upm - 16*tp**2*upm + 16*tp**2):
        return None
    if is_zero(upm):
        return None
    return n/2 + p/2 + tp - 2*tp/upm - sqrt(n**2*upm**2 + 2*n*p*upm**2 + 8*n*tp*upm**2 - 8*n*tp*upm + p**2*upm**2 + 8*p*tp*upm**2 - 8*p*tp*upm - 16*tp**2*upm + 16*tp**2)/(2*upm)

def upm_tn_1(*, p, upm, n, tp, **kwargs):
    """
    Solves tn from the score upm

    Args:
        p (int): the p count
        upm (float|Interval|IntervalUnion): the value or interval for the score upm
        n (int): the n count
        tp (int): the tp count
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tn
    """
    if is_less_than_zero(n**2*upm**2 + 2*n*p*upm**2 + 8*n*tp*upm**2 - 8*n*tp*upm + p**2*upm**2 + 8*p*tp*upm**2 - 8*p*tp*upm - 16*tp**2*upm + 16*tp**2):
        return None
    if is_zero(upm):
        return None
    return n/2 + p/2 + tp - 2*tp/upm + sqrt(n**2*upm**2 + 2*n*p*upm**2 + 8*n*tp*upm**2 - 8*n*tp*upm + p**2*upm**2 + 8*p*tp*upm**2 - 8*p*tp*upm - 16*tp**2*upm + 16*tp**2)/(2*upm)

def upm_tn(*, p, upm, n, tp, **kwargs):
    """
    Solves tn from the score upm

    Args:
        p (int): the p count
        upm (float|Interval|IntervalUnion): the value or interval for the score upm
        n (int): the n count
        tp (int): the tp count
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tn
    """
    return unify_results([upm_tn_0(p=p, upm=upm, n=n, tp=tp),
                          upm_tn_1(p=p, upm=upm, n=n, tp=tp)])
def gm_tp(*, n, p, tn, gm, **kwargs):
    """
    Solves tp from the score gm

    Args:
        n (int): the n count
        p (int): the p count
        tn (int): the tn count
        gm (float|Interval|IntervalUnion): the value or interval for the score gm
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tp
    """
    if is_zero(tn):
        return None
    return gm**2*n*p/tn

def gm_tn(*, n, p, gm, tp, **kwargs):
    """
    Solves tn from the score gm

    Args:
        n (int): the n count
        p (int): the p count
        gm (float|Interval|IntervalUnion): the value or interval for the score gm
        tp (int): the tp count
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tn
    """
    if is_zero(tp):
        return None
    return gm**2*n*p/tp

def fm_tp_0(*, p, tn, n, fm, **kwargs):
    """
    Solves tp from the score fm

    Args:
        p (int): the p count
        tn (int): the tn count
        n (int): the n count
        fm (float|Interval|IntervalUnion): the value or interval for the score fm
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tp
    """
    if is_less_than_zero(fm**2*p + 4*n - 4*tn) or is_less_than_zero(p):
        return None
    return fm*(fm*p - sqrt(p)*sqrt(fm**2*p + 4*n - 4*tn))/2

def fm_tp_1(*, p, tn, n, fm, **kwargs):
    """
    Solves tp from the score fm

    Args:
        p (int): the p count
        tn (int): the tn count
        n (int): the n count
        fm (float|Interval|IntervalUnion): the value or interval for the score fm
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tp
    """
    if is_less_than_zero(fm**2*p + 4*n - 4*tn) or is_less_than_zero(p):
        return None
    return fm*(fm*p + sqrt(p)*sqrt(fm**2*p + 4*n - 4*tn))/2

def fm_tp(*, p, tn, n, fm, **kwargs):
    """
    Solves tp from the score fm

    Args:
        p (int): the p count
        tn (int): the tn count
        n (int): the n count
        fm (float|Interval|IntervalUnion): the value or interval for the score fm
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tp
    """
    return unify_results([fm_tp_0(p=p, tn=tn, n=n, fm=fm),
                          fm_tp_1(p=p, tn=tn, n=n, fm=fm)])
def fm_tn(*, p, n, tp, fm, **kwargs):
    """
    Solves tn from the score fm

    Args:
        p (int): the p count
        n (int): the n count
        tp (int): the tp count
        fm (float|Interval|IntervalUnion): the value or interval for the score fm
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tn
    """
    if is_zero(fm**2) or is_zero(p):
        return None
    return n + tp - tp**2/(fm**2*p)

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
    if is_less_than_zero(mk**2*n**2 + 2*mk**2*n*p + mk**2*p**2 + 2*mk*n**2 + 2*mk*n*p - 4*mk*n*tn - 4*mk*p*tn + n**2):
        return None
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
    if is_less_than_zero(mk**2*n**2 + 2*mk**2*n*p + mk**2*p**2 + 2*mk*n**2 + 2*mk*n*p - 4*mk*n*tn - 4*mk*p*tn + n**2):
        return None
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
    return unify_results([mk_tp_0(p=p, mk=mk, tn=tn, n=n),
                          mk_tp_1(p=p, mk=mk, tn=tn, n=n)])
def mk_tn_0(*, p, mk, n, tp, **kwargs):
    """
    Solves tn from the score mk

    Args:
        p (int): the p count
        mk (float|Interval|IntervalUnion): the value or interval for the score mk
        n (int): the n count
        tp (int): the tp count
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tn
    """
    if is_less_than_zero(mk**2*n**2 + 2*mk**2*n*p + mk**2*p**2 + 2*mk*n*p - 4*mk*n*tp + 2*mk*p**2 - 4*mk*p*tp + p**2):
        return None
    if is_zero(mk):
        return None
    return (mk*n - mk*p + 2*mk*tp - p - sqrt(mk**2*n**2 + 2*mk**2*n*p + mk**2*p**2 + 2*mk*n*p - 4*mk*n*tp + 2*mk*p**2 - 4*mk*p*tp + p**2))/(2*mk)

def mk_tn_1(*, p, mk, n, tp, **kwargs):
    """
    Solves tn from the score mk

    Args:
        p (int): the p count
        mk (float|Interval|IntervalUnion): the value or interval for the score mk
        n (int): the n count
        tp (int): the tp count
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tn
    """
    if is_less_than_zero(mk**2*n**2 + 2*mk**2*n*p + mk**2*p**2 + 2*mk*n*p - 4*mk*n*tp + 2*mk*p**2 - 4*mk*p*tp + p**2):
        return None
    if is_zero(mk):
        return None
    return (mk*n - mk*p + 2*mk*tp - p + sqrt(mk**2*n**2 + 2*mk**2*n*p + mk**2*p**2 + 2*mk*n*p - 4*mk*n*tp + 2*mk*p**2 - 4*mk*p*tp + p**2))/(2*mk)

def mk_tn(*, p, mk, n, tp, **kwargs):
    """
    Solves tn from the score mk

    Args:
        p (int): the p count
        mk (float|Interval|IntervalUnion): the value or interval for the score mk
        n (int): the n count
        tp (int): the tp count
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tn
    """
    return unify_results([mk_tn_0(p=p, mk=mk, n=n, tp=tp),
                          mk_tn_1(p=p, mk=mk, n=n, tp=tp)])
def lrp_tp(*, p, lrp, tn, n, **kwargs):
    """
    Solves tp from the score lrp

    Args:
        p (int): the p count
        lrp (float|Interval|IntervalUnion): the value or interval for the score lrp
        tn (int): the tn count
        n (int): the n count
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tp
    """
    if is_zero(n):
        return None
    return lrp*p*(n - tn)/n

def lrp_tn(*, p, lrp, n, tp, **kwargs):
    """
    Solves tn from the score lrp

    Args:
        p (int): the p count
        lrp (float|Interval|IntervalUnion): the value or interval for the score lrp
        n (int): the n count
        tp (int): the tp count
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tn
    """
    if is_zero(p) or is_zero(lrp):
        return None
    return n - n*tp/(lrp*p)

def lrn_tp(*, p, tn, n, lrn, **kwargs):
    """
    Solves tp from the score lrn

    Args:
        p (int): the p count
        tn (int): the tn count
        n (int): the n count
        lrn (float|Interval|IntervalUnion): the value or interval for the score lrn
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tp
    """
    if is_zero(n):
        return None
    return p*(-lrn*tn + n)/n

def lrn_tn(*, p, n, lrn, tp, **kwargs):
    """
    Solves tn from the score lrn

    Args:
        p (int): the p count
        n (int): the n count
        lrn (float|Interval|IntervalUnion): the value or interval for the score lrn
        tp (int): the tp count
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tn
    """
    if is_zero(lrn) or is_zero(p):
        return None
    return n*(p - tp)/(lrn*p)

def bm_tp(*, p, tn, n, bm, **kwargs):
    """
    Solves tp from the score bm

    Args:
        p (int): the p count
        tn (int): the tn count
        n (int): the n count
        bm (float|Interval|IntervalUnion): the value or interval for the score bm
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tp
    """
    if is_zero(n):
        return None
    return p*(n*(bm + 1) - tn)/n

def bm_tn(*, p, n, bm, tp, **kwargs):
    """
    Solves tn from the score bm

    Args:
        p (int): the p count
        n (int): the n count
        bm (float|Interval|IntervalUnion): the value or interval for the score bm
        tp (int): the tp count
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tn
    """
    if is_zero(p):
        return None
    return n*(p*(bm + 1) - tp)/p

def pt_tp_0(*, pt, p, tn, n, **kwargs):
    """
    Solves tp from the score pt

    Args:
        pt (float|Interval|IntervalUnion): the value or interval for the score pt
        p (int): the p count
        tn (int): the tn count
        n (int): the n count
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tp
    """
    if is_less_than_zero((n - tn)*(4*n*pt**2 - 4*n*pt + n + 4*pt**2*tn - 4*pt*tn - tn)):
        return None
    if is_zero(n) or is_zero(pt**2):
        return None
    return p*(2*n*pt**2 - 2*n*pt + n + 2*pt**2*tn - 2*pt*tn - tn - sqrt((n - tn)*(4*n*pt**2 - 4*n*pt + n + 4*pt**2*tn - 4*pt*tn - tn)))/(2*n*pt**2)

def pt_tp_1(*, pt, p, tn, n, **kwargs):
    """
    Solves tp from the score pt

    Args:
        pt (float|Interval|IntervalUnion): the value or interval for the score pt
        p (int): the p count
        tn (int): the tn count
        n (int): the n count
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tp
    """
    if is_less_than_zero((n - tn)*(4*n*pt**2 - 4*n*pt + n + 4*pt**2*tn - 4*pt*tn - tn)):
        return None
    if is_zero(n) or is_zero(pt**2):
        return None
    return p*(2*n*pt**2 - 2*n*pt + n + 2*pt**2*tn - 2*pt*tn - tn + sqrt((n - tn)*(4*n*pt**2 - 4*n*pt + n + 4*pt**2*tn - 4*pt*tn - tn)))/(2*n*pt**2)

def pt_tp(*, pt, p, tn, n, **kwargs):
    """
    Solves tp from the score pt

    Args:
        pt (float|Interval|IntervalUnion): the value or interval for the score pt
        p (int): the p count
        tn (int): the tn count
        n (int): the n count
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tp
    """
    return unify_results([pt_tp_0(pt=pt, p=p, tn=tn, n=n),
                          pt_tp_1(pt=pt, p=p, tn=tn, n=n)])
def pt_tn_0(*, pt, p, n, tp, **kwargs):
    """
    Solves tn from the score pt

    Args:
        pt (float|Interval|IntervalUnion): the value or interval for the score pt
        p (int): the p count
        n (int): the n count
        tp (int): the tp count
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tn
    """
    if is_less_than_zero(8*p*pt**2 - 16*p*pt + 8*p - 4*pt**2*tp + 4*pt*tp + tp) or is_less_than_zero(tp):
        return None
    if is_zero(p) or is_zero(pt**2 - 2*pt + 1):
        return None
    return n*(-2*p*pt**2 + 4*p*pt - 2*p + 2*pt**2*tp - 2*pt*tp - sqrt(tp)*sqrt(8*p*pt**2 - 16*p*pt + 8*p - 4*pt**2*tp + 4*pt*tp + tp) - tp)/(2*p*(pt**2 - 2*pt + 1))

def pt_tn_1(*, pt, p, n, tp, **kwargs):
    """
    Solves tn from the score pt

    Args:
        pt (float|Interval|IntervalUnion): the value or interval for the score pt
        p (int): the p count
        n (int): the n count
        tp (int): the tp count
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tn
    """
    if is_less_than_zero(8*p*pt**2 - 16*p*pt + 8*p - 4*pt**2*tp + 4*pt*tp + tp) or is_less_than_zero(tp):
        return None
    if is_zero(p) or is_zero(pt**2 - 2*pt + 1):
        return None
    return n*(-2*p*pt**2 + 4*p*pt - 2*p + 2*pt**2*tp - 2*pt*tp + sqrt(tp)*sqrt(8*p*pt**2 - 16*p*pt + 8*p - 4*pt**2*tp + 4*pt*tp + tp) - tp)/(2*p*(pt**2 - 2*pt + 1))

def pt_tn(*, pt, p, n, tp, **kwargs):
    """
    Solves tn from the score pt

    Args:
        pt (float|Interval|IntervalUnion): the value or interval for the score pt
        p (int): the p count
        n (int): the n count
        tp (int): the tp count
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tn
    """
    return unify_results([pt_tn_0(pt=pt, p=p, n=n, tp=tp),
                          pt_tn_1(pt=pt, p=p, n=n, tp=tp)])
def dor_tp(*, p, tn, n, dor, **kwargs):
    """
    Solves tp from the score dor

    Args:
        p (int): the p count
        tn (int): the tn count
        n (int): the n count
        dor (float|Interval|IntervalUnion): the value or interval for the score dor
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tp
    """
    if is_zero(dor*n - dor*tn + tn):
        return None
    return dor*p*(n - tn)/(dor*n - dor*tn + tn)

def dor_tn(*, p, n, dor, tp, **kwargs):
    """
    Solves tn from the score dor

    Args:
        p (int): the p count
        n (int): the n count
        dor (float|Interval|IntervalUnion): the value or interval for the score dor
        tp (int): the tp count
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tn
    """
    if is_zero(dor*p - dor*tp + tp):
        return None
    return dor*n*(p - tp)/(dor*p - dor*tp + tp)

def ji_tp(*, n, p, tn, ji, **kwargs):
    """
    Solves tp from the score ji

    Args:
        n (int): the n count
        p (int): the p count
        tn (int): the tn count
        ji (float|Interval|IntervalUnion): the value or interval for the score ji
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tp
    """
    return ji*(n + p - tn)

def ji_tn(*, n, tp, p, ji, **kwargs):
    """
    Solves tn from the score ji

    Args:
        n (int): the n count
        tp (int): the tp count
        p (int): the p count
        ji (float|Interval|IntervalUnion): the value or interval for the score ji
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tn
    """
    if is_zero(ji):
        return None
    return n + p - tp/ji

def bacc_tp(*, p, tn, n, bacc, **kwargs):
    """
    Solves tp from the score bacc

    Args:
        p (int): the p count
        tn (int): the tn count
        n (int): the n count
        bacc (float|Interval|IntervalUnion): the value or interval for the score bacc
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tp
    """
    if is_zero(n):
        return None
    return p*(2*bacc*n - tn)/n

def bacc_tn(*, p, tp, n, bacc, **kwargs):
    """
    Solves tn from the score bacc

    Args:
        p (int): the p count
        tp (int): the tp count
        n (int): the n count
        bacc (float|Interval|IntervalUnion): the value or interval for the score bacc
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tn
    """
    if is_zero(p):
        return None
    return n*(2*bacc*p - tp)/p

def kappa_tp(*, kappa, p, tn, n, **kwargs):
    """
    Solves tp from the score kappa

    Args:
        kappa (float|Interval|IntervalUnion): the value or interval for the score kappa
        p (int): the p count
        tn (int): the tn count
        n (int): the n count
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tp
    """
    if is_zero(-kappa*n + kappa*p + 2*n):
        return None
    return (kappa*n**2 - kappa*n*tn + kappa*p**2 + kappa*p*tn + 2*n*p - 2*p*tn)/(-kappa*n + kappa*p + 2*n)

def kappa_tn(*, kappa, p, n, tp, **kwargs):
    """
    Solves tn from the score kappa

    Args:
        kappa (float|Interval|IntervalUnion): the value or interval for the score kappa
        p (int): the p count
        n (int): the n count
        tp (int): the tp count
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tn
    """
    if is_zero(kappa*n - kappa*p + 2*p):
        return None
    return (kappa*n**2 + kappa*n*tp + kappa*p**2 - kappa*p*tp + 2*n*p - 2*n*tp)/(kappa*n - kappa*p + 2*p)

def p4_tp_0(*, p, tn, p4, n, **kwargs):
    """
    Solves tp from the score p4

    Args:
        p (int): the p count
        tn (int): the tn count
        p4 (float|Interval|IntervalUnion): the value or interval for the score p4
        n (int): the n count
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tp
    """
    if is_less_than_zero(n**2*p4**2 + 2*n*p*p4**2 + 8*n*p4**2*tn - 8*n*p4*tn + p**2*p4**2 + 8*p*p4**2*tn - 8*p*p4*tn - 16*p4*tn**2 + 16*tn**2):
        return None
    if is_zero(p4):
        return None
    return n/2 + p/2 + tn - 2*tn/p4 - sqrt(n**2*p4**2 + 2*n*p*p4**2 + 8*n*p4**2*tn - 8*n*p4*tn + p**2*p4**2 + 8*p*p4**2*tn - 8*p*p4*tn - 16*p4*tn**2 + 16*tn**2)/(2*p4)

def p4_tp_1(*, p, tn, p4, n, **kwargs):
    """
    Solves tp from the score p4

    Args:
        p (int): the p count
        tn (int): the tn count
        p4 (float|Interval|IntervalUnion): the value or interval for the score p4
        n (int): the n count
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tp
    """
    if is_less_than_zero(n**2*p4**2 + 2*n*p*p4**2 + 8*n*p4**2*tn - 8*n*p4*tn + p**2*p4**2 + 8*p*p4**2*tn - 8*p*p4*tn - 16*p4*tn**2 + 16*tn**2):
        return None
    if is_zero(p4):
        return None
    return n/2 + p/2 + tn - 2*tn/p4 + sqrt(n**2*p4**2 + 2*n*p*p4**2 + 8*n*p4**2*tn - 8*n*p4*tn + p**2*p4**2 + 8*p*p4**2*tn - 8*p*p4*tn - 16*p4*tn**2 + 16*tn**2)/(2*p4)

def p4_tp(*, p, tn, p4, n, **kwargs):
    """
    Solves tp from the score p4

    Args:
        p (int): the p count
        tn (int): the tn count
        p4 (float|Interval|IntervalUnion): the value or interval for the score p4
        n (int): the n count
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tp
    """
    return unify_results([p4_tp_0(p=p, tn=tn, p4=p4, n=n),
                          p4_tp_1(p=p, tn=tn, p4=p4, n=n)])
def p4_tn_0(*, p, p4, n, tp, **kwargs):
    """
    Solves tn from the score p4

    Args:
        p (int): the p count
        p4 (float|Interval|IntervalUnion): the value or interval for the score p4
        n (int): the n count
        tp (int): the tp count
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tn
    """
    if is_less_than_zero(n**2*p4**2 + 2*n*p*p4**2 + 8*n*p4**2*tp - 8*n*p4*tp + p**2*p4**2 + 8*p*p4**2*tp - 8*p*p4*tp - 16*p4*tp**2 + 16*tp**2):
        return None
    if is_zero(p4):
        return None
    return n/2 + p/2 + tp - 2*tp/p4 - sqrt(n**2*p4**2 + 2*n*p*p4**2 + 8*n*p4**2*tp - 8*n*p4*tp + p**2*p4**2 + 8*p*p4**2*tp - 8*p*p4*tp - 16*p4*tp**2 + 16*tp**2)/(2*p4)

def p4_tn_1(*, p, p4, n, tp, **kwargs):
    """
    Solves tn from the score p4

    Args:
        p (int): the p count
        p4 (float|Interval|IntervalUnion): the value or interval for the score p4
        n (int): the n count
        tp (int): the tp count
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tn
    """
    if is_less_than_zero(n**2*p4**2 + 2*n*p*p4**2 + 8*n*p4**2*tp - 8*n*p4*tp + p**2*p4**2 + 8*p*p4**2*tp - 8*p*p4*tp - 16*p4*tp**2 + 16*tp**2):
        return None
    if is_zero(p4):
        return None
    return n/2 + p/2 + tp - 2*tp/p4 + sqrt(n**2*p4**2 + 2*n*p*p4**2 + 8*n*p4**2*tp - 8*n*p4*tp + p**2*p4**2 + 8*p*p4**2*tp - 8*p*p4*tp - 16*p4*tp**2 + 16*tp**2)/(2*p4)

def p4_tn(*, p, p4, n, tp, **kwargs):
    """
    Solves tn from the score p4

    Args:
        p (int): the p count
        p4 (float|Interval|IntervalUnion): the value or interval for the score p4
        n (int): the n count
        tp (int): the tp count
        kwargs (dict): additional keyword arguments

    Returns:
        float|Interval|IntervalUnion: the value or interval for tn
    """
    return unify_results([p4_tn_0(p=p, p4=p4, n=n, tp=tp),
                          p4_tn_1(p=p, p4=p4, n=n, tp=tp)])

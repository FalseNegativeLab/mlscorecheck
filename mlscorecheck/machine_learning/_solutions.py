"""
This module implements the solutions for tp and tn
"""

__all__ = ['solve_sens_spec',
           'solve_sens_acc',
           'solve_sens_ppv',
           'solve_sens_npv',
           'solve_spec_acc',
           'solve_spec_ppv',
           'solve_spec_npv',
           'solve_acc_ppv',
           'solve_acc_npv',
           'solve_ppv_npv']

def solve_sens_spec(*, sens, spec, p, n):
    tp = p*sens
    tn = n*spec

    return tp, tn

def solve_sens_acc(*, sens, acc, p, n):
    tp = p*sens
    tn = acc*n + acc*p - p*sens

    return tp, tn

def solve_sens_ppv(*, sens, ppv, p, n):
    tp = p*sens
    tn = n + p*sens - p*sens/ppv

    return tp, tn

def solve_sens_npv(*, sens, npv, p):
    tp = p*sens
    tn = npv*p*(sens - 1)/(npv - 1)

    return tp, tn

def solve_spec_acc(*, spec, acc, p, n):
    tp = acc*n + acc*p - n*spec
    tn = n*spec

    return tp, tn

def solve_spec_ppv(*, spec, ppv, n):
    tp = n*ppv*(spec - 1)/(ppv - 1)
    tn = n*spec

    return tp, tn

def solve_spec_npv(*, spec, npv, p, n):
    tp = n*spec - n*spec/npv + p
    tn = n*spec

    return tp, tn

def solve_acc_ppv(*, acc, ppv, p, n):
    tp = ppv*(acc*n + acc*p - n)/(2*ppv - 1)
    tn = (acc*n*ppv - acc*n + acc*p*ppv - acc*p + n*ppv)/(2*ppv - 1)

    return tp, tn

def solve_acc_npv(*, acc, npv, p, n):
    tp = (acc*n*npv - acc*n + acc*npv*p - acc*p + npv*p)/(2*npv - 1)
    tn = npv*(acc*n + acc*p - p)/(2*npv - 1)

    return tp, tn

def solve_ppv_npv(*, ppv, npv, p, n):
    tp = ppv*(n*npv - n + npv*p)/(npv + ppv - 1)
    tn = npv*(n*ppv + p*ppv - p)/(npv + ppv - 1)

    return tp, tn,

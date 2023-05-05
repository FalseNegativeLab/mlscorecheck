"""
This module implements the solvers
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
           'solve_ppv_npv',
           'solve_sens_f_beta',
           'solve_spec_f_beta',
           'solve_acc_f_beta',
           'solve_ppv_f_beta',
           'solve_npv_f_beta',
           'solver_functions']

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

def solve_sens_npv(*, sens, npv, p, n):
    _ = n

    tp = p*sens
    tn = npv*p*(sens - 1)/(npv - 1)

    return tp, tn

def solve_spec_acc(*, spec, acc, p, n):
    tp = acc*n + acc*p - n*spec
    tn = n*spec

    return tp, tn

def solve_spec_ppv(*, spec, ppv, p, n):
    _ = p
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

    return tp, tn

def solve_sens_f_beta(*, beta, f_beta, sens, p, n):
    tp = p*sens
    tn = beta**2*p - beta**2*p*sens/f_beta + n + p*sens - p*sens/f_beta

    return tp, tn

def solve_spec_f_beta(*, beta, f_beta, spec, p, n):
    tp = f_beta*(beta**2*p - n*spec + n)/(beta**2 - f_beta + 1)
    tn = n*spec

    return tp, tn

def solve_acc_f_beta(*, beta, f_beta, acc, p, n):
    tp = f_beta*(-acc*n - acc*p + beta**2*p + n)/(beta**2 - 2*f_beta + 1)
    tn = (acc*beta**2*n + acc*beta**2*p - acc*f_beta*n - acc*f_beta*p + acc*n + \
                        acc*p - beta**2*f_beta*p - f_beta*n)/(beta**2 - 2*f_beta + 1)

    return tp, tn

def solve_npv_f_beta(*, beta, f_beta, npv, p, n):
    tp = f_beta*(beta**2*npv*p - beta**2*p + n*npv - n + npv*p)\
                            /(beta**2*npv - beta**2 + f_beta + npv - 1)
    tn = npv*(beta**2*f_beta*p - beta**2*p + f_beta*n + f_beta*p - p)\
                            /(beta**2*npv - beta**2 + f_beta + npv - 1)

    return tp, tn

def solve_ppv_f_beta(*, beta, f_beta, ppv, p, n):
    tp = beta**2*f_beta*p*ppv/(beta**2*ppv - f_beta + ppv)
    tn = (beta**2*f_beta*p*ppv - beta**2*f_beta*p + beta**2*n*ppv - f_beta*n + n*ppv)\
                            /(beta**2*ppv - f_beta + ppv)

    return tp, tn

solver_functions = {
    tuple(sorted(('sens', 'spec'))): solve_sens_spec,
    tuple(sorted(('sens', 'acc'))): solve_sens_acc,
    tuple(sorted(('sens', 'ppv'))): solve_sens_ppv,
    tuple(sorted(('sens', 'npv'))): solve_sens_npv,
    tuple(sorted(('sens', 'f_beta'))): solve_sens_f_beta,
    tuple(sorted(('spec', 'acc'))): solve_spec_acc,
    tuple(sorted(('spec', 'ppv'))): solve_spec_ppv,
    tuple(sorted(('spec', 'npv'))): solve_spec_npv,
    tuple(sorted(('spec', 'f_beta'))): solve_spec_f_beta,
    tuple(sorted(('acc', 'ppv'))): solve_acc_ppv,
    tuple(sorted(('acc', 'npv'))): solve_acc_npv,
    tuple(sorted(('acc', 'f_beta'))): solve_acc_f_beta,
    tuple(sorted(('ppv', 'npv'))): solve_ppv_npv
}

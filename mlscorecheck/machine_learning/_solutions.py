"""
This module implements the solutions for tp and tn
"""

import itertools

from ..core import (accuracy,
                    sensitivity,
                    specificity,
                    positive_predictive_value,
                    negative_predictive_value,
                    Interval)

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
           'problems_to_check',
           'check']

def problems_to_check(all_scores):
    combinations = itertools.combinations(all_scores, 2)

    score_set = set(all_scores)

    problems = [(set(comb), score_set.difference(set(comb))) for comb in combinations]

    return problems

def check_integer_constraints(tp, tn):
    details = {}
    decision = True
    if tp.integer():
        {'tp_integer': True}
        decision = decision and True
    else:
        {'tp_integer': False}
        decision = decision and False
    if tn.integer():
        {'tn_integer': True}
        decision = decision and True
    else:
        {'tn_integer': False}
        decision = decision and False

    details['decision'] = decision

    return decision, details

def check(*, p, n, eps, acc=None, sens=None, spec=None, ppv=None, npv=None, return_details=False):
    scores = {'acc': acc, 'sens': sens, 'spec': spec, 'ppv': ppv, 'npv': npv}
    scores = {key: value for key, value in scores.items() if value is not None}

    if 'acc' in scores and 'sens' in scores and 'spec' in scores:
        if not ((sens <= acc <= spec) or (spec <= acc <= sens)):
            if not return_details:
                return False

            return False, {'acc_sens_spec_check_failed': True,
                           'acc': acc,
                           'sens': sens,
                           'spec': spec}

    problems = problems_to_check(list(scores.keys()))

    intervals = {key: Interval(value - eps, value + eps) for key, value in scores.items()}
    intervals['p'] = p
    intervals['n'] = n

    results = []
    decisions = []

    for problem in problems:
        if problem[0] == {'acc', 'sens'}:
            tp, tn = solve_sens_acc(**intervals)
        elif problem[0] == {'acc', 'spec'}:
            tp, tn = solve_spec_acc(**intervals)
        elif problem[0] == {'acc', 'npv'}:
            tp, tn = solve_acc_npv(**intervals)
        elif problem[0] == {'acc', 'ppv'}:
            tp, tn = solve_acc_ppv(**intervals)
        elif problem[0] == {'sens', 'spec'}:
            tp, tn = solve_sens_spec(**intervals)
        elif problem[0] == {'sens', 'ppv'}:
            tp, tn = solve_sens_ppv(**intervals)
        elif problem[0] == {'sens', 'npv'}:
            tp, tn = solve_sens_npv(**intervals)
        elif problem[0] == {'spec', 'ppv'}:
            tp, tn = solve_spec_ppv(**intervals)
        elif problem[0] == {'spec', 'npv'}:
            tp, tn = solve_spec_npv(**intervals)
        elif problem[0] == {'npv', 'ppv'}:
            tp, tn = solve_ppv_npv(**intervals)

        int_decision, int_details = check_integer_constraints(tp, tn)
        if not int_decision:
            decisions.append(False)
            results.append(int_details)
            continue

        fp = n - tn
        fn = p - tp

        for score in problem[1]:
            if score == 'acc':
                decision, details = check_acc(acc=acc, tp_i=tp, tn_i=tn, total=p+n)
            if score == 'sens':
                decision, details = check_sens(sens=sens, tp_i=tp, p=p)
            if score == 'spec':
                decision, details = check_spec(spec=spec, tn_i=tn, n=n)
            if score == 'ppv':
                decision, details = check_ppv(ppv=ppv, tp_i=tp, fp_i=fp)
            if score == 'npv':
                decision, details = check_npv(npv=npv, tn_i=tn, fn_i=fn)

            details['intervals_computed_from'] = {key: intervals[key].to_tuple()
                                                  for key in problem[0]}
            details['atoms'] = {'tp': tp.to_tuple(),
                                'tn': tn.to_tuple(),
                                'fp': fp.to_tuple(),
                                'fn': fn.to_tuple()}

            decisions.append(decision)
            results.append(details)

    if not return_details:
        return all(decisions)

    return all(decisions), results

def check_acc(*, acc, tp_i, tn_i, total):
    acc_i = accuracy(tp=tp_i, tn=tn_i, total=total)
    decision = acc_i.contains(acc)

    return decision, {'acc_interval': acc_i.to_tuple(),
                        'tp_interval': tp_i.to_tuple(),
                        'tn_interval': tn_i.to_tuple(),
                        'total': total,
                        'acc_real': acc,
                        'decision': decision}

def check_sens(*, sens, tp_i, p):
    sens_i = sensitivity(tp=tp_i, p=p)
    decision = sens_i.contains(sens)

    return decision, {'sens_interval': sens_i.to_tuple(),
                        'tp_interval': tp_i.to_tuple(),
                        'p': p,
                        'sens_real': sens,
                        'decision': decision}

def check_spec(*, spec, tn_i, n):
    spec_i = specificity(tn=tn_i, n=n)
    decision = spec_i.contains(spec)

    return decision, {'spec_interval': spec_i.to_tuple(),
                        'tn_interval': tn_i.to_tuple(),
                        'n': n,
                        'spec_real': spec,
                        'decision': decision}

def check_npv(*, npv, tn_i, fn_i):
    npv_i = negative_predictive_value(tn=tn_i, fn=fn_i)
    decision = npv_i.contains(npv)

    return decision, {'npv_interval': npv_i.to_tuple(),
                        'tn_interval': tn_i.to_tuple(),
                        'fn_interval': fn_i.to_tuple(),
                        'npv_real': npv,
                        'decision': decision}

def check_ppv(*, ppv, tp_i, fp_i):
    ppv_i = positive_predictive_value(tp=tp_i, fp=fp_i)
    decision = ppv_i.contains(ppv)

    return decision, {'ppv_interval': ppv_i.to_tuple(),
                        'tp_interval': tp_i.to_tuple(),
                        'fp_interval': fp_i.to_tuple(),
                        'ppv_real': ppv,
                        'decision': decision}

def solve_sens_spec(*, sens, spec, p, n, **_kwargs):
    tp = p*sens
    tn = n*spec

    return tp, tn

def solve_sens_acc(*, sens, acc, p, n, **_kwargs):
    tp = p*sens
    tn = acc*n + acc*p - p*sens

    return tp, tn

def solve_sens_ppv(*, sens, ppv, p, n, **_kwargs):
    tp = p*sens
    tn = n + p*sens - p*sens/ppv

    return tp, tn

def solve_sens_npv(*, sens, npv, p, **_kwargs):
    tp = p*sens
    tn = npv*p*(sens - 1)/(npv - 1)

    return tp, tn

def solve_spec_acc(*, spec, acc, p, n, **_kwargs):
    tp = acc*n + acc*p - n*spec
    tn = n*spec

    return tp, tn

def solve_spec_ppv(*, spec, ppv, n, **_kwargs):
    tp = n*ppv*(spec - 1)/(ppv - 1)
    tn = n*spec

    return tp, tn

def solve_spec_npv(*, spec, npv, p, n, **_kwargs):
    tp = n*spec - n*spec/npv + p
    tn = n*spec

    return tp, tn

def solve_acc_ppv(*, acc, ppv, p, n, **_kwargs):
    tp = ppv*(acc*n + acc*p - n)/(2*ppv - 1)
    tn = (acc*n*ppv - acc*n + acc*p*ppv - acc*p + n*ppv)/(2*ppv - 1)

    return tp, tn

def solve_acc_npv(*, acc, npv, p, n, **_kwargs):
    tp = (acc*n*npv - acc*n + acc*npv*p - acc*p + npv*p)/(2*npv - 1)
    tn = npv*(acc*n + acc*p - p)/(2*npv - 1)

    return tp, tn

def solve_ppv_npv(*, ppv, npv, p, n, **_kwargs):
    tp = ppv*(n*npv - n + npv*p)/(npv + ppv - 1)
    tn = npv*(n*ppv + p*ppv - p)/(npv + ppv - 1)

    return tp, tn,

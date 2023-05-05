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
from ._solvers import solver_functions

__all__ = ['problems_to_check',
           'check']

def check_acc(*, acc, tp_i, tn_i, p, n):
    acc_i = accuracy(tp=tp_i, tn=tn_i, p=p, n=n)
    decision = acc_i.contains(acc)

    return decision, {'acc_interval': acc_i.to_tuple(),
                        'tp_interval': tp_i.to_tuple(),
                        'tn_interval': tn_i.to_tuple(),
                        'p': p,
                        'n': n,
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

def check_f_beta(*, f_beta, beta, tp_i, fp_i, fn_i):
    f_beta_i = f_beta(tp=tp_i, fp=fp_i, fn=fn_i, beta=beta)
    decision = f_beta_i.contains(f_beta)

    return decision, {'f_beta_interval': f_beta_i.to_tuple(),
                      'tp_interval': tp_i.to_tuple(),
                      'fp_interval': fp_i.to_tuple(),
                      'fn_interval': fn_i.to_tuple(),
                      'beta': beta,
                      'f_beta_real': f_beta,
                      'decision': decision}

check_functions = {
    'acc': (check_acc, (('tp', 'tn'), ('p', 'n'))),
    'sens': (check_sens, (('tp',), ('p',))),
    'spec': (check_spec, (('tn',), ('n',))),
    'ppv': (check_ppv, (('tp', 'fp'), tuple())),
    'npv': (check_npv, (('tn', 'fn'), tuple())),
    'f_beta': (check_f_beta, (('tp', 'fp', 'fn'), 'beta'))
}

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

def check_acc_sens_spec(*, scores, return_details=False):
    acc, sens, spec = scores['acc'], scores['sens'], scores['spec']

    decision = False
    if min(sens, spec) <= acc <= max(sens, spec):
        decision = True

    if not return_details:
        return decision

    return decision, {'acc_sens_spec_check': decision,
                        'acc': acc,
                        'sens': sens,
                        'spec': spec}

def check_f_beta_sens_ppv(*, scores, return_details=False):
    f_beta, sens, ppv = scores['f_beta'], scores['sens'], scores['ppv']

    decision = False
    if min(sens, ppv) <= f_beta <= max(sens, ppv):
        decision = True

    if not return_details:
        return decision

    return decision, {'f_beta_sens_ppv_check': decision,
                        'f_beta': f_beta,
                        'sens': sens,
                        'ppv': ppv}

def check_problem(problem, scores, intervals, all_params):
    solver = solver_functions[tuple(sorted(tuple(problem[0])))]
    params = {'p': all_params['p'],
              'n': all_params['n'],
              **{score: scores[score] for score in problem}}
    if 'f_beta' in problem[0]:
        params['beta'] = all_params['beta']
    tp, tn = solver(**params)

    fp = all_params['n'] - tn
    fn = all_params['p'] - tp

    decisions = []
    results = []

    for score in problem[1]:
        check_specification = check_functions[score]
        check_func = check_specification[0]
        check_params = {**{intervals[key] for key in check_specification[1][0]},
                        **{all_params[key] for key in check_specification[1][1]}}

        decision, details = check_func(**check_params)

        details['intervals_computed_from'] = {key: intervals[key].to_tuple()
                                                for key in problem[0]}
        details['atoms'] = {'tp': tp.to_tuple(),
                            'tn': tn.to_tuple(),
                            'fp': fp.to_tuple(),
                            'fn': fn.to_tuple()}

        decisions.append(decision)
        results.append(details)

    return decisions, results

def check(*, p, n, eps,
          acc=None,
          sens=None,
          spec=None,
          ppv=None,
          npv=None,
          f_beta=None,
          beta=None,
          return_details=False):

    if f_beta is not None and beta is None:
        raise ValueError('f_beta is set, but beta is not specified.\n'
                         'If you intended to use f_1 score, set beta=1')

    scores = {'acc': acc, 'sens': sens, 'spec': spec, 'ppv': ppv, 'npv': npv, 'f_beta': f_beta}
    scores = {key: value for key, value in scores.items() if value is not None}
    all_params = {**scores, **{'p': p, 'n': n}}
    if beta is not None:
        all_params['beta'] = beta

    problems = problems_to_check(list(scores.keys()))

    intervals = {key: Interval(value - eps, value + eps) for key, value in scores.items()}

    results = []
    decisions = []

    for problem in problems:
        check_problem(problem, scores, intervals, all_params)

    if not return_details:
        return all(decisions)

    return all(decisions), results



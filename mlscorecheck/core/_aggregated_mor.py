"""
This module implements some core consistency tests for mean-of-rations aggregations
"""

import numpy as np
import pulp as pl

from ._aggregated_assemble_results import assemble_results_problem, assemble_results_problems
from ._folds import (stratified_configurations_sklearn, determine_fold_configurations)

__all__ = ['consistency_1',
            'consistency_grouped',
            'add_accuracy_mor',
            'add_balanced_accuracy_mor',
            'add_specificity_mor',
            'add_sensitivity_mor',
            'generate_structure_1',
            'generate_structure_group']

def add_accuracy_mor(problem, tps, tns, p, n, acc, eps, groups, acc_bounds=None):
    n_items = len(tps)
    totals = np.array(p) + np.array(n)

    n_groups = len(groups)
    weights = np.repeat(1.0, n_items)
    for group in groups:
        for idx in group:
            weights[idx] = len(group)

    problem += sum((1.0 / (n_groups * weights[idx])) * (tps[idx] + tns[idx]) / totals[idx] for idx in range(n_items)) <= acc + eps
    problem += sum((1.0 / (n_groups * weights[idx])) * (-1) * (tps[idx] + tns[idx]) / totals[idx] for idx in range(n_items)) <= -(acc - eps)

    for idx in range(n_groups):
        if acc_bounds[idx] is not None:
            group = groups[idx]
            denom = 1.0 / len(group)
            problem += sum((tps[jdx] + tns[jdx]) * (1.0 / (p[jdx] + n[jdx])) * denom for jdx in group) >= acc_bounds[idx][0]
            problem += sum((tps[jdx] + tns[jdx]) * (1.0 / (p[jdx] + n[jdx])) * denom for jdx in group) <= acc_bounds[idx][1]

    return problem

def add_balanced_accuracy_mor(problem, tps, tns, p, n, bacc, eps, groups, bacc_bounds=None):
    n_items = len(tps)

    n_groups = len(groups)
    weights = np.repeat(1.0, n_items)
    for group in groups:
        for idx in group:
            weights[idx] = len(group)

    problem += 0.5 * (sum((1.0 / (n_groups * weights[idx])) * (tps[idx] * (1.0 / p[idx])) for idx in range(n_items)) + sum((1.0 / (n_groups*weights[idx])) * (tns[idx] * (1.0 / n[idx])) for idx in range(n_items))) <= bacc + eps
    problem += 0.5 * (sum((1.0 / (n_groups * weights[idx])) * (-1) * (tps[idx] * (1.0 / p[idx])) for idx in range(n_items)) + sum((1.0 / (n_groups*weights[idx])) * (-1) * (tns[idx] * (1.0 / n[idx])) for idx in range(n_items))) <= -(bacc - eps)

    for idx in range(n_groups):
        if bacc_bounds[idx] is not None:
            group = groups[idx]
            denom_p = 1.0 / (len(group))
            denom_n = 1.0 / (len(group))
            problem += sum((tps[jdx]*(1.0/p[jdx])*denom_p + tns[jdx]*(1.0/n[jdx])*denom_n)/2 for jdx in group) >= bacc_bounds[idx][0]
            problem += sum((tps[jdx]*(1.0/p[jdx])*denom_p + tns[jdx]*(1.0/n[jdx])*denom_n)/2 for jdx in group) <= bacc_bounds[idx][1]

    return problem

def add_sensitivity_mor(problem, tps, p, sens, eps, groups, sens_bounds=None):
    n_items = len(tps)

    n_groups = len(groups)
    weights = np.repeat(1.0, n_items)
    for group in groups:
        for idx in group:
            weights[idx] = len(group)

    problem += sum((1.0 / (n_groups * weights[idx])) * (tps[idx] * (1.0 / p[idx])) for idx in range(n_items)) <= sens + eps
    problem += sum((1.0 / (n_groups * weights[idx])) * (-1) * (tps[idx] * (1.0 / p[idx])) for idx in range(n_items)) <= -(sens - eps)

    for idx in range(n_groups):
        if sens_bounds[idx] is not None:
            group = groups[idx]
            denom_p = 1.0 / (len(group))

            problem += sum((tps[jdx] * (1.0 / p[jdx])) * denom_p for jdx in group) >= sens_bounds[idx][0]
            problem += sum((tps[jdx] * (1.0 / p[jdx])) * denom_p for jdx in group) <= sens_bounds[idx][1]

    return problem

def add_specificity_mor(problem, tns, n, spec, eps, groups, spec_bounds=None):
    n_items = len(tns)

    n_groups = len(groups)
    weights = np.repeat(1.0, n_items)
    for group in groups:
        for idx in group:
            weights[idx] = len(group)

    problem += sum((1.0 / (n_groups * weights[idx])) * (tns[idx] * (1.0 / n[idx])) for idx in range(n_items)) <= spec + eps
    problem += sum((1.0 / (n_groups * weights[idx])) * (-1) * (tns[idx] * (1.0 / n[idx])) for idx in range(n_items)) <= -(spec - eps)

    for idx in range(n_groups):
        if spec_bounds[idx] is not None:
            group = groups[idx]
            denom_n = 1.0 / len(group)

            problem += sum((tns[jdx]) * (1.0 / n[jdx]) * denom_n for jdx in group) >= spec_bounds[idx][0]
            problem += sum((tns[jdx]) * (1.0 / n[jdx]) * denom_n for jdx in group) <= spec_bounds[idx][1]

    return problem

def generate_structure_group(problem_setup):
    ps = []
    ns = []
    tps = []
    tns = []
    score_bounds = []
    groups = []

    for pdx, problem in enumerate(problem_setup):
        if 'fold_configuration' in problem:
            folds = problem['fold_configuration']
        else:
            folds = determine_fold_configurations(problem['p'], problem['n'], problem['n_folds'], problem['n_repeats'])

        groups.append([])
        for fdx, fold in enumerate(folds):
            ps.append(fold['p'])
            ns.append(fold['n'])
            tps.append(pl.LpVariable(f'tp_{pdx}_{fdx}', 0, fold['p'], pl.LpInteger))
            tns.append(pl.LpVariable(f'tn_{pdx}_{fdx}', 0, fold['n'], pl.LpInteger))
            groups[-1].append(len(tps)-1)
        score_bounds.append(problem.get('score_bounds'))

    return ps, ns, tps, tns, score_bounds, groups

def generate_structure_1(problem):
    ps = []
    ns = []
    tps = []
    tns = []
    score_bounds = []
    tptn_bounds = []
    groups = []

    if 'fold_configuration' in problem:
        folds = problem['fold_configuration']
    else:
        folds = determine_fold_configurations(problem['p'], problem['n'], problem['n_folds'], problem['n_repeats'])

    for fdx, fold in enumerate(folds):
        ps.append(fold['p'])
        ns.append(fold['n'])
        tps.append(pl.LpVariable(f'tp_{fdx}', 0, fold['p'], pl.LpInteger))
        tns.append(pl.LpVariable(f'tn_{fdx}', 0, fold['n'], pl.LpInteger))
        groups.append([fdx])
        score_bounds.append(fold.get('score_bounds', problem.get('score_bounds')))
        tptn_bounds.append(fold.get('tptn_bounds', problem.get('tptn_bounds')))

    return ps, ns, tps, tns, score_bounds, groups, tptn_bounds

def consistency_1(problem,
                    scores,
                    eps,
                    return_details=False):
    """
    Checking the consistency of aggregated figures, supposing mean of ratios calculation

    Args:
        p (np.array): vector of the assumed numbers of positives
        n (np.array): vector of the assumed numbers of negatives
        acc (float): the observed mean accuracy
        sens (float): the observed mean sensitivity
        spec (float): the observed mean specificity
        eps (float): the assumed +/- numerical uncertainty of the observed figures

    Returns:
        boolean: True if the observed scores are consistent with the assumed figures, False otherwise
    """
    lp_problem = pl.LpProblem("feasibility")

    ps, ns, tps, tns, score_bounds, groups, tptn_bounds = generate_structure_1(problem)

    lp_problem += tps[0]

    for score in scores:
        if score == 'acc':
            prob = add_accuracy_mor(problem=lp_problem,
                                    tps=tps,
                                    tns=tns,
                                    p=ps,
                                    n=ns,
                                    acc=scores[score],
                                    eps=eps[score] if isinstance(eps, dict) else eps,
                                    groups=groups,
                                    acc_bounds=[sb.get(score) if sb is not None else None for sb in score_bounds])
        if score == 'bacc':
            prob = add_balanced_accuracy_mor(problem=lp_problem,
                                    tps=tps,
                                    tns=tns,
                                    p=ps,
                                    n=ns,
                                    bacc=scores[score],
                                    eps=eps[score] if isinstance(eps, dict) else eps,
                                    groups=groups,
                                    bacc_bounds=[sb.get(score) if sb is not None else None for sb in score_bounds])
        if score == 'sens':
            prob = add_sensitivity_mor(problem=lp_problem,
                                        tps=tps,
                                        p=ps,
                                        sens=scores[score],
                                        eps=eps[score] if isinstance(eps, dict) else eps,
                                        groups=groups,
                                        sens_bounds=[sb.get(score) if sb is not None else None for sb in score_bounds])
        if score == 'spec':
            prob = add_specificity_mor(problem=lp_problem,
                                        tns=tns,
                                        n=ns,
                                        spec=scores[score],
                                        eps=eps[score] if isinstance(eps, dict) else eps,
                                        groups=groups,
                                        spec_bounds=[sb.get(score) if sb is not None else None for sb in score_bounds])

    if tptn_bounds is not None:
        for idx in range(len(tps)):
            if tptn_bounds[idx] is not None:
                if tptn_bounds[idx].get('tp') is not None:
                    prob += tps[idx] >= tptn_bounds[idx]['tp'][0]
                    prob += tps[idx] <= tptn_bounds[idx]['tp'][1]
                if tptn_bounds[idx].get('tn') is not None:
                    prob += tns[idx] >= tptn_bounds[idx]['tn'][0]
                    prob += tns[idx] <= tptn_bounds[idx]['tn'][1]

    if not return_details:
        return prob.solve() == 1

    return prob.solve() == 1, assemble_results_problem(prob, ps, ns)

def consistency_grouped(problems,
                            scores,
                            eps,
                            return_details=False):
    """
    Checking the consistency of aggregated figures, supposing mean of ratios calculation

    Args:
        p (np.array): vector of the assumed numbers of positives
        n (np.array): vector of the assumed numbers of negatives
        acc (float): the observed mean accuracy
        sens (float): the observed mean sensitivity
        spec (float): the observed mean specificity
        eps (float): the assumed +/- numerical uncertainty of the observed figures

    Returns:
        boolean: True if the observed scores are consistent with the assumed figures, False otherwise
    """
    lp_problem = pl.LpProblem("feasibility")

    ps, ns, tps, tns, score_bounds, groups = generate_structure_group(problems)

    lp_problem += tps[0]

    for score in scores:
        if score == 'acc':
            prob = add_accuracy_mor(problem=lp_problem,
                                    tps=tps,
                                    tns=tns,
                                    p=ps,
                                    n=ns,
                                    acc=scores[score],
                                    eps=eps[score] if isinstance(eps, dict) else eps,
                                    groups=groups,
                                    acc_bounds=[sb.get(score) if sb is not None else None for sb in score_bounds])
        if score == 'bacc':
            prob = add_balanced_accuracy_mor(problem=lp_problem,
                                    tps=tps,
                                    tns=tns,
                                    p=ps,
                                    n=ns,
                                    bacc=scores[score],
                                    eps=eps[score] if isinstance(eps, dict) else eps,
                                    groups=groups,
                                    bacc_bounds=[sb.get(score) if sb is not None else None for sb in score_bounds])
        if score == 'sens':
            prob = add_sensitivity_mor(problem=lp_problem,
                                        tps=tps,
                                        p=ps,
                                        sens=scores[score],
                                        eps=eps[score] if isinstance(eps, dict) else eps,
                                        groups=groups,
                                        sens_bounds=[sb.get(score) if sb is not None else None for sb in score_bounds])
        if score == 'spec':
            prob = add_specificity_mor(problem=lp_problem,
                                        tns=tns,
                                        n=ns,
                                        spec=scores[score],
                                        eps=eps[score] if isinstance(eps, dict) else eps,
                                        groups=groups,
                                        spec_bounds=[sb.get(score) if sb is not None else None for sb in score_bounds])

    if not return_details:
        return prob.solve() == 1

    return prob.solve() == 1, assemble_results_problems(prob, ps, ns, groups)


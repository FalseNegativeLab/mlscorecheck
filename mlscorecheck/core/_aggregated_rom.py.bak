"""
This module implements some consistency checks for ratio-of-means aggregations
"""

import numpy as np
import pulp as pl

from ._aggregated_assemble_results import assemble_results

__all__ = ['consistency_aggregated_integer_programming_rom']

def add_accuracy_rom(problem, tps, tns, p, n, acc, eps, acc_bounds=None):
    n_items = len(tps)
    total = np.sum(p + n)

    problem += sum((1.0 / total) * (tps[idx] + tns[idx]) for idx in range(n_items)) <= acc + eps
    problem += sum((1.0 / total) * (-1) * (tps[idx] + tns[idx]) for idx in range(n_items)) <= -(acc - eps)

    if acc_bounds is not None:
        if isinstance(acc_bounds, tuple):
            for idx in range(n_items):
                problem += (tps[idx] + tns[idx]) / (p[idx] + n[idx]) >= acc_bounds[0]
                problem += (tps[idx] + tns[idx]) / (p[idx] + n[idx]) <= acc_bounds[1]
        elif isinstance(acc_bounds, list):
            for idx in range(n_items):
                problem += (tps[idx] + tns[idx]) / (p[idx] + n[idx]) >= acc_bounds[idx][0]
                problem += (tps[idx] + tns[idx]) / (p[idx] + n[idx]) <= acc_bounds[idx][1]

    return problem

def add_balanced_accuracy_rom(problem, tps, tns, p, n, bacc, eps, bacc_bounds=None):
    n_items = len(tps)
    total_p = np.sum(p)
    total_n = np.sum(n)

    problem += 0.5 * (sum((1.0 / total_p) * (tps[idx]) for idx in range(n_items)) + sum((1.0 / total_n) * (tns[idx]) for idx in range(n_items))) <= bacc + eps
    problem += 0.5 * (sum((1.0 / total_p) * (-1) * (tps[idx]) for idx in range(n_items)) + sum((1.0 / total_n) * (-1) * (tns[idx]) for idx in range(n_items))) <= -(bacc - eps)

    if bacc_bounds is not None:
        if isinstance(bacc_bounds, tuple):
            for idx in range(n_items):
                problem += (tps[idx]*(1.0 / p[idx]) + tns[idx]*(1.0 / n[idx]))/2 >= bacc_bounds[0]
                problem += (tps[idx]*(1.0 / p[idx]) + tns[idx]*(1.0 / n[idx]))/2 <= bacc_bounds[1]
        elif isinstance(bacc_bounds, list):
            for idx in range(n_items):
                problem += (tps[idx]*(1.0 / p[idx]) + tns[idx]*(1.0 / n[idx]))/2 >= bacc_bounds[idx][0]
                problem += (tps[idx]*(1.0 / p[idx]) + tns[idx]*(1.0 / n[idx]))/2 <= bacc_bounds[idx][1]

    return problem

def add_sensitivity_rom(problem, tps, p, sens, eps, sens_bounds=None):
    n_items = len(tps)
    total = np.sum(p)

    problem += sum((1.0 / total) * (tps[idx]) for idx in range(n_items)) <= sens + eps
    problem += sum((1.0 / total) * (-1) * (tps[idx]) for idx in range(n_items)) <= -(sens - eps)

    if sens_bounds is not None:
        if isinstance(sens_bounds, tuple):
            for idx in range(n_items):
                problem += (tps[idx]) * (1.0 / p[idx]) >= sens_bounds[0]
                problem += (tps[idx]) * (1.0 / p[idx]) <= sens_bounds[1]
        elif isinstance(sens_bounds, list):
            for idx in range(n_items):
                problem += (tps[idx]) * (1.0 / p[idx]) >= sens_bounds[idx][0]
                problem += (tps[idx]) * (1.0 / p[idx]) <= sens_bounds[idx][1]

    return problem

def add_specificity_rom(problem, tns, n, spec, eps, spec_bounds=None):
    n_items = len(tns)
    total = np.sum(n)

    problem += sum((1.0 / total) * (tns[idx]) for idx in range(n_items)) <= spec + eps
    problem += sum((1.0 / total) * (-1) * (tns[idx]) for idx in range(n_items)) <= -(spec - eps)

    if spec_bounds is not None:
        if isinstance(spec_bounds, tuple):
            for idx in range(n_items):
                problem += (tns[idx]) * (1.0 / n[idx]) >= spec_bounds[0]
                problem += (tns[idx]) * (1.0 / n[idx]) <= spec_bounds[1]
        elif isinstance(spec_bounds, list):
            for idx in range(n_items):
                problem += (tns[idx]) * (1.0 / n[idx]) >= spec_bounds[idx][0]
                problem += (tns[idx]) * (1.0 / n[idx]) <= spec_bounds[idx][1]

    return problem

def consistency_aggregated_integer_programming_rom(p,
                                                    n,
                                                    scores,
                                                    eps,
                                                    score_bounds=None,
                                                    tptn_bounds=None,
                                                    return_details=False):
    """
    Checking the consistency of aggregated figures, supposing ratio of means calculation

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
    if score_bounds is None:
        score_bounds = {}

    prob = pl.LpProblem("feasibility")

    n_items = len(p)

    tps = [pl.LpVariable(f"tp{i}", 0, p[i], pl.LpInteger) for i in range(n_items)]
    tns = [pl.LpVariable(f"tn{i}", 0, n[i], pl.LpInteger) for i in range(n_items)]

    prob += tps[0]

    for score in scores:
        if score == 'acc':
            prob = add_accuracy_rom(problem=prob,
                                    tps=tps,
                                    tns=tns,
                                    p=p,
                                    n=n,
                                    acc=scores[score],
                                    eps=eps[score] if isinstance(eps, dict) else eps,
                                    acc_bounds=score_bounds.get(score))
        if score == 'bacc':
            prob = add_balanced_accuracy_rom(problem=prob,
                                                tps=tps,
                                                tns=tns,
                                                p=p,
                                                n=n,
                                                bacc=scores[score],
                                                eps=eps[score] if isinstance(eps, dict) else eps,
                                                bacc_bounds=score_bounds.get(score))
        if score == 'sens':
            prob = add_sensitivity_rom(problem=prob,
                                        tps=tps,
                                        p=p,
                                        sens=scores[score],
                                        eps=eps[score] if isinstance(eps, dict) else eps,
                                        sens_bounds=score_bounds.get(score))
        if score == 'spec':
            prob = add_specificity_rom(problem=prob,
                                        tns=tns,
                                        n=n,
                                        spec=scores[score],
                                        eps=eps[score] if isinstance(eps, dict) else eps,
                                        spec_bounds=score_bounds.get(score))

    if tptn_bounds is not None:
        if isinstance(tptn_bounds, dict):
            for idx in range(len(tps)):
                if tptn_bounds.get('tp') is not None:
                    prob += tps[idx] >= tptn_bounds['tp'][0]
                    prob += tps[idx] <= tptn_bounds['tp'][1]
                if tptn_bounds.get('tn') is not None:
                    prob += tns[idx] >= tptn_bounds['tn'][0]
                    prob += tns[idx] <= tptn_bounds['tn'][1]
        elif isinstance(tptn_bounds, list):
            for idx in range(len(tps)):
                if tptn_bounds[idx].get('tp') is not None:
                    prob += tps[idx] >= tptn_bounds[idx]['tp'][0]
                    prob += tps[idx] <= tptn_bounds[idx]['tp'][1]
                if tptn_bounds[idx].get('tn') is not None:
                    prob += tns[idx] >= tptn_bounds[idx]['tn'][0]
                    prob += tns[idx] <= tptn_bounds[idx]['tn'][1]

    if not return_details:
        return prob.solve() == 1

    return prob.solve() == 1, assemble_results(prob)

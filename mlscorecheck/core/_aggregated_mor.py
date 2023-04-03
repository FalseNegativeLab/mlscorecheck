"""
This module implements some core consistency tests for mean-of-rations aggregations
"""

import pulp as pl

__all__ = ['consistency_aggregated_integer_programming_mor']

def consistency_aggregated_integer_programming_mor(p, n, acc, sens, spec, eps):
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
    n_items = p.shape[0]

    prob= pl.LpProblem("feasibility")

    tps= [pl.LpVariable("tp" + str(i), 0, p[i], pl.LpInteger) for i in range(n_items)]
    tns= [pl.LpVariable("tn" + str(i), 0, n[i], pl.LpInteger) for i in range(n_items)]

    prob+= tps[0]

    prob+= sum([(1/n_items)*tps[i]*(1.0/p[i]) for i in range(n_items)]) <= sens + eps
    prob+= sum([(1/n_items)*(-1)*tps[i]*(1.0/p[i]) for i in range(n_items)]) <= -(sens - eps)
    prob+= sum([(1/n_items)*tns[i]*(1.0/n[i]) for i in range(n_items)]) <= spec + eps
    prob+= sum([(1/n_items)*(-1)*tns[i]*(1.0/n[i]) for i in range(n_items)]) <= -(spec - eps)
    prob+= sum([(1/n_items)*(tps[i] + tns[i])*(1.0/(p[i] + n[i])) for i in range(n_items)]) <= acc + eps
    prob+= sum([(1/n_items)*(-1)*(tps[i] + tns[i])*(1.0/(p[i] + n[i])) for i in range(n_items)]) <= -(acc - eps)

    return prob.solve() == 1

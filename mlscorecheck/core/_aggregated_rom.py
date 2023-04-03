"""
This module implements some consistency checks for ratio-of-means aggregations
"""

import numpy as np
import pulp as pl

__all__ = ['consistency_aggregated_integer_programming_rom']

def consistency_aggregated_integer_programming_rom(p, n, acc, sens, spec, eps):
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
    prob= pl.LpProblem("feasibility")

    tps= [pl.LpVariable("tp" + str(i), 0, p[i], pl.LpInteger) for i in range(20)]
    tns= [pl.LpVariable("tn" + str(i), 0, n[i], pl.LpInteger) for i in range(20)]

    prob+= tps[0]

    prob+= sum([(1.0/(np.sum(p)))*tps[i] for i in range(20)]) <= sens + eps
    prob+= sum([(1.0/(np.sum(p)))*(-1)*tps[i] for i in range(20)]) <= -(sens - eps)
    prob+= sum([(1.0/(np.sum(n)))*tns[i] for i in range(20)]) <= spec + eps
    prob+= sum([(1.0/(np.sum(n)))*(-1)*tns[i] for i in range(20)]) <= -(spec - eps)
    prob+= sum([(1.0/(np.sum(p+n)))*(tps[i] + tns[i]) for i in range(20)]) <= acc + eps
    prob+= sum([(1.0/(np.sum(p+n)))*(-1)*(tps[i] + tns[i]) for i in range(20)]) <= -(acc - eps)

    return prob.solve() == 1


"""
This module implements the aggregated AUC related functionalities
"""

import numpy as np

import pulp as pl

from cvxopt import matrix
from cvxopt.solvers import qp, cp, cpl
import cvxopt.solvers as solvers

from ._auc_single import (
    translate_scores,
    prepare_intervals,
    augment_intervals,
    auc_maxa,
    auc_armin,
    acc_min,
    acc_max,
    acc_rmin,
    acc_rmax,
    macc_min
)

__all__ = [
    "auc_min_aggregated",
    "auc_max_aggregated",
    "auc_rmin_aggregated",
    "auc_maxa_evaluate",
    "auc_maxa_solve",
    "auc_maxa_aggregated",
    "auc_amin_aggregated",
    "auc_amax_aggregated",
    "auc_armin_aggregated",
    "auc_from_aggregated",
    "acc_min_aggregated",
    "acc_max_aggregated",
    "acc_rmin_aggregated",
    "acc_rmax_aggregated",
    "acc_from_aggregated",
    "macc_min_aggregated",
    "max_acc_from_aggregated",
    "R",
    "F",
    "perturb_solutions",
    "multi_perturb_solutions",
    "estimate_acc_interval",
    "estimate_tpr_interval",
    "estimate_fpr_interval"
]

def acc_expression(tps, tns, ps, ns):
    """
    The mean accuracy expression

    Args:
        tps (list|np.array): the true positives
        tns (list|np.array): the true negatives
        ps (list|np.array): the positives
        ns (list|np.array): the negatives
    
    Returns:
        float|obj: the average accuracy
    """
    return sum((tps[idx] + tns[idx])* (1.0/(ps[idx] + ns[idx])) 
               for idx in range(len(ps))) * 1.0/(len(ps))

def tpr_expression(tps, ps):
    """
    The mean tpr expression

    Args:
        tps (list|np.array): the true positives
        ps (list|np.array): the positives
    
    Returns:
        float|obj: the average tpr
    """
    return sum([tps[idx] * (1.0/ps[idx]) 
                for idx in range(len(ps))]) * (1.0/len(ps))

def fpr_expression(tns, ns):
    """
    The mean fpr expression

    Args:
        tns (list|np.array): the true negatives
        ns (list|np.array): the negatives
    
    Returns:
        float|obj: the average fpr
    """
    return sum([1 - tns[idx] * (1.0/ns[idx]) 
                for idx in range(len(ns))]) * (1.0/len(ns))

def init_lp_variables(ps, ns):
    """
    Initializes the linear programming variables

    Args:
        ps (list): the postives
        ns (list): the negatives
    
    Returns:
        list, list: the true positive and true negative variables
    """
    tps = [pl.LpVariable(f"tp_{idx}", lowBound=0, upBound=ps[idx], cat=pl.LpInteger) 
            for idx in range(len(ps))]
    tns = [pl.LpVariable(f"tn_{idx}", lowBound=0, upBound=ns[idx], cat=pl.LpInteger) 
            for idx in range(len(ns))]
    
    return tps, tns

def extract_tp_tn_values(problem: pl.LpProblem):
    """
    Extracts the exact tp and tn values from a solved optimization problem

    Args:
        problem (pl.LpProblem): the linear programming problem
    
    Returns:
        np.array, np.array: the true positives and true negatives
    """
    k = int(len(problem.variables()) / 2)

    tp_values = np.zeros(k).astype(float)
    tn_values = np.zeros(k).astype(float)

    for variable in problem.variables():
        ids = variable.name.split('_')
        if ids[0] == 'tp':
            tp_values[int(ids[1])] = variable.varValue
        else:
            tn_values[int(ids[1])] = variable.varValue
    
    return tp_values, tn_values

def estimate_acc_interval(fpr, tpr, ps, ns):
    """
    Estimate the accuracy interval

    Args:
        fpr (float, float): the average fpr value interval
        tpr (float, float): the average tpr value interval
        ps (list): the numbers of positives
        ns (list): the numbers of negatives
    
    Returns:
        float, float: the lower and upper bounds on accuracy
    """

    tps, tns = init_lp_variables(ps, ns)

    problem = pl.LpProblem("acc_minimization")

    problem += tpr_expression(tps, ps) == tpr[0]
    problem += fpr_expression(tns, ns) == fpr[1]

    problem += acc_expression(tps, tns, ps, ns)

    problem.solve(pl.PULP_CBC_CMD(msg=False))

    tp_values, tn_values = extract_tp_tn_values(problem)

    avg_min_acc = acc_expression(tp_values, tn_values, ps, ns)

    problem = pl.LpProblem("acc_maximization")

    problem += tpr_expression(tps, ps) == tpr[1]
    problem += fpr_expression(tns, ns) == fpr[0]

    problem += -acc_expression(tps, tns, ps, ns)

    problem.solve(pl.PULP_CBC_CMD(msg=False))

    tp_values, tn_values = extract_tp_tn_values(problem)

    avg_max_acc = acc_expression(tp_values, tn_values, ps, ns)
    
    return (avg_min_acc, avg_max_acc)

def estimate_tpr_interval(fpr, acc, ps, ns):
    """
    Estimate the tpr interval

    Args:
        fpr (float, float): the average fpr value interval
        acc (float, float): the average acc value interval
        ps (list): the numbers of positives
        ns (list): the numbers of negatives
    
    Returns:
        float, float: the lower and upper bounds on tpr
    """
    tps, tns = init_lp_variables(ps, ns)

    problem = pl.LpProblem("tpr_minimization")

    problem += acc_expression(tps, tns, ps, ns) == acc[0]
    problem += fpr_expression(tns, ns) == fpr[1]

    problem += tpr_expression(tps, ps)

    problem.solve(pl.PULP_CBC_CMD(msg=False))

    tp_values, _ = extract_tp_tn_values(problem)

    avg_min_tpr = tpr_expression(tp_values, ps)

    problem = pl.LpProblem("tpr_maximization")

    problem += acc_expression(tps, tns, ps, ns) == acc[1]
    problem += fpr_expression(tns, ns) == fpr[0]

    problem += -tpr_expression(tps, ps)
    problem.solve(pl.PULP_CBC_CMD(msg=False))

    tp_values, _ = extract_tp_tn_values(problem)

    avg_max_tpr = tpr_expression(tp_values, ps)
    
    return (avg_min_tpr, avg_max_tpr)

def estimate_fpr_interval(tpr, acc, ps, ns):
    """
    Estimate the fpr interval

    Args:
        tpr (float, float): the average tpr value interval
        acc (float, float): the average acc value interval
        ps (list): the numbers of positives
        ns (list): the numbers of negatives
    
    Returns:
        float, float: the lower and upper bounds on fpr
    """
    tps, tns = init_lp_variables(ps, ns)

    problem = pl.LpProblem("fpr_minimization")

    problem += acc_expression(tps, tns, ps, ns) == acc[1]
    problem += tpr_expression(tps, ps) == tpr[0]

    problem += fpr_expression(tns, ns)

    problem.solve(pl.PULP_CBC_CMD(msg=False))

    _, tn_values = extract_tp_tn_values(problem)

    avg_min_fpr = fpr_expression(tn_values, ns)

    problem = pl.LpProblem("fpr_maximization")

    problem += acc_expression(tps, tns, ps, ns) == acc[0]
    problem += tpr_expression(tps, ps) == tpr[1]

    problem += -fpr_expression(tns, ns)
    problem.solve(pl.PULP_CBC_CMD(msg=False))

    _, tn_values = extract_tp_tn_values(problem)

    avg_max_fpr = fpr_expression(tn_values, ns)
    
    return (avg_min_fpr, avg_max_fpr)

def augment_intervals_aggregated(
        intervals: dict,
        ps: np.array,
        ns: np.array
    ):
    """
    Augment the intervals based on the relationship between tpr, fpr and acc

    Args:
        intervals (dict): the intervals of scores
        ps (np.array): the numbers of positive samples
        ns (np.array): the numbers of negative samples

    Returns:
        dict: the intervals augmented
    """
    intervals = {**intervals}

    if "tpr" not in intervals and ("acc" in intervals and "fpr" in intervals):
        intervals["tpr"] = estimate_tpr_interval(intervals['fpr'], intervals['acc'], ps, ns)
    if "fpr" not in intervals and ("acc" in intervals and "tpr" in intervals):
        intervals["fpr"] = estimate_fpr_interval(intervals['tpr'], intervals['acc'], ps, ns)
    if "acc" not in intervals and ("fpr" in intervals and "tpr" in intervals):
        intervals["acc"] = estimate_acc_interval(intervals['fpr'], intervals['tpr'], ps, ns)

    return intervals

def R(
        x: float, 
        k: int, 
        lower: np.array=None, 
        upper: np.array=None
    ) -> np.array:
    """
    The "representative" function

    1 - R(x, k, lower, upper) = F(R(1 - x, k, 1 - F(upper), 1 - F(lower))) 
    
    holds.

    Args:
        x (float): the desired average
        k (int): the number of dimensions
        lower (np.array|None): the lower bounds
        upper (np.array|None): the upper bounds
    
    Returns:
        np.array: the representative vector
    """
    if lower is None:
        lower = np.repeat(0.0, k)
    if upper is None:
        upper = np.repeat(1.0, k)

    print(x, np.sum(lower), np.sum(upper), k, lower, upper)

    x = x * len(lower)
    if np.sum(lower) > x or np.sum(upper) < x:
        raise ValueError('infeasible configuration')

    solution = lower.copy().astype(float)
    x = x - np.sum(lower)

    idx = 0
    while x > 0 and idx < len(lower):
        if upper[idx] - lower[idx] < x:
            solution[idx] = upper[idx]
            x -= upper[idx] - lower[idx]
        else:
            solution[idx] = solution[idx] + x
            x = 0.0
        idx += 1
    
    return np.array(solution).astype(float)

def F(x: np.array) -> np.array:
    """
    The flipping operator

    Args:
        x (np.array): the vector to flip
    
    Returns:
        np.array: the flipped vector
    """
    return x[::-1]

def auc_min_aggregated(
        fpr: float, 
        tpr: float, 
        k: int, 
        return_solutions: bool=False
    ) -> float:
    """
    The average area under the minimum curves at the average fpr, tpr

    Args:
        fpr (list): upper bound on average false positive rate
        tpr (list): lower bound on average true positive rate
        return_solutions (bool): whether to return the solutions for the 
        underlying curves

    Returns:
        float | (float, np.array, np.array, np.array, np.array): the area 
        or the area, the solutions and the bounds
    """

    RL_avg_tpr = R(tpr, k)
    RL_avg_fpr = R(fpr, k)

    results = float(np.mean([a * b for a, b in zip(RL_avg_tpr, 1.0 - RL_avg_fpr)]))

    if return_solutions:
        results = results, (RL_avg_fpr, RL_avg_tpr, np.repeat(0.0, k), np.repeat(1.0, k))

    return results

def auc_max_aggregated(
        fpr: float, 
        tpr: float, 
        k: int, 
        return_solutions: bool=False
    ) -> float:
    """
    The average area under the maximum curves at the average fpr, tpr

    tpr >= fpr always holds for the solutions of the input tpr >= fpr

    Args:
        fpr (list): lower bound on average false positive rate
        tpr (list): upper bound on average true positive rate
        return_solutions (bool): whether to return the solutions for the 
        underlying curves

    Returns:
        float | (float, np.array, np.array): the area or the area and the 
        solutions
    """

    RL_avg_tpr = R(tpr, k)
    RL_avg_fpr = R(fpr, k)

    results = float(1 - np.mean([a * b for a, b in zip(1 - RL_avg_tpr, RL_avg_fpr)]))

    if return_solutions:
        results = results, (RL_avg_fpr, RL_avg_tpr, np.repeat(0.0, k), np.repeat(1.0, k))

    return results


def auc_rmin_aggregated(
        fpr: float, 
        tpr: float, 
        k: int, 
        return_solutions: bool=False
    ) -> float:
    """
    The average area under the regulated minimum curves at the average fpr, tpr

    Args:
        fpr (list): lower bound on average false positive rate
        tpr (list): upper bound on average true positive rate
        return_solutions (bool): whether to return the solutions for the 
        underlying curves

    Returns:
        float | (float, np.array, np.array): the area or the area and the 
        solutions
    """
    if tpr < fpr:
            raise ValueError(
                'sens >= 1 - spec does not hold'
            )
    
    results = 0.5 + (fpr - tpr) ** 2 / 2.0

    if return_solutions:
        results = results, (np.repeat(fpr, k), np.repeat(tpr, k), np.repeat(0.0, k), np.repeat(1.0, k))

    return results


def auc_maxa_evaluate(
        ps: np.array, 
        ns: np.array, 
        accs: np.array
    ) -> float:
    """
    Evaluates the maxa solutions

    Args:
        ps (np.array): the numbers of positives in the evaluation sets
        ns (np.array): the numbers of negatives in the evaluation sets
        accs (np.array): the accuracies
    
    Returns:
        float | (float, np.array): the area or the area and the solutions
    """
    return np.mean([auc_maxa(acc, p, n) for acc, p, n in zip(accs, ps, ns)])


def auc_maxa_solve(
        ps: np.array, 
        ns: np.array, 
        avg_acc: float, 
        return_solutions: bool=False
    ) -> float:
    """
    Solves the maxa problem

    Args:
        ps (np.array): the numbers of positives in the evaluation sets
        ns (np.array): the numbers of negatives in the evaluation sets
        avg_acc (float): the average accuracy (upper bound)
        return_solutions (bool): whether to return the solutions for the 
        underlying curves
    
    Returns:
        float | (float, np.array, np.array, np.array): the area or the 
        area, the solutions, and the lower and upper bounds
    """
    ps = np.array(ps)
    ns = np.array(ns)

    k = ps.shape[0]
    
    w = (ps + ns)**2/(2*ps*ns)
    Q = 2.0*np.diag(w)
    q = -2.0*w
    A = np.repeat(1.0/k, k).reshape(-1, 1).T
    b = np.array([avg_acc])
    G = np.vstack([np.eye(k), -np.eye(k)]).astype(float)
    lower = np.array([max(p, n)/(p + n) for p, n in zip(ps, ns)])
    h = np.hstack([np.repeat(1.0, k), -lower])

    Q = matrix(Q)
    q = matrix(q)
    G = matrix(G)
    h = matrix(h)
    A = matrix(A)
    b = matrix(b)

    res = qp(P=Q, q=q, G=G, h=h, A=A, b=b)

    result = np.array(res['x'])[:, 0]

    results = float(auc_maxa_evaluate(ps, ns, result))

    if return_solutions:
        results = results, (result.astype(float), lower, np.repeat(1.0, k))

    return results


def auc_maxa_aggregated(
        acc: float, 
        ps: np.array, 
        ns: np.array, 
        return_solutions: bool=False
    ):
    """
    Solves the maxa problem

    Args:
        ps (np.array): the numbers of positives in the evaluation sets
        ns (np.array): the numbers of negatives in the evaluation sets
        avg_acc (float): the average accuracy (upper bound)
        return_solutions (bool): whether to return the solutions for the 
        underlying curves
    
    Returns:
        float | (float, np.array): the area or the area and the solutions
    """

    if acc < np.mean([max(p, n)/(p + n) for p, n in zip(ps, ns)]):
        raise ValueError("accuracy too small")

    return auc_maxa_solve(ps, ns, acc, return_solutions)


def perturb_solutions(
        values: np.array, 
        lower_bounds: np.array, 
        upper_bounds: np.array, 
        random_state: int=None
    ) -> np.array:
    """
    Applies a perturbation to a solution, by keeping the lower and upper bound and the sum

    Args:
        values (np.array): the values to perturb
        lower_bounds (np.array): the lower bounds
        upper_bounds (np.array): the upper bounds
        random_state (int|None): the random seed to use
    
    Returns:
        np.array: the perturbed values
    """
    random_state = np.random.RandomState(random_state)
    greater = np.where(values > lower_bounds)[0]
    lower = np.where(values < upper_bounds)[0]
    
    greater = random_state.choice(greater)
    lower = random_state.choice(lower)

    diff = min(values[greater] - lower_bounds[greater], upper_bounds[lower] - values[lower])
    step = random_state.random_sample() * diff

    values = values.copy()
    values[greater] -= step
    values[lower] += step

    return values


def multi_perturb_solutions(
        n_perturbations: int,
        values: np.array, 
        lower_bounds: np.array, 
        upper_bounds: np.array, 
        random_state: int=None
    ) -> np.array:
    """
    Applies a multiple perturbations to a solution vector,
    keeping its average

    Args:
        n_perturbations (int): the number of perturbations
        values (np.array): the values to perturb
        lower_bounds (np.array): the lower bounds
        upper_bounds (np.array): the upper bounds
        random_state (int|None): the random seed to use
    
    Returns:
        np.array: the perturbed values
    """
    for _ in range(n_perturbations):
        values = perturb_solutions(
            values, 
            lower_bounds, 
            upper_bounds, 
            random_state
        )

    return values


def auc_amin_aggregated(
        acc: float, 
        ps: np.array, 
        ns: np.array, 
        return_solutions: bool=False
    ) -> float:
    """
    The minimum average AUC based on average accuracy

    Args:
        acc (float): the average accuracy
        ps (np.array): the number of positive samples
        ns (np.array): the number of negative samples
        return_solutions (bool): whether to return the solutions to the
                                underlying optimization problem
    
    Returns:
        float | (float, (np.array, np.array, np.array, np.array, np.array)):
            the AUC or the AUC and the following details: the acc values for 
            the individual underlying curves, the ps, ns (in proper order), the
            lower bounds and the upper bounds
    """
    ps = np.array(ps)
    ns = np.array(ns)

    k = len(ps)

    weights = np.repeat(1.0, k) + np.array([float(max(p, n))/min(p, n) for p, n in zip(ps, ns)])
    thresholds = np.array([max(p, n) for p, n in zip(ps, ns)]) / (ps + ns)

    sorting = np.argsort(weights)
    ps = ps[sorting]
    ns = ns[sorting]
    weights = weights[sorting]

    if acc < np.mean(thresholds):
        auc = 0.0
        accs = np.repeat(acc, k)
        lower = np.repeat(0.0, k)
        upper = thresholds
    else:
        accs = R(acc, k, lower=thresholds)
        auc = float(np.sum(accs * (weights) - weights + 1.0)/k)
        lower = thresholds
        upper = np.repeat(1.0, k)

    results = auc

    if return_solutions:
        results = results, (accs, ps, ns, lower, upper)
    
    return results


def auc_amax_aggregated(
        acc: float, 
        ps: np.array, 
        ns: np.array, 
        return_solutions: bool=False
    ) -> float:
    """
    The maximum average AUC based on average accuracy

    Args:
        acc (float): the average accuracy
        ps (np.array): the number of positive samples
        ns (np.array): the number of negative samples
        return_solutions (bool): whether to return the solutions to the
                                underlying optimization problem
    
    Returns:
        float | (float, (np.array, np.array, np.array, np.array, np.array)):
            the AUC or the AUC and the following details: the acc values for 
            the individual underlying curves, the ps, ns (in proper order), the
            lower bounds and the upper bounds
    """
    ps = np.array(ps)
    ns = np.array(ns)

    k = len(ps)

    weights = np.repeat(1.0, k) + np.array([float(max(p, n))/min(p, n) for p, n in zip(ps, ns)])
    thresholds = np.array([min(p, n) for p, n in zip(ps, ns)]) / (ps + ns)

    sorting = np.argsort(weights)
    ps = ps[sorting]
    ns = ns[sorting]
    weights = weights[sorting]

    if acc > np.mean(thresholds):
        auc = 1.0
        accs = np.repeat(acc, k)
        upper = np.repeat(1.0, k)
        lower = thresholds
    else:
        accs = R(acc, k, upper=thresholds)
        auc = float(np.sum(accs * weights[::-1])/k)
        upper = thresholds
        lower = np.repeat(0.0, k)

    results = auc

    if return_solutions:
        results = results, (accs, ps, ns, lower, upper)
    
    return results


def check_cvxopt(results, message):
    """
    Checking the cvxopt results

    Args:
        results (dict): the output of cvxopt
        message (str): the additional message

    Raises:
        ValueError: when the solution is not optimal
    """
    if results['status'] != 'optimal':
        raise ValueError('no optimal solution found for the configuration ' +
                         f'({message})')

def auc_armin_solve(
        ps: np.array, 
        ns: np.array, 
        avg_acc: float, 
        return_bounds: bool=False
    ):
    """
    Solves the armin quadratic optimization problem

    Args:
        ps (np.array): the number of positive samples
        ns (np.array): the number of negative samples
        avg_acc (float): the average accuracy
        return_solutions (bool): whether to return the solutions to the
                                underlying optimization problem
    
    Returns:
        float | (float, np.array):
            the AUC or the AUC with the lower bounds
    """
    k = ps.shape[0]
    w = (np.array([max(p, n)/min(p, n) for p, n in zip(ps, ns)]))

    p = (-2*w - 2*w**2)

    # note the multiplier of 2 in Q to get the weighting of the quadratic
    # and linear term correct in the cxopt formalism
    Q = 2*np.diag(1 + w**2 + 2*w)
    A = np.repeat(1.0/k, k).reshape(-1, 1).T
    b = np.array([avg_acc])
    G = np.vstack([np.eye(k), -np.eye(k)]).astype(float)
    lower = np.array([max(p, n)/(p + n) for p, n in zip(ps, ns)])
    h = np.hstack([np.repeat(1.0, k), -lower])
    
    Q = matrix(Q)
    p = matrix(p)
    G = matrix(G)
    h = matrix(h)
    A = matrix(A)
    b = matrix(b)

    res = qp(P=Q, q=p, G=G, h=h, A=A, b=b)

    actual = solvers.options.get('show_progress', None)
    solvers.options['show_progress'] = False

    check_cvxopt(res, 'auc_armin_aggregated')

    solvers.options['show_progress'] = actual

    results = np.array(res['x'])[:, 0]

    if return_bounds:
        results = results, lower

    return results

def auc_armin_aggregated(
        acc: float, 
        ps: np.array, 
        ns: np.array, 
        return_solutions: bool=False
    ):
    """
    The regulated minimum based AUC based on average accuracy

    Args:
        acc (float): the average accuracy
        ps (np.array): the number of positive samples
        ns (np.array): the number of negative samples
        return_solutions (bool): whether to return the solutions to the
                                underlying optimization problem
    
    Returns:
        float | (float, (np.array, np.array, np.array, np.array, np.array)):
            the AUC or the AUC and the following details: the acc values for 
            the individual underlying curves, the ps, ns (in proper order), the
            lower bounds and the upper bounds
    """
    ps = np.array(ps)
    ns = np.array(ns)

    lower = np.array([min(p, n)/(p + n) for p, n in zip(ps, ns)])
    upper = np.array([max(p, n)/(p + n) for p, n in zip(ps, ns)])

    if acc < np.mean(lower):
        raise ValueError("acc too small for the configuration "
                          "(auc_armin_aggregated)")
    elif acc >= np.mean(lower) and acc <= np.mean(upper):
        auc = 0.5
        results = auc
        if return_solutions:
            results = results, ((lower + upper)/2, ps, ns, lower, upper)
        return results

    accs, lower = auc_armin_solve(ps, ns, acc, return_bounds=True)

    auc = float(np.mean([auc_armin(acc, p, n) for acc, p, n in zip(accs, ps, ns)]))

    results = auc

    if return_solutions:
        results = results, (accs, ps, ns, lower, np.repeat(1.0, len(ps)))

    return results

def acc_min_aggregated(
        auc: float, 
        ps: np.array, 
        ns: np.array, 
        return_solutions: bool=False
    ):
    """
    The minimum based accuracy

    Args:
        auc (float): the average accuracy
        ps (np.array): the number of positive samples
        ns (np.array): the number of negative samples
        return_solutions (bool): whether to return the solutions to the
                                underlying optimization problem
    
    Returns:
        float | (float, (np.array, np.array, np.array, np.array, np.array)):
            the acc or the acc and the following details: the AUC values for 
            the individual underlying curves, the ps, ns (in proper order), the
            lower bounds and the upper bounds
    """
    ps = np.array(ps)
    ns = np.array(ns)

    k = len(ps)

    w = np.array([min(p, n)/(p + n) for p, n in zip(ps, ns)])
    sorting = np.argsort(w)

    ps = ps[sorting]
    ns = ns[sorting]
    w = w[sorting]

    aucs = R(auc, k)

    acc = float(np.mean([acc_min(auc, p, n) for auc, p, n in zip(aucs, ps, ns)]))

    results = acc

    if return_solutions:
        results = results, (aucs, ps, ns, np.repeat(0.0, k), np.repeat(1.0, k))

    return results

def acc_max_aggregated(
        auc: float, 
        ps: np.array, 
        ns: np.array, 
        return_solutions: bool=False
    ):
    """
    The maximum based accuracy

    Args:
        auc (float): the average accuracy
        ps (np.array): the number of positive samples
        ns (np.array): the number of negative samples
        return_solutions (bool): whether to return the solutions to the
                                underlying optimization problem
    
    Returns:
        float | (float, (np.array, np.array, np.array, np.array, np.array)):
            the acc or the acc and the following details: the AUC values for 
            the individual underlying curves, the ps, ns (in proper order), the
            lower bounds and the upper bounds
    """
    ps = np.array(ps)
    ns = np.array(ns)

    k = len(ps)

    w = np.array([min(p, n)/(p + n) for p, n in zip(ps, ns)])
    sorting = np.argsort(w)[::-1]

    ps = ps[sorting]
    ns = ns[sorting]
    w = w[sorting]

    aucs = R(auc, k)

    acc = float(np.mean([acc_max(auc, p, n) for auc, p, n in zip(aucs, ps, ns)]))

    results = acc

    if return_solutions:
        results = results, (aucs, ps, ns, np.repeat(0.0, k), np.repeat(1.0, k))

    return results

def acc_rmin_aggregated(
        auc: float, 
        ps: np.array, 
        ns: np.array, 
        return_solutions: bool=False
    ):
    """
    The regulated minimum based accuracy

    This is independent from the AUC, as when the ROC curve is
    expected to run above the random classification line, the
    minimum accuracy along the curve is either p/(p + n) or
    n / (p + n).

    Args:
        auc (float): the average accuracy
        ps (np.array): the number of positive samples
        ns (np.array): the number of negative samples
        return_solutions (bool): whether to return the solutions to the
                                underlying optimization problem
    
    Returns:
        float | (float, (np.array, np.array, np.array)):
            the acc or the acc and the following details: the AUC values for 
            the individual underlying curves, the lower bounds and the upper bounds
    """
    _ = auc

    ps = np.array(ps)
    ns = np.array(ns)

    acc = np.mean([min(p, n)/(p + n) for p, n in zip(ps, ns)])
    
    results = acc

    if return_solutions:
        results = acc, (np.repeat(0.7, len(ps)), np.repeat(0.5, len(ps)), np.repeat(1.0, len(ps)))
    
    return results

class F_acc_rmax:
    """
    Implements the convex programming objective for the regulated
    accuracy maximization.
    """
    def __init__(
            self, 
            ps: np.array, 
            ns: np.array, 
            avg_auc: float
        ):
        """
        The constructor of the object

        Args:
            ps (np.array): the number of positive samples
            ns (np.array): the number of negative samples
            avg_auc (float): the average AUC (upper bound)
        """
        self.ps = ps
        self.ns = ns
        self.avg_auc = avg_auc
        self.k = len(ps)
        self.maxs = np.array([max(p, n) for p, n in zip(ps, ns)])
        self.mins = np.array([min(p, n) for p, n in zip(ps, ns)])
        self.weights = self.mins / (ps + ns)

    def __call__(
            self, 
            x: matrix=None, 
            z: matrix=None):
        """
        The call method according to the specification in cvxopt

        Evaluates the function

        f(auc) = - sum limits_{i=1}^{k} sqrt{2auc_i - 1}*w_i,

        where w_i = min(p_i, n_i)/(p_i + n_i).

        The function originates from the objective

        f_0(auc) = dfrac{1}{k} sum limits_{i=1}^{k}acc_rmax(auc_i, p_i, n_i),

        by removing the multiplier 1/k and the constants parts.

        Since the function is to be maximized, a negative sign is added to turn
        it into minimization.

        Also:
            auc = sp.Symbol('auc')
            f = sp.sqrt(2*auc - 1)
            sp.diff(f, auc), sp.diff(sp.diff(f, auc), auc)

            (1/sqrt(2*auc - 1), -1/(2*auc - 1)**(3/2))

        Args:
            x (cvxopt.matrix | None): a vector in the parameter space
            z (cvxopt.matrix | None): a weight vector
        
        Returns:
            (n, matrix): the number of non-linear constraints and a feasible
                point when x is None and z is None
            (matrix, matrix): the objective value at x, and the gradient
                at x when x is not None but z is None
            (matrix, matrix, matrx): the objective value at x, the gradient
                at x and the weighted sum of the Hessian of the objective and
                all non-linear constraints with the weights z if z is not None
        """
        if x is None and z is None:
            # The number of non-linear constraints and one point of
            # the feasible region
            return (0, matrix(1.0, (self.k, 1)))
        
        if x is not None:
            # the function to be evaluated
            f = matrix(-np.sum(np.sqrt(2*x-1)*self.weights.reshape(-1, 1)))
            # the gradient
            Df = matrix(-1.0/np.sqrt(2*x-1)*self.weights.reshape(-1, 1)).T
        
        if z is None:
            return (f, Df)
        
        # the Hessian
        hess = np.diag((2*np.array(x)[:, 0] - 1.0)**(-3/2)*self.weights)
        
        hess = matrix(z[0]*hess)

        return (f, Df, hess)

def acc_rmax_evaluate(
        ps: np.array, 
        ns: np.array, 
        aucs: np.array
    ):
    """
    Evaluates a particular configuration of rmax parameters

    Args:
        ps (np.array): the number of positive samples
        ns (np.array): the number of negative samples
        aucs (np.array): the AUC parameters
    
    Returns:
        float: the mean accuracy
    """

    maxs = np.array([max(p, n) for p, n in zip(ps, ns)])
    mins = np.array([min(p, n) for p, n in zip(ps, ns)])

    return np.mean((maxs + mins * np.sqrt(2*aucs - 1)) / (ps + ns))

def acc_rmax_solve(
        ps: np.array, 
        ns: np.array, 
        avg_auc: float, 
        return_solutions: bool=False
    ):
    """
    Solves the regulated maximium curves problem

    Args:
        ps (np.array): the number of positive samples
        ns (np.array): the number of negative samples
        avg_auc (np.array): the average AUC
        return_solutions (bool): whether to return the solutions and
                                further details

    Returns:
        float | (float, np.array, np.array, np.array): the mean accuracy, 
        or the mean accuracy, the auc parameters, lower bounds and upper 
        bounds
    """
    F = F_acc_rmax(ps, ns, avg_auc)

    k = ps.shape[0]

    lower_bounds = np.repeat(0.5, k)

    A = np.repeat(1.0/k, k).reshape(-1, 1).T
    b = np.array([avg_auc])
    G = np.vstack([np.eye(k), -np.eye(k)]).astype(float)
    h = np.hstack([np.repeat(1.0, k), -lower_bounds])
    
    G = matrix(G)
    h = matrix(h)
    A = matrix(A)
    b = matrix(b)

    res = cp(F, G, h, A=A, b=b)

    actual = solvers.options.get('show_progress', None)
    solvers.options['show_progress'] = False

    check_cvxopt(res, 'acc_rmax_aggregated')

    solvers.options['show_progress'] = actual

    aucs = np.array(res['x'])[:, 0]

    acc = acc_rmax_evaluate(ps, ns, aucs)

    results = acc

    if return_solutions:
        results = results, (aucs, ps, ns, lower_bounds, np.repeat(1.0, k))

    return results

def acc_rmax_aggregated(
        auc: float, 
        ps: np.array, 
        ns: np.array, 
        return_solutions: bool=False
    ):
    """
    The regulated maximum based accuracy

    Args:
        auc (float): the average accuracy
        ps (np.array): the number of positive samples
        ns (np.array): the number of negative samples
        return_solutions (bool): whether to return the solutions to the
                                underlying optimization problem
    
    Returns:
        float | (float, (np.array, np.array, np.array)):
            the acc or the acc and the following details: the AUC values for 
            the individual underlying curves, the lower bounds and the upper bounds
    """

    if auc < 0.5:
        raise ValueError("auc too small (acc_rmax_aggregated)")

    ps = np.array(ps)
    ns = np.array(ns)

    return acc_rmax_solve(ps, ns, auc, return_solutions)

class F_macc_min:
    """
    Implements the convex programming objective for the maximum accuracy
    minimization.
    """
    def __init__(
            self, 
            ps: np.array, 
            ns: np.array, 
            avg_auc: float
        ):
        """
        The constructor of the object

        Args:
            ps (np.array): the number of positive samples
            ns (np.array): the number of negative samples
            avg_auc (float): the average AUC (upper bound)
        """
        self.ps = ps
        self.ns = ns
        self.avg_auc = avg_auc
        self.k = len(ps)
        self.maxs = np.array([max(p, n) for p, n in zip(ps, ns)])
        self.mins = np.array([min(p, n) for p, n in zip(ps, ns)])
        self.weights = np.sqrt(2*(ps*ns))/(ps + ns)
        self.lower_bounds = 1.0 - np.array([min(p, n)/(2*max(p, n)) for p, n in zip(ps, ns)])

    def __call__(
            self, 
            x: matrix=None, 
            z: matrix=None
        ):
        """
        The call method according to the specification in cvxopt

        Evaluates the function originating from the objective

        f_0(auc) = dfrac{1}{k} sum limits_{i=1}^{k}macc_min(auc_i, p_i, n_i),

        f_0(auc) = 1/k sum (1 - sqrt(2*p_i*n_i*(1 - auc_i))/(p_i + n_i))

        by removing the multiplier 1/k and the constants parts.

        Also:
            auc = sp.Symbol('auc')
            f = sp.sqrt(1 - auc)
            sp.diff(f, auc), sp.diff(sp.diff(f, auc), auc)

            (-1/(2*sqrt(1 - auc)), -1/(4*(1 - auc)**(3/2)))
        
        Args:
            x (cvxopt.matrix | None): a vector in the parameter space
            z (cvxopt.matrix | None): a weight vector
        
        Returns:
            (n, matrix): the number of non-linear constraints and a feasible
                point when x is None and z is None
            (matrix, matrix): the objective value at x, and the gradient
                at x when x is not None but z is None
            (matrix, matrix, matrx): the objective value at x, the gradient
                at x and the weighted sum of the Hessian of the objective and
                all non-linear constraints with the weights z if z is not None
        
        TODO: something with all variables being 1
        """
        if x is None and z is None:
            return (0, matrix(self.lower_bounds, (self.k, 1)))
            #return (0, matrix(np.repeat(1.0, self.k), (self.k, 1)))
        
        if x is not None:
            f = matrix(-np.sum(np.sqrt(1 - x)*self.weights.reshape(-1, 1)))
            Df = matrix(1.0/(2*np.sqrt(1 - x))*self.weights.reshape(-1, 1)).T
        
        if z is None:
            return (f, Df)
        
        hess = np.diag(1.0/(4*np.array(1 - x)[:, 0]**(3/2))*self.weights)
        
        hess = matrix(z[0]*hess)

        return (f, Df, hess)

def macc_min_evaluate(
        ps: np.array, 
        ns: np.array, 
        aucs: np.array
    ):
    """
    Evaluates a particular macc_min configuration

    Args:
        ps (np.array): the number of positive samples
        ns (np.array): the number of negative samples
        aucs (np.array): the AUCs
    
    Returns:
        float: the average accuracy
    """
    return np.mean([macc_min(auc, p, n) for auc, p, n in zip(aucs, ps, ns)])

def macc_min_solve(
        ps: np.array, 
        ns: np.array, 
        avg_auc: float, 
        return_solutions: bool=False
    ):
    """
    Solves the maximum accuracy minimum curves problem

    Args:
        ps (np.array): the number of positive samples
        ns (np.array): the number of negative samples
        avg_auc (np.array): the average AUC
        return_solutions (bool): whether to return the solutions and
                                further details

    Returns:
        float | (float, np.array, np.array, np.array): the mean accuracy, 
        or the mean accuracy, the auc parameters, lower bounds and upper 
        bounds
    """
    F = F_macc_min(ps, ns, avg_auc)

    k = ps.shape[0]

    lower_bounds = 1.0 - np.array([min(p, n)/(2*max(p, n)) for p, n in zip(ps, ns)])
    upper_bounds = np.repeat(1.0 - np.min(1/((ps+1)*(ns+1))), k)

    A = np.repeat(1.0/k, k).reshape(-1, 1).T
    b = np.array([avg_auc])
    G = np.vstack([np.eye(k), -np.eye(k)]).astype(float)
    h = np.hstack([upper_bounds, -lower_bounds])
    
    G = matrix(G)
    h = matrix(h)
    A = matrix(A)
    b = matrix(b)

    res = cp(F, G, h, A=A, b=b)

    actual = solvers.options.get('show_progress', None)
    solvers.options['show_progress'] = False

    check_cvxopt(res, 'macc_min_aggregated')

    solvers.options['show_progress'] = actual

    aucs = (np.array(res['x'])[:, 0])

    acc = macc_min_evaluate(ps, ns, aucs)

    results = acc

    if return_solutions:
        results = acc, (aucs, lower_bounds, np.repeat(1.0, k))
    
    return results

def macc_min_aggregated(
        auc: float, 
        ps: np.array, 
        ns: np.array, 
        return_solutions: bool=False
    ):
    """
    The minimum for the maximum average accuracy from average AUC

    Args:
        auc (float): the average accuracy
        ps (np.array): the number of positive samples
        ns (np.array): the number of negative samples
        return_solutions (bool): whether to return the solutions to the
                                underlying optimization problem
    
    Returns:
        float | (float, (np.array, np.array, np.array)):
            the acc or the acc and the following details: the AUC values for 
            the individual underlying curves, the lower bounds and the upper bounds
    """
    ps = np.array(ps)
    ns = np.array(ns)

    lower_bounds = 1.0 - np.array([min(p, n)/(2*max(p, n)) for p, n in zip(ps, ns)])

    if auc < np.mean(lower_bounds):
        raise ValueError("auc too small (macc_min_aggregated)")
    
    if auc == 1.0:
        # the gradient would go to infinity in this case
        results = 1.0

        if return_solutions:
            results = results, (np.repeat(1.0, len(ps)), lower_bounds, np.repeat(1.0, len(ps)))

    return macc_min_solve(ps, ns, auc, return_solutions)

def check_applicability_aggregated(
        intervals: dict,
        lower: str,
        upper: str,
        ps: int,
        ns: int
    ):
    """
    Checks the applicability of the methods

    Args:
        intervals (dict): the score intervals
        lower (str): the lower bound method
        upper (str): the upper bound method
        p (int): the number of positive samples
        n (int): the number of negative samples

    Raises:
        ValueError: when the methods are not applicable with the
                    specified scores
    """
    if lower in ['min', 'rmin'] or upper in ['max']:
        if "fpr" not in intervals or "tpr" not in intervals:
            raise ValueError("fpr, tpr or their complements must be specified")
    if lower in ['amin', 'armin'] or upper in ['amax', 'maxa']:
        if ps is None or ns is None:
            raise ValueError("p and n must be specified")
    if lower in ['amin', 'armin'] or upper in ['amax', 'maxa']:
        if "acc" not in intervals:
            raise ValueError("acc must be specified")


def auc_from_aggregated(
    *,
    scores: dict,
    eps: float,
    k: int,
    ps: np.array = None,
    ns: np.array = None,
    lower: str = "min",
    upper: str = "max"
) -> tuple:
    """
    This function applies the estimation schemes to estimate AUC from scores

    Args:
        scores (dict): the reported scores
        eps (float): the numerical uncertainty
        k (int): the number of evaluation sets
        ps (int): the numbers of positive samples
        ns (int): the numbers of negative samples
        lower (str): ('min'/'rmin'/'amin'/'armin') - the type of
                        estimation for the lower bound
        upper (str): ('max'/'maxa'/'amax') - the type of estimation for
                        the upper bound

    Returns:
        tuple(float, float): the interval for the AUC
    """

    scores = translate_scores(scores)
    intervals = prepare_intervals(scores, eps)

    if ps is not None and ns is not None:
        intervals = augment_intervals_aggregated(intervals, ps, ns)

    check_applicability_aggregated(intervals, lower, upper, ps, ns)


    if lower == 'min':
        lower0 = auc_min_aggregated(intervals["fpr"][1], intervals["tpr"][0], k)
    elif lower == 'rmin':
        lower0 = auc_rmin_aggregated(intervals["fpr"][0], intervals["tpr"][1], k)
    elif lower == 'amin':
        lower0 = auc_amin_aggregated(intervals["acc"][0], ps, ns)
    elif lower == 'armin':
        lower0 = auc_armin_aggregated(intervals["acc"][0], ps, ns)
    else:
        raise ValueError(f"unsupported lower bound {lower}")

    if upper == 'max':
        upper0 = auc_max_aggregated(intervals["fpr"][0], intervals["tpr"][1], k)
    elif upper == 'amax':
        print(intervals['acc'][1], ps, ns)
        upper0 = auc_amax_aggregated(intervals["acc"][1], ps, ns)
    elif upper == 'maxa':
        upper0 = auc_maxa_aggregated(intervals["acc"][1], ps, ns)
    else:
        raise ValueError(f"unsupported upper bound {upper}")

    return (lower0, upper0)


def acc_from_aggregated(
    *,
    scores: dict,
    eps: float,
    ps: int,
    ns: int,
    lower: str = "min",
    upper: str = "max"
) -> tuple:
    """
    This function applies the estimation schemes to estimate acc from scores

    Args:
        scores (dict): the reported scores
        eps (float): the numerical uncertainty
        p (int): the number of positive samples
        n (int): the number of negative samples
        lower (str): 'min'/'rmin'
        upper (str): 'max'/'rmax' - the type of upper bound

    Returns:
        tuple(float, float): the interval for the accuracy
    """

    intervals = prepare_intervals(scores, eps)

    if "auc" not in intervals:
        raise ValueError("auc must be specified")

    lower0 = None
    upper0 = None

    if lower == 'min':
        lower0 = acc_min_aggregated(intervals['auc'][0], ps, ns)
    elif lower == 'rmin':
        lower0 = acc_rmin_aggregated(intervals['auc'][0], ps, ns)
    else:
        raise ValueError(f"unsupported lower bound {lower}")

    if upper == 'max':
        upper0 = acc_max_aggregated(intervals['auc'][1], ps, ns)
    elif upper == 'rmax':
        upper0 = acc_rmax_aggregated(intervals['auc'][1], ps, ns)
    else:
        raise ValueError(f"unsupported upper bound {upper}")

    return (lower0, upper0)


def max_acc_from_aggregated(
    *,
    scores: dict,
    eps: float,
    ps: int,
    ns: int,
    lower: str = "min",
    upper: str = "max"
) -> tuple:
    """
    This function applies the estimation schemes to estimate maximum accuracy
    from scores

    Args:
        scores (dict): the reported scores
        eps (float): the numerical uncertainty
        p (int): the number of positive samples
        n (int): the number of negative samples
        lower (str): 'min'
        upper (str): 'max'/'rmax' - the type of upper bound

    Returns:
        tuple(float, float): the interval for the maximum accuracy
    """

    intervals = prepare_intervals(scores, eps)

    if "auc" not in intervals:
        raise ValueError("auc must be specified")

    lower0 = None
    upper0 = None

    if lower == 'min':
        lower0 = macc_min_aggregated(intervals['auc'][0], ps, ns)
    else:
        raise ValueError(f"unsupported lower bound {lower}")

    if upper == 'max':
        upper0 = acc_max_aggregated(intervals['auc'][1], ps, ns)
    elif upper == 'rmax':
        upper0 = acc_rmax_aggregated(intervals['auc'][1], ps, ns)
    else:
        raise ValueError(f"unsupported upper bound {upper}")

    return (lower0, upper0)



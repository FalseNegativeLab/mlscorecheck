"""
This module implements the aggregated AUC related functionalities
"""

import numpy as np

from cvxopt import matrix
from cvxopt.solvers import qp, cp, cpl

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
    "R",
    "F",
    "perturb_solutions",
    "multi_perturb_solutions"
]

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
        results = results, (RL_avg_fpr, RL_avg_tpr)

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
                'sens >= 1 - spec does not hold for "\
                            "the corrected minimum curve'
            )
    
    results = 0.5 + (fpr - tpr) ** 2 / 2.0

    if return_solutions:
        results = results, (np.repeat(fpr, k), np.repeat(tpr, k))

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
    #q = (ps + ns)**2/(2*ps*ns)/k*2
    #q = np.repeat(0.0, k)
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

    print(res['status'])

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

    accs = R(acc, k, lower=thresholds)
    auc = float(np.sum(accs * (weights) - weights + 1.0)/k)

    results = auc

    if return_solutions:
        results = results, (accs, ps, ns, thresholds, np.repeat(1.0, k))
    
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

    accs = R(acc, k, upper=thresholds)
    auc = float(np.sum(accs * weights[::-1])/k)

    results = auc

    if return_solutions:
        results = results, (accs, ps, ns, np.repeat(0.0, k), thresholds)
    
    return results

def check_cvxopt(results, message):
    if results['status'] != 'optimal':
        raise ValueError('no optimal solution found for the configuration ',
                         f'({message})')

def auc_armin_solve(ps, ns, avg_acc, return_bounds=False):
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

    check_cvxopt(res, 'auc_armin_aggregated')

    results = np.array(res['x'])[:, 0]

    if return_bounds:
        results = results, lower

    return results

def auc_armin_aggregated(acc, ps, ns, return_solutions=False):
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

    accs, lower = auc_armin_solve(ps, ns, acc, return_bounds=True)

    auc = float(np.mean([auc_armin(acc, p, n) for acc, p, n in zip(accs, ps, ns)]))

    results = auc

    if return_solutions:
        results = results, (accs, ps, ns, lower, np.repeat(1.0, len(ps)))

    return results

def acc_min_aggregated(auc, ps, ns, return_solutions=False):
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

def acc_max_aggregated(auc, ps, ns, return_solutions=False):
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

def acc_rmin_aggregated(auc, ps, ns, return_solutions=False):
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
    def __init__(self, ps, ns, avg_auc):
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

    def __call__(self, x=None, z=None):
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

def acc_rmax_evaluate(ps, ns, aucs):
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

def acc_rmax_solve(ps, ns, avg_auc, return_solutions=False):
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

    check_cvxopt(res, 'acc_rmax_aggregated')

    aucs = np.array(res['x'][:, 0])

    acc = acc_rmax_evaluate(ps, ns, aucs)

    results = acc

    if return_solutions:
        results = results, (aucs, lower_bounds, np.repeat(1.0, k))

    return results

def acc_rmax_aggregated(auc, ps, ns, return_solutions=False):
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
    ps = np.array(ps)
    ns = np.array(ns)

    return acc_rmax_solve(ps, ns, auc, return_solutions)

class F_macc_min:
    """
    Implements the convex programming objective for the maximum accuracy
    minimization.
    """
    def __init__(self, ps, ns, avg_auc):
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

    def __call__(self, x=None, z=None):
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
        """
        if x is None and z is None:
            return (0, matrix(self.lower_bounds, (self.k, 1)))
        
        if x is not None:
            f = matrix(-np.sum(np.sqrt(1 - x)*self.weights.reshape(-1, 1)))
            Df = matrix(1.0/(2*np.sqrt(1 - x))*self.weights.reshape(-1, 1)).T
        
        if z is None:
            return (f, Df)
        
        hess = np.diag(1.0/(4*np.array(1 - x)[:, 0]**(3/2))*self.weights)
        
        hess = matrix(z[0]*hess)

        return (f, Df, hess)

def macc_min_evaluate(ps, ns, aucs):
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

def macc_min_solve(ps, ns, avg_auc, return_solutions=False):
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

    A = np.repeat(1.0/k, k).reshape(-1, 1).T
    b = np.array([avg_auc])
    G = np.vstack([np.eye(k), -np.eye(k)]).astype(float)
    h = np.hstack([np.repeat(1.0, k), -lower_bounds])
    
    G = matrix(G)
    h = matrix(h)
    A = matrix(A)
    b = matrix(b)

    print(lower_bounds)

    res = cp(F, G, h, A=A, b=b)

    check_cvxopt(res, 'macc_min_aggregated')

    print(res['status'])

    aucs = (np.array(res['x'])[:, 0])

    acc = macc_min_evaluate(ps, ns, aucs)

    results = acc

    if return_solutions:
        results = acc, (aucs, lower_bounds, np.repeat(1.0, k))
    
    return results

def macc_min_aggregated(auc, ps, ns, return_solutions=False):
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

    return macc_min_solve(ps, ns, auc, return_solutions)

def auc_from_aggregated(
    *,
    scores: dict,
    eps: float,
    p: int,
    n: int,
    lower: str = "min",
    upper: str = "max",
    k: int = None,
    raise_error: bool = False
) -> tuple:
    """
    This module applies the estimation scheme A to estimate AUC from scores
    in k-fold evaluation

    Args:
        scores (dict): the reported scores average scores
        eps (float): the numerical uncertainty
        p (int): the number of positive samples
        n (int): the number of negative samples
        lower (str): ('min'/'cmin') - the type of estimation for the lower bound
        upper (str): ('max'/'amax') - the type of estimation for the upper bound
        k (int): the number of folds (if any)

    Returns:
        tuple(float, float): the interval for the average AUC
    """

    scores = translate(scores)

    if ("sens" in scores) + ("spec" in scores) + ("acc" in scores) < 2:
        raise ValueError("Not enough scores specified for the estimation")

    if p is None or n is None:
        raise ValueError("For k-fold estimation p and n are needed")
    if p % k != 0 or n % k != 0:
        raise ValueError("For k-fold, p and n must be divisible by k")

    intervals = prepare_intervals_for_auc_estimation(scores, eps, p, n)

    if lower == "min":
        RL_avg_sens = R(intervals["sens"][0], k)
        RL_avg_spec = R(intervals["spec"][0], k)

        lower0 = np.mean([a * b for a, b in zip(RL_avg_sens, RL_avg_spec[::-1])])
    elif lower == "cmin":
        if intervals["sens"][0] < 1 - intervals["spec"][0]:
            raise ValueError(
                'sens >= 1 - spec does not hold for "\
                            "the corrected minimum curve'
            )
        lower0 = 0.5 + (1 - intervals["spec"][0] - intervals["sens"][0]) ** 2 / 2.0
    else:
        raise ValueError("Unsupported lower bound")

    if upper == "max":
        RU_avg_sens = R(intervals["sens"][1], k)
        RU_avg_spec = R(intervals["spec"][1], k)

        upper0 = 1 - np.mean(
            [(1 - a) * (1 - b) for a, b in zip(RU_avg_sens, RU_avg_spec[::-1])]
        )
    elif upper == "amax":
        if not intervals["acc"][0] >= max(p, n) / (p + n):
            raise ValueError("accuracy too small")

        upper0 = 1 - ((1 - intervals["acc"][1]) * (p + n)) ** 2 / (2 * n * p)
    else:
        raise ValueError("Unsupported upper bound")

    return (float(lower0), float(upper0))

def acc_from_aggregated(
    *, 
    scores: dict, 
    eps: float, 
    ps: int, 
    ns: int, 
    upper: str = "max",
    raise_errors: bool = False
) -> tuple:
    """
    This module applies the estimation scheme A to estimate AUC from scores

    Args:
        scores (dict): the reported scores
        eps (float): the numerical uncertainty
        p (int): the number of positive samples
        n (int): the number of negative samples
        upper (str): 'max'/'cmax' - the type of upper bound

    Returns:
        tuple(float, float): the interval for the maximum accuracy
    """

    if not raise_errors:
        return acc_from_auc_kfold_wrapper(
            scores=scores,
            eps=eps,
            ps=ps,
            ns=ns,
            upper=upper
        )

    scores = translate(scores)

    auc = (max(scores["auc"] - eps, 0), min(scores["auc"] + eps, 1))

    #if np.sqrt((1 - auc[0])*2*p*n) > min(p, n):
    if auc[0] < 1 - min(p, n)/(2*max(p, n)):
        lower = 1 - (np.sqrt(2 * p * n - 2 * auc[0] * p * n)) / (p + n)
    else:
        lower = max(p, n)/(p + n)
        #raise ValueError("AUC too small")
    
    if upper == "max":
        upper = (auc[1] * min(p, n) + max(p, n)) / (p + n)
    else:
        upper = (max(p, n) + min(p, n) * np.sqrt(2 * (auc[1] - 0.5))) / (p + n)

    return (float(lower), float(upper))

"""
This module implements the maximum accuracy estimations in the aggregated case
"""

import numpy as np

from cvxopt import matrix
from cvxopt.solvers import cp
from cvxopt import solvers

from ._utils import prepare_intervals, translate_folding

from ._acc_single import macc_min
from ._auc_aggregated import check_cvxopt
from ._acc_aggregated import acc_max_aggregated, acc_rmax_aggregated

__all__ = [
    "macc_min_aggregated",
    "max_acc_from_aggregated",
    "max_acc_lower_from_aggregated",
    "max_acc_upper_from_aggregated",
    "FMAccMin",
]


class FMAccMin:  # pylint: disable=too-few-public-methods
    """
    Implements the convex programming objective for the maximum accuracy
    minimization.
    """

    def __init__(self, ps: np.array, ns: np.array):
        """
        The constructor of the object

        Args:
            ps (np.array): the number of positive samples
            ns (np.array): the number of negative samples
        """
        self.ps = ps
        self.ns = ns
        self.k = len(ps)
        self.weights = np.sqrt(2 * (ps * ns)) / (ps + ns)
        self.lower_bounds = 1.0 - np.array(
            [min(p, n) / (2 * max(p, n)) for p, n in zip(ps, ns)]
        )
        self.upper_bounds = np.repeat(1.0 - np.min(1 / ((ps + 1) * (ns + 1))), self.k)

    def __call__(self, x: matrix = None, z: matrix = None):
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
            (int, matrix): the number of non-linear constraints and a feasible
                point when x is None and z is None
            (matrix, matrix): the objective value at x, and the gradient
                at x when x is not None but z is None
            (matrix, matrix, matrx): the objective value at x, the gradient
                at x and the weighted sum of the Hessian of the objective and
                all non-linear constraints with the weights z if z is not None
        """
        if x is None and z is None:
            return (0, matrix(self.lower_bounds, (self.k, 1)))
            # return (0, matrix(np.repeat(1.0, self.k), (self.k, 1)))

        if np.any(
            np.array(x)[:, 0] > self.upper_bounds
        ):  # or np.any(np.array(x)[:, 0] < self.lower_bounds):
            return None, None

        # if x is not None:
        f = matrix(
            -np.sum(np.clip(np.sqrt(1 - x), 0.0, 2.0) * self.weights.reshape(-1, 1))
        )
        df = matrix(
            1.0 / (2 * np.clip(np.sqrt(1 - x), 0.0, 2.0)) * self.weights.reshape(-1, 1)
        ).T

        if z is None:
            return (f, df)

        hess = np.diag(
            1.0
            / (4 * np.clip(np.array(1 - x), 0.0, 2.0)[:, 0] ** (3 / 2))
            * self.weights
        )

        hess = matrix(z[0] * hess)

        return (f, df, hess)


def macc_min_evaluate(ps: np.array, ns: np.array, aucs: np.array):
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
    ps: np.array, ns: np.array, avg_auc: float, return_solutions: bool = False
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

    Raises:
        ValueError: when no optimal solution is found
    """
    F = FMAccMin(ps, ns)  # pylint: disable=invalid-name

    k = ps.shape[0]

    lower_bounds = 1.0 - np.array([min(p, n) / (2 * max(p, n)) for p, n in zip(ps, ns)])
    upper_bounds = np.repeat(1.0 - np.min(1 / ((ps + 1) * (ns + 1))), k)
    # print(upper_bounds)

    A = np.repeat(1.0 / k, k).reshape(-1, 1).T  # pylint: disable=invalid-name
    b = np.array([avg_auc]).astype(float)
    G = np.vstack([np.eye(k), -np.eye(k)]).astype(float)  # pylint: disable=invalid-name
    h = np.hstack([upper_bounds, -lower_bounds])

    G = matrix(G)  # pylint: disable=invalid-name
    h = matrix(h)
    A = matrix(A)  # pylint: disable=invalid-name
    b = matrix(b)

    actual = solvers.options.get("show_progress", None)
    solvers.options["show_progress"] = False

    results = cp(F, G, h, A=A, b=b)

    solvers.options["show_progress"] = actual

    check_cvxopt(results, "macc_min_aggregated")

    aucs = np.array(results["x"])[:, 0]

    results = macc_min_evaluate(ps, ns, aucs)

    if return_solutions:
        results = results, (aucs, lower_bounds, upper_bounds)

    return results


def reduce_macc_min_edge_case(
    auc: float, ps: np.array, ns: np.array, eps: float, upper_bounds: np.array
):
    """
    Solves the reduced macc min problem

    Args:
        auc (float): the AUC value
        ps (np.array): the numbers of positives
        ns (np.array): the numbers of negatives
        eps (float): the epsilon
        upper_bounds (np.array): the upper bounds

    Returns:
        float, np.array, np.array: the result, the aucs and the upper bounds
    """
    n_1s = 0
    auc_p = auc

    sorting = np.argsort(upper_bounds)
    rev_sorting = np.zeros(len(ps), dtype=int)
    rev_sorting[sorting] = np.arange(len(ps))

    upper_bounds_p = upper_bounds[sorting]
    ps_p = ps[sorting]
    ns_p = ns[sorting]

    while len(ps_p) > 0 and auc_p >= np.mean(upper_bounds_p) - eps:
        n_1s += 1
        ps_p = ps_p[:-1]
        ns_p = ns_p[:-1]
        upper_bounds_p = upper_bounds_p[:-1]
        if len(ps_p) > 0:
            auc_p = (auc_p * (len(ps_p) + 1) - 1) / len(ps_p)

    if len(ps_p) > 0:
        results, (aucs, _, upper_bounds) = macc_min_solve(
            ps_p, ns_p, auc_p, return_solutions=True
        )

        aucs = np.hstack([aucs, np.repeat(1.0, n_1s)])
        upper_bounds = np.hstack([upper_bounds, np.repeat(1.0, n_1s)])
        results = macc_min_evaluate(ps[sorting], ns[sorting], aucs)

        aucs = aucs[rev_sorting]
        upper_bounds = upper_bounds[rev_sorting]
    else:
        results = 1.0
        aucs = np.repeat(1.0, len(ps))

    return results, aucs, upper_bounds


def macc_min_aggregated(
    auc: float, ps: np.array, ns: np.array, return_solutions: bool = False
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

    Raises:
        ValueError: when the auc is less then the desired lower bound or no
        optimal solution is found
    """
    ps = np.array(ps)
    ns = np.array(ns)

    eps = np.min(1.0 / (ps * ns))

    k = len(ps)

    lower_bounds = 1.0 - np.array([min(p, n) / (2 * max(p, n)) for p, n in zip(ps, ns)])

    if auc < np.mean(lower_bounds):
        raise ValueError("auc too small (macc_min_aggregated)")

    upper_bounds = np.repeat(1.0 - np.min(1 / ((ps + 1) * (ns + 1))), k)

    if auc == 1.0:  # or auc >= np.mean(upper_bounds):
        # the gradient would go to infinity in this case
        results = 1.0

        if return_solutions:
            results = results, (
                np.repeat(1.0, len(ps)),
                lower_bounds,
                np.repeat(1.0, len(ps)),
            )
        return results
    if auc >= np.mean(upper_bounds) - eps:
        results, aucs, upper_bounds = reduce_macc_min_edge_case(
            auc, ps, ns, eps, upper_bounds
        )

        if return_solutions:
            return results, (aucs, lower_bounds, upper_bounds)
        return results

    return macc_min_solve(ps, ns, auc, return_solutions)


def max_acc_lower_from_aggregated(
    *,
    scores: dict,
    eps: float,
    ps: int = None,
    ns: int = None,
    folding: dict = None,
    lower: str = "min",
) -> tuple:
    """
    This function applies the lower bound estimation schemes to estimate
    maximum accuracy from scores

    Args:
        scores (dict): the reported scores
        eps (float): the numerical uncertainty
        p (int): the number of positive samples
        n (int): the number of negative samples
        folding (dict): description of a folding, alternative to specifying
                        ps and ns, contains the keys 'p', 'n', 'n_repeats',
                        'n_folds', 'folding' (currently 'stratified_sklearn'
                        supported for 'folding')
        lower (str): 'min'

    Returns:
        float: the lower bound for the maximum accuracy

    Raises:
        ValueError: when no optimal solution is found, or the parameters
        violate the expectations of the estimation scheme
    """

    intervals = prepare_intervals(scores, eps)

    if "auc" not in intervals:
        raise ValueError("auc must be specified")

    if (ps is not None or ns is not None) and folding is not None:
        raise ValueError("specify either (ps and ns) or folding")

    if ps is None and ns is None and folding is not None:
        ps, ns = translate_folding(folding)

    if lower == "min":
        lower0 = macc_min_aggregated(intervals["auc"][0], ps, ns)
    else:
        raise ValueError(f"unsupported lower bound {lower}")

    return lower0


def max_acc_upper_from_aggregated(
    *,
    scores: dict,
    eps: float,
    ps: int = None,
    ns: int = None,
    folding: dict = None,
    upper: str = "max",
) -> tuple:
    """
    This function applies the upper bound estimation schemes to estimate
    maximum accuracy from scores

    Args:
        scores (dict): the reported scores
        eps (float): the numerical uncertainty
        p (int): the number of positive samples
        n (int): the number of negative samples
        folding (dict): description of a folding, alternative to specifying
                        ps and ns, contains the keys 'p', 'n', 'n_repeats',
                        'n_folds', 'folding' (currently 'stratified_sklearn'
                        supported for 'folding')
        upper (str): 'max'/'rmax' - the type of upper bound

    Returns:
        float: the upper bound for the maximum accuracy

    Raises:
        ValueError: when no optimal solution is found, or the parameters
        violate the expectations of the estimation scheme
    """

    intervals = prepare_intervals(scores, eps)

    if "auc" not in intervals:
        raise ValueError("auc must be specified")

    if (ps is not None or ns is not None) and folding is not None:
        raise ValueError("specify either (ps and ns) or folding")

    if ps is None and ns is None and folding is not None:
        ps, ns = translate_folding(folding)

    if upper == "max":
        upper0 = acc_max_aggregated(intervals["auc"][1], ps, ns)
    elif upper == "rmax":
        upper0 = acc_rmax_aggregated(intervals["auc"][1], ps, ns)
    else:
        raise ValueError(f"unsupported upper bound {upper}")

    return upper0


def max_acc_from_aggregated(
    *,
    scores: dict,
    eps: float,
    ps: int = None,
    ns: int = None,
    folding: dict = None,
    lower: str = "min",
    upper: str = "max",
) -> tuple:
    """
    This function applies the estimation schemes to estimate
    maximum acc from scores

    Args:
        scores (dict): the reported scores
        eps (float): the numerical uncertainty
        p (int): the number of positive samples
        n (int): the number of negative samples
        folding (dict): description of a folding, alternative to specifying
                        ps and ns, contains the keys 'p', 'n', 'n_repeats',
                        'n_folds', 'folding' (currently 'stratified_sklearn'
                        supported for 'folding')
        lower (str): 'min'
        upper (str): 'max'/'rmax' - the type of upper bound

    Returns:
        tuple(float, float): the interval for the accuracy

    Raises:
        ValueError: when no optimal solution is found, or the parameters
        violate the expectations of the estimation scheme
    """

    lower0 = max_acc_lower_from_aggregated(
        scores=scores, eps=eps, ps=ps, ns=ns, folding=folding, lower=lower
    )

    upper0 = max_acc_upper_from_aggregated(
        scores=scores, eps=eps, ps=ps, ns=ns, folding=folding, upper=upper
    )

    return (lower0, upper0)

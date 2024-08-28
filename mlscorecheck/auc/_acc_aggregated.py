"""
This module implements the aggregated AUC based accuracy 
estimation related functionalities
"""

import numpy as np

from cvxopt import matrix
from cvxopt.solvers import cp
from cvxopt import solvers

from ._acc_single import (
    prepare_intervals,
    acc_min,
    acc_max,
    macc_min,
)
from ._auc_aggregated import R, check_cvxopt

__all__ = [
    "acc_min_aggregated",
    "acc_max_aggregated",
    "acc_rmin_aggregated",
    "acc_rmax_aggregated",
    "acc_from_aggregated",
    "macc_min_aggregated",
    "max_acc_from_aggregated",
]


def acc_min_aggregated(
    auc: float, ps: np.array, ns: np.array, return_solutions: bool = False
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

    Raises:
        ValueError: when the configuration is not feasible
    """
    ps = np.array(ps)
    ns = np.array(ns)

    k = len(ps)

    w = np.array([min(p, n) / (p + n) for p, n in zip(ps, ns)])
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
    auc: float, ps: np.array, ns: np.array, return_solutions: bool = False
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

    Raises:
        ValueError: when the configuration is not feasible
    """
    ps = np.array(ps)
    ns = np.array(ns)

    k = len(ps)

    w = np.array([min(p, n) / (p + n) for p, n in zip(ps, ns)])
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
    auc: float, ps: np.array, ns: np.array, return_solutions: bool = False
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

    acc = np.mean([min(p, n) / (p + n) for p, n in zip(ps, ns)])

    results = acc

    if return_solutions:
        results = acc, (
            np.repeat(0.7, len(ps)),
            np.repeat(0.5, len(ps)),
            np.repeat(1.0, len(ps)),
        )

    return results


class FAccRMax:  # pylint: disable=too-few-public-methods
    """
    Implements the convex programming objective for the regulated
    accuracy maximization.
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
        self.maxs = np.array([max(p, n) for p, n in zip(ps, ns)])
        self.mins = np.array([min(p, n) for p, n in zip(ps, ns)])
        self.weights = self.mins / (ps + ns)

    def __call__(self, x: matrix = None, z: matrix = None):
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

        # if x is not None:
        # the function to be evaluated
        f = matrix(-np.sum(np.sqrt(2 * x - 1) * self.weights.reshape(-1, 1)))
        # the gradient
        df = matrix(-1.0 / np.sqrt(2 * x - 1) * self.weights.reshape(-1, 1)).T

        if z is None:
            return (f, df)

        # the Hessian
        hess = np.diag((2 * np.array(x)[:, 0] - 1.0) ** (-3 / 2) * self.weights)

        hess = matrix(z[0] * hess)

        return (f, df, hess)


def acc_rmax_evaluate(ps: np.array, ns: np.array, aucs: np.array):
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

    return np.mean((maxs + mins * np.sqrt(2 * aucs - 1)) / (ps + ns))


def acc_rmax_solve(
    ps: np.array, ns: np.array, avg_auc: float, return_solutions: bool = False
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
        float | (float, np.array, np.array, np.array, np.array, np.array): the
        mean accuracy, or the mean accuracy, the auc parameters, the vectors of
        ps, ns, and the lower bounds and upper bounds

    Raises:
        ValueError: when no optimal solution is found
    """
    F = FAccRMax(ps, ns)  # pylint: disable=invalid-name

    k = ps.shape[0]

    lower_bounds = np.repeat(0.5, k)

    A = np.repeat(1.0 / k, k).reshape(-1, 1).T  # pylint: disable=invalid-name
    b = np.array([avg_auc])
    G = np.vstack([np.eye(k), -np.eye(k)]).astype(float)  # pylint: disable=invalid-name
    h = np.hstack([np.repeat(1.0, k), -lower_bounds])

    G = matrix(G)  # pylint: disable=invalid-name
    h = matrix(h)
    A = matrix(A)  # pylint: disable=invalid-name
    b = matrix(b)

    actual = solvers.options.get("show_progress", None)
    solvers.options["show_progress"] = False

    res = cp(F, G, h, A=A, b=b)

    solvers.options["show_progress"] = actual

    check_cvxopt(res, "acc_rmax_aggregated")

    aucs = np.array(res["x"])[:, 0]

    results = acc_rmax_evaluate(ps, ns, aucs)

    if return_solutions:
        results = results, (aucs, ps, ns, lower_bounds, np.repeat(1.0, k))

    return results


def acc_rmax_aggregated(
    auc: float, ps: np.array, ns: np.array, return_solutions: bool = False
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
        float | (float, np.array, np.array, np.array, np.array, np.array): the
        mean accuracy, or the mean accuracy, the auc parameters, the vectors of
        ps, ns, and the lower bounds and upper bounds

    Raises:
        ValueError: when auc < 0.5 or no optimal solution is found
    """

    if auc < 0.5:
        raise ValueError("auc too small (acc_rmax_aggregated)")

    ps = np.array(ps)
    ns = np.array(ns)

    return acc_rmax_solve(ps, ns, auc, return_solutions)


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
        self.maxs = np.array([max(p, n) for p, n in zip(ps, ns)])
        self.mins = np.array([min(p, n) for p, n in zip(ps, ns)])
        self.weights = np.sqrt(2 * (ps * ns)) / (ps + ns)
        self.lower_bounds = 1.0 - np.array(
            [min(p, n) / (2 * max(p, n)) for p, n in zip(ps, ns)]
        )

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
            # return (0, matrix(np.repeat(1.0, self.k), (self.k, 1)))

        # if x is not None:
        f = matrix(-np.sum(np.sqrt(1 - x) * self.weights.reshape(-1, 1)))
        df = matrix(1.0 / (2 * np.sqrt(1 - x)) * self.weights.reshape(-1, 1)).T

        if z is None:
            return (f, df)

        hess = np.diag(1.0 / (4 * np.array(1 - x)[:, 0] ** (3 / 2)) * self.weights)

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

    A = np.repeat(1.0 / k, k).reshape(-1, 1).T  # pylint: disable=invalid-name
    b = np.array([avg_auc])
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

    lower_bounds = 1.0 - np.array([min(p, n) / (2 * max(p, n)) for p, n in zip(ps, ns)])

    if auc < np.mean(lower_bounds):
        raise ValueError("auc too small (macc_min_aggregated)")

    if auc == 1.0:
        # the gradient would go to infinity in this case
        results = 1.0

        if return_solutions:
            results = results, (
                np.repeat(1.0, len(ps)),
                lower_bounds,
                np.repeat(1.0, len(ps)),
            )
        return results

    return macc_min_solve(ps, ns, auc, return_solutions)


def acc_from_aggregated(
    *,
    scores: dict,
    eps: float,
    ps: int,
    ns: int,
    lower: str = "min",
    upper: str = "max",
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

    Raises:
        ValueError: when no optimal solution is found, or the parameters
        violate the expectations of the estimation scheme
    """

    intervals = prepare_intervals(scores, eps)

    if "auc" not in intervals:
        raise ValueError("auc must be specified")

    lower0 = None
    upper0 = None

    if lower == "min":
        lower0 = acc_min_aggregated(intervals["auc"][0], ps, ns)
    elif lower == "rmin":
        lower0 = acc_rmin_aggregated(intervals["auc"][0], ps, ns)
    else:
        raise ValueError(f"unsupported lower bound {lower}")

    if upper == "max":
        upper0 = acc_max_aggregated(intervals["auc"][1], ps, ns)
    elif upper == "rmax":
        upper0 = acc_rmax_aggregated(intervals["auc"][1], ps, ns)
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
    upper: str = "max",
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

    Raises:
        ValueError: when no optimal solution is found, or the parameters
        violate the expectations of the estimation scheme
    """

    intervals = prepare_intervals(scores, eps)

    if "auc" not in intervals:
        raise ValueError("auc must be specified")

    lower0 = None
    upper0 = None

    if lower == "min":
        lower0 = macc_min_aggregated(intervals["auc"][0], ps, ns)
    else:
        raise ValueError(f"unsupported lower bound {lower}")

    if upper == "max":
        upper0 = acc_max_aggregated(intervals["auc"][1], ps, ns)
    elif upper == "rmax":
        upper0 = acc_rmax_aggregated(intervals["auc"][1], ps, ns)
    else:
        raise ValueError(f"unsupported upper bound {upper}")

    return (lower0, upper0)

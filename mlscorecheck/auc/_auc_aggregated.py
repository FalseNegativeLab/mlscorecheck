"""
This module implements the aggregated AUC related functionalities
"""

import numpy as np

import pulp as pl

from cvxopt import matrix
from cvxopt.solvers import qp
from cvxopt import solvers

from ._utils import (
    R,
    check_cvxopt,
    translate_folding,
    translate_scores,
    prepare_intervals,
)
from ._auc_single import auc_maxa, auc_armin

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
    "estimate_acc_interval",
    "estimate_tpr_interval",
    "estimate_fpr_interval",
    "augment_intervals_aggregated",
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
    return (
        sum(
            (tps[idx] + tns[idx]) * (1.0 / (ps[idx] + ns[idx]))
            for idx in range(len(ps))
        )
        * 1.0
        / (len(ps))
    )


def tpr_expression(tps, ps):
    """
    The mean tpr expression

    Args:
        tps (list|np.array): the true positives
        ps (list|np.array): the positives

    Returns:
        float|obj: the average tpr
    """
    return sum(tps[idx] * (1.0 / ps[idx]) for idx in range(len(ps))) * (1.0 / len(ps))


def fpr_expression(tns, ns):
    """
    The mean fpr expression

    Args:
        tns (list|np.array): the true negatives
        ns (list|np.array): the negatives

    Returns:
        float|obj: the average fpr
    """
    return sum(1 - tns[idx] * (1.0 / ns[idx]) for idx in range(len(ns))) * (
        1.0 / len(ns)
    )


def init_lp_variables(ps, ns):
    """
    Initializes the linear programming variables

    Args:
        ps (list): the postives
        ns (list): the negatives

    Returns:
        list, list: the true positive and true negative variables
    """
    tps = [
        pl.LpVariable(f"tp_{idx}", lowBound=0, upBound=ps[idx], cat=pl.LpInteger)
        for idx in range(len(ps))
    ]
    tns = [
        pl.LpVariable(f"tn_{idx}", lowBound=0, upBound=ns[idx], cat=pl.LpInteger)
        for idx in range(len(ns))
    ]

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
        ids = variable.name.split("_")
        if ids[0] == "tp":
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


def augment_intervals_aggregated(intervals: dict, ps: np.array, ns: np.array) -> dict:
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
        intervals["tpr"] = estimate_tpr_interval(
            intervals["fpr"], intervals["acc"], ps, ns
        )
    if "fpr" not in intervals and ("acc" in intervals and "tpr" in intervals):
        intervals["fpr"] = estimate_fpr_interval(
            intervals["tpr"], intervals["acc"], ps, ns
        )
    if "acc" not in intervals and ("fpr" in intervals and "tpr" in intervals):
        intervals["acc"] = estimate_acc_interval(
            intervals["fpr"], intervals["tpr"], ps, ns
        )

    return intervals


def auc_min_aggregated(
    fpr: float, tpr: float, k: int, return_solutions: bool = False
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

    r_avg_tpr = R(tpr, k)
    r_avg_fpr = R(fpr, k)

    results = float(np.mean([a * b for a, b in zip(r_avg_tpr, 1.0 - r_avg_fpr)]))

    if return_solutions:
        results = results, (
            r_avg_fpr,
            r_avg_tpr,
            np.repeat(0.0, k),
            np.repeat(1.0, k),
        )

    return results


def auc_max_aggregated(
    fpr: float, tpr: float, k: int, return_solutions: bool = False
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
        float | (float, np.array, np.array, np.array, np.array): the area or the area, the
        solutions and the bounds
    """

    r_avg_tpr = R(tpr, k)
    r_avg_fpr = R(fpr, k)

    results = float(1 - np.mean([a * b for a, b in zip(1 - r_avg_tpr, r_avg_fpr)]))

    if return_solutions:
        results = results, (
            r_avg_fpr,
            r_avg_tpr,
            np.repeat(0.0, k),
            np.repeat(1.0, k),
        )

    return results


def auc_rmin_aggregated(
    fpr: float, tpr: float, k: int, return_solutions: bool = False
) -> float:
    """
    The average area under the regulated minimum curves at the average fpr, tpr

    Args:
        fpr (list): lower bound on average false positive rate
        tpr (list): upper bound on average true positive rate
        return_solutions (bool): whether to return the solutions for the
        underlying curves

    Returns:
        float | (float, np.array, np.array, np.array, np.array): the area or the area, the
        solutions and the bounds

    Raises:
        ValueError: when tpr >= fpr does not hold
    """
    if tpr < fpr:
        raise ValueError("tpr >= fpr does not hold")

    results = 0.5 + (fpr - tpr) ** 2 / 2.0

    if return_solutions:
        results = results, (
            np.repeat(fpr, k),
            np.repeat(tpr, k),
            np.repeat(0.0, k),
            np.repeat(1.0, k),
        )

    return results


def auc_maxa_evaluate(ps: np.array, ns: np.array, accs: np.array) -> float:
    """
    Evaluates the maxa solutions

    Args:
        ps (np.array): the numbers of positives in the evaluation sets
        ns (np.array): the numbers of negatives in the evaluation sets
        accs (np.array): the accuracies

    Returns:
        float: the area under the curve
    """
    return np.mean([auc_maxa(acc, p, n) for acc, p, n in zip(accs, ps, ns)])


def auc_maxa_solve(
    ps: np.array, ns: np.array, avg_acc: float, return_solutions: bool = False
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

    Raises:
        ValueError: when no optimal solution is found
    """
    # pylint: disable=too-many-locals
    ps = np.array(ps)
    ns = np.array(ns)

    k = ps.shape[0]

    w = (ps + ns) ** 2 / (2 * ps * ns)
    Q = 2.0 * np.diag(w)  # pylint: disable=invalid-name
    q = -2.0 * w
    A = np.repeat(1.0 / k, k).reshape(-1, 1).T  # pylint: disable=invalid-name
    b = np.array([avg_acc])
    G = np.vstack([np.eye(k), -np.eye(k)]).astype(float)  # pylint: disable=invalid-name
    lower = np.array([max(p, n) / (p + n) for p, n in zip(ps, ns)])
    h = np.hstack([np.repeat(1.0, k), -lower])

    Q = matrix(Q)  # pylint: disable=invalid-name
    q = matrix(q)
    G = matrix(G)  # pylint: disable=invalid-name
    h = matrix(h)
    A = matrix(A)  # pylint: disable=invalid-name
    b = matrix(b)

    actual = solvers.options.get("show_progress", None)
    solvers.options["show_progress"] = False

    res = qp(P=Q, q=q, G=G, h=h, A=A, b=b)

    solvers.options["show_progress"] = actual

    check_cvxopt(res, "auc_maxa_solve")

    result = np.array(res["x"])[:, 0]

    results = float(auc_maxa_evaluate(ps, ns, result))

    if return_solutions:
        results = results, (result.astype(float), lower, np.repeat(1.0, k))

    return results


def auc_maxa_aggregated(
    acc: float, ps: np.array, ns: np.array, return_solutions: bool = False
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
        float | (float, np.array, np.array, np.array): the area or the area,
        the solutions and the lower and upper bounds

    Raises:
        ValueError: when the mean accuracy is smaller than the mean max(p, n)/(p + n),
        or no optimal solution is found
    """

    if acc < np.mean([max(p, n) / (p + n) for p, n in zip(ps, ns)]):
        raise ValueError("accuracy too small")

    return auc_maxa_solve(ps, ns, acc, return_solutions)


def auc_amin_aggregated(
    acc: float, ps: np.array, ns: np.array, return_solutions: bool = False
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

    Raises:
        ValueError: when the configuration is infeasible
    """
    ps = np.array(ps)
    ns = np.array(ns)

    k = len(ps)

    weights = np.repeat(1.0, k) + np.array(
        [float(max(p, n)) / min(p, n) for p, n in zip(ps, ns)]
    )
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
        auc = float(np.sum(accs * (weights) - weights + 1.0) / k)
        lower = thresholds
        upper = np.repeat(1.0, k)

    results = auc

    if return_solutions:
        results = results, (accs, ps, ns, lower, upper)

    return results


def auc_amax_aggregated(
    acc: float, ps: np.array, ns: np.array, return_solutions: bool = False
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

    Raises:
        ValueError: when the configuration is infeasible
    """
    ps = np.array(ps)
    ns = np.array(ns)

    k = len(ps)

    weights = np.repeat(1.0, k) + np.array(
        [float(max(p, n)) / min(p, n) for p, n in zip(ps, ns)]
    )
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
        auc = float(np.sum(accs * weights[::-1]) / k)
        upper = thresholds
        lower = np.repeat(0.0, k)

    results = auc

    if return_solutions:
        results = results, (accs, ps, ns, lower, upper)

    return results


def auc_armin_solve(
    ps: np.array, ns: np.array, avg_acc: float, return_bounds: bool = False
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

    Raises:
        ValueError: when no optimal solution is found
    """
    # pylint: disable=too-many-locals
    k = ps.shape[0]
    w = np.array([max(p, n) / min(p, n) for p, n in zip(ps, ns)])

    p = -2 * w - 2 * w**2

    # note the multiplier of 2 in Q to get the weighting of the quadratic
    # and linear term correct in the cxopt formalism
    Q = 2 * np.diag(1 + w**2 + 2 * w)  # pylint: disable=invalid-name
    A = np.repeat(1.0 / k, k).reshape(-1, 1).T  # pylint: disable=invalid-name
    b = np.array([avg_acc])
    G = np.vstack([np.eye(k), -np.eye(k)]).astype(float)  # pylint: disable=invalid-name
    lower = np.array([max(p, n) / (p + n) for p, n in zip(ps, ns)])
    h = np.hstack([np.repeat(1.0, k), -lower])

    Q = matrix(Q)  # pylint: disable=invalid-name
    p = matrix(p)
    G = matrix(G)  # pylint: disable=invalid-name
    h = matrix(h)
    A = matrix(A)  # pylint: disable=invalid-name
    b = matrix(b)

    actual = solvers.options.get("show_progress", None)
    solvers.options["show_progress"] = False

    res = qp(P=Q, q=p, G=G, h=h, A=A, b=b)

    solvers.options["show_progress"] = actual

    check_cvxopt(res, "auc_armin_aggregated")

    results = np.array(res["x"])[:, 0]

    if return_bounds:
        results = results, lower

    return results


def auc_armin_aggregated(
    acc: float, ps: np.array, ns: np.array, return_solutions: bool = False
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

    Raises:
        ValueError: when no optimal solution is found or the configuration is
        infeasible
    """
    ps = np.array(ps)
    ns = np.array(ns)

    lower = np.array([min(p, n) / (p + n) for p, n in zip(ps, ns)])
    upper = np.array([max(p, n) / (p + n) for p, n in zip(ps, ns)])

    if acc < np.mean(lower):
        raise ValueError("acc too small for the configuration (auc_armin_aggregated)")
    if np.mean(lower) <= acc <= np.mean(upper):
        auc = 0.5
        results = auc
        if return_solutions:
            results = results, ((lower + upper) / 2, ps, ns, lower, upper)
        return results

    accs, lower = auc_armin_solve(ps, ns, acc, return_bounds=True)

    auc = float(np.mean([auc_armin(acc, p, n) for acc, p, n in zip(accs, ps, ns)]))

    results = auc

    if return_solutions:
        results = results, (accs, ps, ns, lower, np.repeat(1.0, len(ps)))

    return results


def check_applicability_aggregated(
    intervals: dict, lower: str, upper: str, ps: int, ns: int
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
    if lower in ["min", "rmin"] or upper in ["max"]:
        if "fpr" not in intervals or "tpr" not in intervals:
            raise ValueError("fpr, tpr or their complements must be specified")
    if lower in ["amin", "armin"] or upper in ["amax", "maxa"]:
        if ps is None or ns is None:
            raise ValueError("p and n must be specified")
    if lower in ["amin", "armin"] or upper in ["amax", "maxa"]:
        if "acc" not in intervals:
            raise ValueError("acc must be specified")


def auc_from_aggregated(
    *,
    scores: dict,
    eps: float,
    k: int = None,
    ps: np.array = None,
    ns: np.array = None,
    folding: dict = None,
    lower: str = "min",
    upper: str = "max",
) -> tuple:
    """
    This function applies the estimation schemes to estimate AUC from scores

    Args:
        scores (dict): the reported scores
        eps (float): the numerical uncertainty
        k (int): the number of evaluation sets
        ps (int): the numbers of positive samples
        ns (int): the numbers of negative samples
        folding (dict): description of a folding, alternative to specifying
                        ps and ns, contains the keys 'p', 'n', 'n_repeats',
                        'n_folds', 'folding' (currently 'stratified_sklearn'
                        supported for 'folding')
        lower (str): ('min'/'rmin'/'amin'/'armin') - the type of
                        estimation for the lower bound
        upper (str): ('max'/'maxa'/'amax') - the type of estimation for
                        the upper bound

    Returns:
        tuple(float, float): the interval for the AUC

    Raises:
        ValueError: when no optimal solution is found, or the configuration is
        infeasible, or not enough data is provided for the estimation method
    """

    scores = translate_scores(scores)
    intervals = prepare_intervals(scores, eps)

    if (ps is not None or ns is not None or k is not None) and folding is not None:
        raise ValueError("specify either (ps and ns/k) or folding")

    if ps is None and ns is None and k is None and folding is not None:
        ps, ns = translate_folding(folding)
        k = len(ps)

    if ps is not None and ns is not None:
        intervals = augment_intervals_aggregated(intervals, ps, ns)

    check_applicability_aggregated(intervals, lower, upper, ps, ns)

    if lower == "min":
        lower0 = auc_min_aggregated(intervals["fpr"][1], intervals["tpr"][0], k)
    elif lower == "rmin":
        lower0 = auc_rmin_aggregated(intervals["fpr"][0], intervals["tpr"][1], k)
    elif lower == "amin":
        lower0 = auc_amin_aggregated(intervals["acc"][0], ps, ns)
    elif lower == "armin":
        lower0 = auc_armin_aggregated(intervals["acc"][0], ps, ns)
    else:
        raise ValueError(f"unsupported lower bound {lower}")

    if upper == "max":
        upper0 = auc_max_aggregated(intervals["fpr"][0], intervals["tpr"][1], k)
    elif upper == "amax":
        upper0 = auc_amax_aggregated(intervals["acc"][1], ps, ns)
    elif upper == "maxa":
        upper0 = auc_maxa_aggregated(intervals["acc"][1], ps, ns)
    else:
        raise ValueError(f"unsupported upper bound {upper}")

    return (lower0, upper0)

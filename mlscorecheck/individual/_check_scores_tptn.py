"""
This module implements the checking of scores with possible
tp and tn combinations.
"""

import copy

from ..core import NUMERICAL_TOLERANCE, logger, update_uncertainty

from ._interval import Interval, IntervalUnion
from ._utils import resolve_aliases_and_complements, create_intervals
from ._tptn_solution_bundles import tptn_solutions, sens_tp, spec_tn, is_applicable_tptn
from ._pair_solutions import solution_specifications

__all__ = [
    "check_scores_tptn_pairs",
    "_check_scores_tptn_pairs",
    "_check_scores_tptn_intervals",
]

preferred_order = [
    "acc",
    "sens",
    "spec",
    "bacc",
    "npv",
    "ppv",
    "f1p",
    "f1n",
    "fbp",
    "fbn",
    "fm",
    "bm",
    "pt",
    "lrp",
    "lrn",
    "mk",
    "dor",
    "ji",
    "gm",
    "upm",
    "kappa",
    "mcc",
]


def iterate_tptn(
    *,
    score: str,
    score_value: float,
    valid_pairs: dict,
    sol_fun,
    params: dict,
    iterate_by: str
) -> dict:
    """
    Iterate through the potential values of tp or tn and construct feasible pairs

    Args:
        score (str): name of the score being used
        score_value (float): the value of the score
        valid_pairs (dict(int,Interval|IntervalUnion)): the valid tp,tn pairs
        sol_fun (callable): the solution function providing an interval for ``tp`` if
                            iterate_by is ``tn`` and vice versa.
        params (dict): the parameters to use
        iterate_by (str): the figure to iterate by (``tp``/``tn``)

    Returns:
        dict(int,Interval|IntervalUnion): the feasible (``tp``,``tn``) pairs
    """
    results = {}

    for value in valid_pairs:
        if not is_applicable_tptn(
            score, score_value, "tp" if iterate_by == "tn" else "tn"
        ):
            results[value] = valid_pairs[value]
            continue

        interval = sol_fun(**params, **{iterate_by: value})

        if interval is None:
            results[value] = valid_pairs[value]
            continue

        interval = interval.intersection(valid_pairs[value]).shrink_to_integers()

        if not interval.is_empty():
            results[value] = interval

    return results


def update_sens(p: int, valid_pairs: dict, score_int, solve_for: str) -> dict:
    """
    Update sensitivity intervals

    Args:
        p (int): the number of positives
        valid_pairs (dict(int,Interval|IntervalUnion)): the actual intervals
        score_int (Interval): the score interval
        solve_for (str): 'tp'/'tn' - the figure to solve for

    Returns:
        dict(int,Interval|IntervalUnion): the updated intervals
    """
    ints = sens_tp(sens=score_int, p=p)
    ints = ints.intersection(Interval(0, p + 1)).shrink_to_integers()

    if solve_for == "tp":
        valid_pairs = {
            key: value.intersection(ints) for key, value in valid_pairs.items()
        }
        valid_pairs = {
            key: value for key, value in valid_pairs.items() if not value.is_empty()
        }
    else:
        valid_pairs = {
            value: interval
            for value, interval in valid_pairs.items()
            if ints.contains(value)
        }

    return valid_pairs


def update_spec(n: int, valid_pairs: dict, score_int, solve_for: str) -> dict:
    """
    Update specificity intervals

    Args:
        n (int): the number of negatives
        valid_pairs (dict(int,Interval|IntervalUnion)): the actual intervals
        score_int (Interval|IntervalUnion): the score interval
        solve_for (str): 'tp'/'tn' - the figure to solve for

    Returns:
        dict(int,Interval|IntervalUnion): the updated intervals
    """
    ints = spec_tn(spec=score_int, n=n)
    ints = ints.intersection(Interval(0, n + 1)).shrink_to_integers()

    if solve_for == "tn":
        valid_pairs = {
            key: value.intersection(ints) for key, value in valid_pairs.items()
        }
        valid_pairs = {
            key: value for key, value in valid_pairs.items() if not value.is_empty()
        }
    else:
        valid_pairs = {
            value: interval
            for value, interval in valid_pairs.items()
            if ints.contains(value)
        }

    return valid_pairs


def initialize_valid_pairs(
    p: int, n: int, iterate_by: str, init_tptn_intervals: dict
) -> dict:
    """
    Initializes the valid pairs, either from the original input or the
    prefiltered intervals.

    Args:
        p (int): the number of positives
        n (int): the number of negatives
        iterate_by (str): the figure to iterate by ('tp'/'tn')
        init_tptn_intervals (dict(str,tuple)): the prefiltered 'tp' and 'tn' intervals

    Returns:
        dict(int,Interval|IntervalUnion): the initialized pairs
    """
    if init_tptn_intervals is not None:
        tp_int = IntervalUnion(init_tptn_intervals["tp"])
        tn_int = IntervalUnion(init_tptn_intervals["tn"])
        valid_pairs = {}
        if iterate_by == "tp":
            for interval in tp_int.intervals:
                for tp in range(interval.lower_bound, interval.upper_bound + 1):
                    valid_pairs[tp] = copy.deepcopy(tn_int)
        else:
            for interval in tn_int.intervals:
                for tn in range(interval.lower_bound, interval.upper_bound + 1):
                    valid_pairs[tn] = copy.deepcopy(tp_int)
        return valid_pairs

    init_interval = Interval(0, n + 1) if iterate_by == "tp" else Interval(0, p + 1)
    return {key: init_interval for key in range(p + 1 if iterate_by == "tp" else n + 1)}


def _check_scores_tptn_pairs(
    p: int,
    n: int,
    scores: dict,
    eps,
    *,
    numerical_tolerance: float = NUMERICAL_TOLERANCE,
    solve_for: str = None,
    init_tptn_intervals: dict = None
) -> dict:
    """
    Check scores by iteratively reducing the set of feasible ``tp``, ``tn`` pairs.

    Args:
        p (int): the number of positives
        n (int): the number of negatives
        scores (dict): the available reported scores
        eps (float|dict(str,float)): the numerical uncertainties for all scores or each
                                        score individually
        numerical_tolerance (float): the additional numerical tolerance
        solve_for (str): the figure solving for (the other is used to iterate by) (``tp``/``tn``)
                        If None, the optimal one is being used.
        init_tptn_intervals (None|dict(str,tuple)): the initial tp and tn intervals

    Returns:
        dict: a summary of the results. When the ``inconsistency`` flag is True, it indicates
        that the set of feasible ``tp``, ``tn`` pairs is empty. The list under the key
        ``details`` provides further details from the analysis of the scores one after the other.
        Under the key ``n_valid_tptn_pairs`` one finds the number of tp and tn pairs compatible with
        all scores.
    """
    if solve_for is None:
        solve_for = "tn" if p < n else "tp"

    if solve_for not in {"tp", "tn"}:
        raise ValueError(
            "The specified ``solve_for`` variable needs to be either "
            "``tp`` or ``tn``."
        )

    iterate_by = "tn" if solve_for == "tp" else "tp"

    # resolving aliases and complements
    scores = resolve_aliases_and_complements(scores)

    # updating the uncertainties
    eps = eps if isinstance(eps, dict) else {score: eps for score in scores}
    eps = update_uncertainty(eps, numerical_tolerance)

    params = {
        "p": p,
        "n": n,
        "beta_positive": scores.get("beta_positive"),
        "beta_negative": scores.get("beta_negative"),
    }

    valid_pairs = initialize_valid_pairs(p, n, iterate_by, init_tptn_intervals)

    details = []

    for score in [score for score in preferred_order if score in scores]:
        logger.info(
            "testing %s, feasible tptn pairs: %d",
            score,
            p * n if valid_pairs is None else len(valid_pairs),
        )

        score_int = Interval(scores[score] - eps[score], scores[score] + eps[score])

        params[score] = score_int

        detail = {
            "testing": score,
            "score_interval": score_int,
            "n_tptn_pairs_before": p * n if valid_pairs is None else len(valid_pairs),
        }

        if score not in {"sens", "spec"}:
            valid_pairs = iterate_tptn(
                score=score,
                score_value=scores[score],
                valid_pairs=valid_pairs,
                sol_fun=tptn_solutions[score][solve_for],
                params=params,
                iterate_by=iterate_by,
            )
        elif score == "sens":
            valid_pairs = update_sens(
                p=p, valid_pairs=valid_pairs, score_int=score_int, solve_for=solve_for
            )
        else:
            # score == 'spec'
            valid_pairs = update_spec(
                n=n, valid_pairs=valid_pairs, score_int=score_int, solve_for=solve_for
            )

        detail["n_tptn_pairs_after"] = len(valid_pairs)
        detail["decision"] = "continue" if len(valid_pairs) > 0 else "infeasible"
        details.append(detail)

        if len(valid_pairs) == 0:
            logger.info("no more feasible tp,tn pairs left")
            break

    total_count = sum(interval.integer_counts() for interval in valid_pairs.values())
    logger.info("constructing final tp, tn pair set")
    logger.info("final number of intervals: %d", len(valid_pairs))
    logger.info("final number of pairs: %d", total_count)

    return {
        "inconsistency": len(valid_pairs) == 0,
        "details": details,
        "n_valid_tptn_pairs": total_count,
        "iterate_by": iterate_by,
        "solve_for": solve_for,
        "evidence": (
            {
                iterate_by: list(valid_pairs.keys())[0],
                solve_for: valid_pairs[list(valid_pairs.keys())[0]].representing_int(),
            }
            if len(valid_pairs) > 0
            else None
        ),
    }


def check_all_negative_base(sols: list) -> bool:
    """
    Check if all solutions have negative base

    Args:
        sols (list(dict)): the list of solutions

    Returns:
        bool: True if all solutions have negative base, False otherwise
    """
    return all(sol.get("message") == "negative base" for sol in sols)


def check_any_zero_division(sols: list) -> bool:
    """
    Check if any solution has zero division

    Args:
        sols (list(dict)): the list of solutions

    Returns:
        bool: True if at least one solution has zero division, False otherwise
    """
    return any(sol.get("message") == "zero division" for sol in sols)


def update_tptn(tp, tn, sols: list):
    """
    Updates the tp and tn intervals based on the solutions

    Args:
        tp (Interval|IntervalUnion): the true positive interval
        tn (Interval|IntervalUnion): the true negative interval
        sols (list(dict)): the list of the solutions

    Returns:
        IntervalUnion, IntervalUnion: the updated tp and tn intervals
    """
    tp_union = IntervalUnion([sol["tp"] for sol in sols if sol["tp"] is not None])
    tn_union = IntervalUnion([sol["tn"] for sol in sols if sol["tn"] is not None])

    logger.info("the tp solutions: %s", tp_union)
    logger.info("the tn solutions: %s", tn_union)

    tp = tp.intersection(tp_union).shrink_to_integers()
    tn = tn.intersection(tn_union).shrink_to_integers()

    return tp, tn


def _check_scores_tptn_intervals(
    p: int,
    n: int,
    scores: dict,
    eps,
    *,
    numerical_tolerance: float = NUMERICAL_TOLERANCE
) -> dict:
    """
    Check scores by iteratively reducing the set of feasible ``tp``, ``tn`` pairs.

    Args:
        p (int): the number of positives
        n (int): the number of negatives
        scores (dict): the available reported scores
        eps (float|dict(str,float)): the numerical uncertainties for all scores or each
                                        score individually
        numerical_tolerance (float): the additional numerical tolerance

    Returns:
        dict: a summary of the results. When the ``inconsistency`` flag is True, it indicates
        that the set of feasible ``tp``, ``tn`` pairs is empty. The list under the key
        ``details`` provides further details from the analysis of the scores one after the other.
        Under the keys ``tp`` and ``tn`` one finds the final intervals or interval unions.
    """
    logger.info("checking the scores %s", str(scores))

    # resolving aliases and complements
    scores = resolve_aliases_and_complements(scores)

    # updating the uncertainties
    eps = eps if isinstance(eps, dict) else {score: eps for score in scores}
    eps = update_uncertainty(eps, numerical_tolerance)

    params = {
        "p": p,
        "n": n,
        "beta_positive": scores.get("beta_positive"),
        "beta_negative": scores.get("beta_negative"),
    }
    params |= create_intervals(scores, eps)

    tp = Interval(0, p)
    tn = Interval(0, n)

    details = []

    score_names = list(scores.keys())

    for idx, score0 in enumerate(score_names):
        for score1 in score_names[idx + 1 :]:
            detail = {
                "base_score_0": score0,
                "base_score_1": score1,
                "base_score_0_interval": params[score0].to_tuple(),
                "base_score_1_interval": params[score1].to_tuple(),
            }

            if tuple(sorted([score0, score1])) not in solution_specifications:
                logger.info("there is no solution for %s and %s", score0, score1)
                details.append(
                    detail
                    | {
                        "inconsistency": False,
                        "explanation": "there is no solution for the pair",
                    }
                )
                continue

            logger.info(
                "evaluating the tp and tn solution for %s and %s", score0, score1
            )

            sols = solution_specifications[(tuple(sorted([score0, score1])))].evaluate(
                params
            )

            if check_all_negative_base(sols):
                details.append(
                    detail
                    | {
                        "inconsistency": True,
                        "explanation": "all solutions lead to negative bases",
                    }
                )
                logger.info("all negative bases - iteration finished")
                break
            if check_any_zero_division(sols):
                details.append(
                    detail
                    | {
                        "inconsistency": False,
                        "explanation": "zero division indicates an "
                        "underdetermined system",
                    }
                )
                logger.info("all zero divisions - iteration continued")
                continue

            logger.info(
                "intervals before: %s, %s", str(tp.to_tuple()), str(tn.to_tuple())
            )
            tp, tn = update_tptn(tp, tn, sols)
            logger.info(
                "intervals after: %s, %s", str(tp.to_tuple()), str(tn.to_tuple())
            )

            details.append(
                detail
                | {
                    "tp_after": tp.to_tuple(),
                    "tn_after": tn.to_tuple(),
                    "inconsistency": (tp.is_empty()) and (tn.is_empty()),
                }
            )

        else:
            continue
        break

    return {
        "inconsistency": (tp.is_empty()) and (tn.is_empty()),
        "tp": tp.to_tuple(),
        "tn": tn.to_tuple(),
        "details": details,
    }


def check_scores_tptn_pairs(
    p: int,
    n: int,
    scores: dict,
    eps,
    *,
    numerical_tolerance: float = NUMERICAL_TOLERANCE,
    solve_for: str = None,
    prefilter_by_pairs: bool = False
) -> dict:
    """
    Check scores by iteratively reducing the set of feasible ``tp``, ``tn`` pairs.

    Args:
        p (int): the number of positives
        n (int): the number of negatives
        scores (dict): the available reported scores
        eps (float|dict(str,float)): the numerical uncertainties for all scores or each
                                        score individually
        numerical_tolerance (float): the additional numerical tolerance
        solve_for (str): the figure solving for (the other is used to iterate by) (``tp``/``tn``)
                        If None, the optimal one is being used.
        prefilter_by_pairs (bool): whether to prefilter the tp and tn intervals by the pairwise
                                    solutions

    Returns:
        dict: a summary of the results. When the ``inconsistency`` flag is True, it indicates
        that the set of feasible ``tp``, ``tn`` pairs is empty. The list under the key
        ``details`` provides further details from the analysis of the scores one after the other.
        Under the key ``n_valid_tptn_pairs`` one finds the number of tp and tn pairs compatible with
        all scores. Under the key ``prefiltering_details`` one finds the results of the prefiltering
        by using the solutions for the score pairs.
    """
    if not prefilter_by_pairs:
        return _check_scores_tptn_pairs(
            p,
            n,
            scores,
            eps,
            numerical_tolerance=numerical_tolerance,
            solve_for=solve_for,
        )
    results_interval = _check_scores_tptn_intervals(
        p, n, scores, eps, numerical_tolerance=numerical_tolerance
    )

    results = _check_scores_tptn_pairs(
        p,
        n,
        scores,
        eps,
        numerical_tolerance=numerical_tolerance,
        solve_for=solve_for,
        init_tptn_intervals={
            "tp": results_interval["tp"],
            "tn": results_interval["tn"],
        },
    )

    results["prefiltering_details"] = results_interval

    return results

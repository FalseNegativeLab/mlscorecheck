"""
This module implements the checking of scores with possible
tp and tn combinations.
"""

from ..core import NUMERICAL_TOLERANCE, logger

from ._interval import Interval, IntervalUnion
from ._check_score_pairs import resolve_aliases_and_complements, create_intervals
from ._tptn_solution_bundles import tptn_solutions, sens_tp, spec_tn
from ._solutions import solution_specifications

__all__ = ['check_scores_tptn', 'check_scores_tptn_interval']

preferred_order = ['acc', 'sens', 'spec', 'bacc', 'npv', 'ppv', 'f1p', 'f1m', 'fbp', 'fbm',
                    'fm', 'bm', 'pt', 'lrp', 'lrn', 'mk', 'dor',
                    'ji', 'gm', 'upm', 'p4', 'kappa', 'mcc']

def iterate_tptn(valid_pairs, sol_fun, params, iterate_by):
    """
    Iterate through the potential values of tp or tn and construct feasible pairs

    Args:
        to_iterate (list): the values to iterate through
        sol_fun (callable): the solution function providing an interval for ``tp`` if
                            ``variable`` is set to ``tn`` and vice versa.
        params (dict): the general parameters of the solution functions
        upper_bound (int): the upper bound for the variable ``sol_fun`` solves
        variable (str): ``tp``/``tn`` the variable to solve for

    Returns:
        set(tuple): the feasible (``tp``,``tn``) pairs
    """
    results = {}

    for value in valid_pairs:
        interval = sol_fun(**params, **{iterate_by: value})

        if interval is None:
            continue

        interval = interval.intersection(valid_pairs[value]).shrink_to_integers()

        if not interval.is_empty():
            results[value] = interval

    return results

def update_sens(p, valid_pairs, score_int, solve_for):
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
    ints = ints.intersection(Interval(0, p+1)).shrink_to_integers()

    if solve_for == 'tp':
        valid_pairs = {key: value.intersection(ints) for key, value in valid_pairs.items()}
        valid_pairs = {key: value for key, value in valid_pairs.items() if not value.is_empty()}
    else:
        valid_pairs = {value: interval for value, interval in valid_pairs.items()
                        if ints.contains(value)}

    return valid_pairs

def update_spec(n, valid_pairs, score_int, solve_for):
    """
    Update specificity intervals

    Args:
        n (int): the number of negatives
        valid_pairs (dict(int,Interval|IntervalUnion)): the actual intervals
        score_int (Interval): the score interval
        solve_for (str): 'tp'/'tn' - the figure to solve for

    Returns:
        dict(int,Interval|IntervalUnion): the updated intervals
    """
    ints = spec_tn(spec=score_int, n=n)
    ints = ints.intersection(Interval(0, n+1)).shrink_to_integers()

    if solve_for == 'tn':
        valid_pairs = {key: value.intersection(ints) for key, value in valid_pairs.items()}
        valid_pairs = {key: value for key, value in valid_pairs.items() if not value.is_empty()}
    else:
        valid_pairs = {value: interval for value, interval in valid_pairs.items()
                        if ints.contains(value)}

    return valid_pairs

def check_scores_tptn(p,
                        n,
                        scores,
                        eps,
                        *,
                        numerical_tolerance=NUMERICAL_TOLERANCE,
                        solve_for=None):
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
        that the set of feasible ``tp``,``tn`` pairs is empty. The list under the key
        ``details`` provides further details from the analysis of the scores one after the other.
    """
    if solve_for is None:
        solve_for = 'tn' if p < n else 'tp'

    if solve_for not in {'tp', 'tn'}:
        raise ValueError('The specified ``solve_for`` variable needs to be either '\
                            '``tp`` or ``tn``.')

    iterate_by = 'tn' if solve_for == 'tp' else 'tp'

    # resolving aliases and complements
    scores = resolve_aliases_and_complements(scores)

    # updating the uncertainties
    eps = eps if isinstance(eps, dict) else {score: eps for score in scores}
    eps = {key: value + numerical_tolerance for key, value in eps.items()}

    params = {'p': p, 'n': n,
                'beta_plus': scores.get('beta_plus'),
                'beta_minus': scores.get('beta_minus')}

    init_interval = Interval(0, n+1) if iterate_by == 'tp' else Interval(0, p+1)
    valid_pairs = {key: init_interval for key in range(p+1 if iterate_by == 'tp' else n+1)}

    details = []

    for score in [score for score in preferred_order if score in scores]:
        logger.info('testing %s, feasible tptn pairs: %d',
                    score, p*n if valid_pairs is None else len(valid_pairs))

        score_int = Interval(scores[score] - eps[score], scores[score] + eps[score])

        params[score] = score_int

        detail = {'testing': params[score],
                    'score_interval': score_int,
                    'n_tptn_pairs_before': p*n if valid_pairs is None else len(valid_pairs)}

        if score not in {'sens', 'spec'}:
            valid_pairs = iterate_tptn(valid_pairs=valid_pairs,
                                        sol_fun=tptn_solutions[score][solve_for],
                                        params=params,
                                        iterate_by=iterate_by)
        elif score == 'sens':
            valid_pairs = update_sens(p=p,
                                        valid_pairs=valid_pairs,
                                        score_int=score_int,
                                        solve_for=solve_for)
        else:
            # score == 'spec'
            valid_pairs = update_spec(n=n,
                                        valid_pairs=valid_pairs,
                                        score_int=score_int,
                                        solve_for=solve_for)

        detail['n_tptn_pairs_after'] = len(valid_pairs)
        detail['decision'] = 'continue' if len(valid_pairs) > 0 else 'infeasible'
        details.append(detail)

        if len(valid_pairs) == 0:
            logger.info('no more feasible tp,tn pairs left')
            break

    total_count = sum(interval.integer_counts() for interval in valid_pairs.values())
    logger.info('constructing final tp, tn pair set')
    logger.info('final number of intervals: %d', len(valid_pairs))
    logger.info('final number of pairs: %d', total_count)

    return {'inconsistency': len(valid_pairs) == 0,
            'details': details,
            'valid_tptn_pairs': total_count}

def check_scores_tptn_interval(p,
                                n,
                                scores,
                                eps,
                                *,
                                numerical_tolerance=NUMERICAL_TOLERANCE):
    # resolving aliases and complements
    scores = resolve_aliases_and_complements(scores)

    # updating the uncertainties
    eps = eps if isinstance(eps, dict) else {score: eps for score in scores}
    eps = {key: value + numerical_tolerance for key, value in eps.items()}

    params = {'p': p, 'n': n,
                'beta_plus': scores.get('beta_plus'),
                'beta_minus': scores.get('beta_minus')}
    intervals = create_intervals(scores, eps)

    tp = Interval(0, p)
    tn = Interval(0, n)

    details = []

    score_names = list(scores.keys())
    scores_used = {score: False for score in score_names}

    for idx, score0 in enumerate(score_names):
        for score1 in score_names[idx+1:]:
            detail = {'base_score_0': score0,
                        'base_score_1': score1}

            if tuple(sorted([score0, score1])) not in solution_specifications:
                logger.info(f'there is no solution for {score0} and {score1}')
                details.append(detail | {'inconsistency': False, 'explanation': 'there is no solution for the pair'})
                continue

            logger.info(f'evaluating the tp and tn solution for {score0} and {score1}')

            sols = solution_specifications[(tuple(sorted([score0, score1])))].evaluate(params | intervals)

            if all(sol.get('message') == 'negative base' for sol in sols):
                details.append(detail | {'inconsistency': True, 'explanation': 'all solutions lead to negative bases'})
                break
            if any(sol.get('message') == 'zero division' for sol in sols):
                details.append(detail | {'inconsistency': False, 'explanation': 'zero division indicates an underdetermined system'})
                continue

            scores_used[score0] = True
            scores_used[score1] = True

            tp_union = IntervalUnion([sol['tp'] for sol in sols if sol['tp'] is not None])
            tn_union = IntervalUnion([sol['tn'] for sol in sols if sol['tn'] is not None])

            logger.info(f'the tp solutions: {tp_union}')
            logger.info(f'the tn solutions: {tn_union}')

            tp = tp.intersection(tp_union)
            tn = tn.intersection(tn_union)
            tp = tp.shrink_to_integers()
            tn = tn.shrink_to_integers()

            details.append({'base_score_0': score0,
                            'base_score_1': score1,
                            'tp': tp_union.to_tuple(),
                            'tn': tn_union.to_tuple(),
                            'tp_after': tp.to_tuple(),
                            'tn_after': tn.to_tuple(),
                            'inconsistency': (tp.is_empty()) and (tn.is_empty())})

        else:
            continue
        break

    for score, flag in scores_used.items():
        if flag:
            continue

        logger.info('TODO')


    return {'inconsistency': (tp.is_empty()) and (tn.is_empty()),
            'details': details}
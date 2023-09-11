
def determine_edge_cases(score, p, n, beta_positive=None, beta_negative=None):
    """
    Determining the edge cases of a score

    Args:
        scores (dict(str,float)): the dictionary of scores
        p (int): the number of positives
        n (int): the number of negatives

    Returns:
        list(float): the list of edge case values the score can take
    """
    edge_cases = set()

    tp_cases = [{'tp': 0, 'fn': p}, {'tp': p, 'fn': 0}]
    tn_cases = [{'tn': 0, 'fp': n}, {'tn': n, 'fp': 0}]

    nans = score_specifications[score].get('nans')

    for arg0 in tp_cases:
        for arg1 in tn_cases:
            params = {**arg0, **arg1, 'p': p, 'n': n}
            if beta_positive is not None:
                params['beta_positive'] = beta_positive
            if beta_negative is not None:
                params['beta_negative'] = beta_negative
            edge_cases.add(safe_call(score_functions_standardized_all[score],
                                        {**params, 'sqrt': sqrt},
                                        nans))

    return list(edge_cases)

def check_individual_scores(scores, p, n, eps, numerical_tolerance=1e-6):
    """
    The main check functionality

    Args:
        scores (dict): the scores to check
        p (int): the number of positives
        n (int): the number of negatives
        eps (float|dict): the numerical uncertainty of the scores
        numerical_tolerance (float): in practice, beyond the numerical uncertainty of
                                    the scores, some further tolerance is applied. This is
                                    orders of magnitude smaller than the uncertainty of the
                                    scores. It does ensure that the specificity of the test
                                    is 1, it might slightly decrease the sensitivity.

    Returns:
        dict: the result of the check. The 'consistency' flag contains the
        overall decision, the 'succeeded' and 'failed' lists contain
        the details of the individual tests
    """
    check_uncertainty_and_tolerance(eps, numerical_tolerance)

    scores = resolve_aliases_and_complements(scores)
    scores_orig = scores

    scores = {key: value for key, value in scores.items() if key in supported_scores}

    problems = create_problems_2(list(scores.keys()))

    if 'beta_negative' in scores_orig:
        scores['beta_negative'] = scores_orig['beta_negative']
    if 'beta_positive' in scores_orig:
        scores['beta_positive'] = scores_orig['beta_positive']

    results = [check_2v1(scores, eps, problem, p, n,
                            numerical_tolerance=numerical_tolerance)
                for problem in problems]

    succeeded = []
    failed = []
    edge_scores = set()

    for result in results:
        edge_scores = edge_scores.union(set(result['edge_scores']))
        if result['inconsistency']:
            failed.append(result)
        else:
            succeeded.append(result)

    return {
        'tests_succeeded': succeeded,
        'tests_failed': failed,
        'underdetermined': len(failed) == 0 and all(tmp['underdetermined'] for tmp in succeeded),
        'edge_scores': list(edge_scores),
        'inconsistency': len(failed) > 0
        }

def check_zero_division(result):
    """
    Check if zero division occured in a particular case

    Args:
        result (dict): the dictionary of results

    Returns:
        None|dict: None if there is no zero division, some explanation otherwise
    """
    if result.get('message', None) == 'zero division':
        return {'inconsistency': False,
                'explanation': 'zero division indicates an underdetermined system'}
    return None

def check_negative_base(result):
    """
    Check if negative base occured in a particular case

    Args:
        result (dict): the dictionary of results

    Returns:
        None|dict: None if there is no negative base, some explanation otherwise
    """
    if result.get('message', None) == 'negative base':
        return {'inconsistency': True,
                'explanation': 'negative base indicates a non-suitable formula'}
    return None

def check_empty_interval(interval, name):
    """
    Check if the interval is empty

    Args:
        interval (Interval|IntervalUnion): the interval
        name (str): name of the variable the interval is determined for

    Returns:
        None|dict: None if the interval is not empty, some explanation otherwise
    """
    if interval.is_empty():
        return {'inconsistency': True,
                'explanation': f'the interval for {name} does not contain integers'}
    return None

def check_intersection(target, reconstructed):
    """
    Checks the intersection of the target score and the reconstructed interval

    Args:
        target (Interval|IntervalUnion): the interval
        reconstructed (Interval|IntervalUnion): the reconstructed interval

    Returns:
        dict: a dictionary containing the consistency decision and the explanation
    """

    if target.intersection(reconstructed).is_empty():
        return {'inconsistency': True,
                'explanation': f'the target score interval ({target}) and '\
                                f'the reconstructed intervals ({reconstructed}) '\
                                'do not intersect',
                'target_interval_reconstructed': reconstructed.to_tuple()}
    return {'inconsistency': False,
            'explanation': f'the target score interval ({target}) and '\
                                f'the reconstructed intervals ({reconstructed}) '\
                                'do intersect',
            'target_interval_reconstructed': reconstructed.to_tuple()}

def evaluate_1_solution(target_interval, result, p, n, score_function, beta_positive=None, beta_negative=None):
    """
    Carry out the evaluation for 1 particular solution

    Args:
        target_interval (Interval|IntervalUnion): the interval of the target score
        result (dict): the result of the evaluation
        p (int): the number of positives
        n (int): the number of negatives
        score_function (callable): the score function to be called

    Returns:
        dict: the dictionary of the result of the evaluation
    """

    if tmp := check_zero_division(result):
        return tmp

    if tmp := check_negative_base(result):
        return tmp

    if not isinstance(result['tp'], (Interval, IntervalUnion)):
        result['tp'] = Interval(result['tp'], result['tp'])
    if not isinstance(result['tn'], (Interval, IntervalUnion)):
        result['tn'] = Interval(result['tn'], result['tn'])

    tp = result['tp'].shrink_to_integers().intersection(Interval(0, p))
    tn = result['tn'].shrink_to_integers().intersection(Interval(0, n))

    if tmp := check_empty_interval(tp, 'tp'):
        return tmp

    if tmp := check_empty_interval(tn, 'tn'):
        return tmp

    score = safe_call(score_function, {**result, 'p': p, 'n': n, 'sqrt': sqrt, 'beta_positive': beta_positive, 'beta_negative': beta_negative})

    return check_intersection(target_interval, score)

def check_2v1(scores,
                eps,
                problem,
                p,
                n,
                *,
                numerical_tolerance=NUMERICAL_TOLERANCE):
    """
    Check one particular problem

    Args:
        scores (dict(str,float)): the scores
        eps (float|dict(str,float)): the numerical uncertainty(ies)
        problem (tuple(str,str,str)): the problem specification in the form
                                        (base_score0, base_score1, target_score)
        p (int): the number of positives
        n (int): the number of negatives
        numerical_tolerance (float): in practice, beyond the numerical uncertainty of
                                    the scores, some further tolerance is applied. This is
                                    orders of magnitude smaller than the uncertainty of the
                                    scores. It does ensure that the specificity of the test
                                    is 1, it might slightly decrease the sensitivity.

    Returns:
        dict: the result of the evaluation
    """
    # extracting the problem
    score0, score1, target = problem

    logger.info('checking %s and %s against %s', score0, score1, target)

    intervals = create_intervals({key: scores[key] for key in scores
                                    if key in problem},
                                    eps,
                                    numerical_tolerance=numerical_tolerance)
    if 'beta_negative' in scores:
        intervals['beta_negative'] = scores['beta_negative']
    if 'beta_positive' in scores:
        intervals['beta_positive'] = scores['beta_positive']

    # evaluating the solution
    if tuple(sorted([score0, score1])) not in solutions:
        return {'details': {'message': f'solutions for {score0} and {score1} are not available'},
                'edge_scores': [],
                'underdetermined': True,
                'inconsistency': False}

    results = solutions[tuple(sorted([score0, score1]))].evaluate({**intervals,
                                                                     **{'p': p, 'n': n}})

    output = []

    # iterating and evaluating all sub-solutions
    for result in results:
        res = {'score_0': score0,
                'score_0_interval': intervals[score0].to_tuple(),
                'score_1': score1,
                'score_1_interval': intervals[score1].to_tuple(),
                'target_score': target,
                'target_interval': intervals[target].to_tuple() if isinstance(target, (Interval, IntervalUnion)) else (intervals[target], intervals[target]),
                'solution': result}

        evaluation = evaluate_1_solution(intervals[target],
                                            result,
                                            p,
                                            n,
                                            functions_standardized[target],
                                            scores.get('beta_positive'),
                                            scores.get('beta_negative'))

        output.append({**res, **evaluation})

    # constructing the final output
    # there can be multiple solutions to a problem, if one of them is consistent,
    # the triplet is considered consistent
    return {'details': output,
            'edge_scores': list({key for key in [score0, score1]
                            if scores[key] in determine_edge_cases(key, p, n, scores.get('beta_positive'), scores.get('beta_negative'))}),
            'underdetermined': all(tmp.get('message') == 'zero division'
                                    for tmp in output),
            'inconsistency': all(tmp['inconsistency'] for tmp in output)}



@pytest.mark.parametrize("zeros", [[], ['tp'], ['tn'], ['fp'], ['fn'],
                                    ['tp', 'tn'], ['fp', 'fn'], ['tp', 'fp'], ['tn', 'fn']])
@pytest.mark.parametrize("random_state", random_seeds)
@pytest.mark.skip('deprecated')
def test_check_individual_scores(zeros, random_state):
    """
    Testing the check function
    """

    evaluation, _ = generate_1_problem(random_state=random_state,
                                                zeros=zeros,
                                                add_complements=True)
    evaluation['sqrt'] = sqrt
    evaluation['beta_positive'] = 2
    evaluation['beta_negative'] = 2

    score_values = {key: safe_call(functions[key], evaluation, scores[key].get('nans'))
                    for key in functions}
    score_values = {key: value for key, value in score_values.items() if value is not None}

    score_values['beta_positive'] = 2
    score_values['beta_negative'] = 2

    result = check_individual_scores(score_values, evaluation['p'], evaluation['n'], eps=1e-4)

    assert result['underdetermined'] or not result['inconsistency']

@pytest.mark.parametrize("zeros", [[], ['tp'], ['tn'], ['fp'], ['fn'],
                                    ['tp', 'fp'], ['tn', 'fn'], ['tp', 'tn'], ['fp', 'fn']])
@pytest.mark.parametrize("random_state", random_seeds)
@pytest.mark.skip('deprecated')
def test_check_failure(zeros, random_state):
    """
    Testing the failure
    """

    evaluation, _ = generate_1_problem(random_state=random_state,
                                                zeros=zeros,
                                                add_complements=True)
    evaluation['sqrt'] = sqrt
    evaluation['beta_positive'] = 2
    evaluation['beta_negative'] = 2

    score_values = {key: safe_call(functions[key], evaluation, scores[key].get('nans'))
                    for key in functions}
    score_values = {key: value for key, value in score_values.items() if value is not None}

    score_values['beta_positive'] = 2
    score_values['beta_negative'] = 2

    result = check_individual_scores(score_values,
                                        evaluation['p']*2,
                                        evaluation['n']+50,
                                        eps=1e-4)

    # at least two non-edge cases are needed to ensure the discovery of inconsistency
    edges = (len(score_values) - len(result['edge_scores'])) < 2

    assert edges or result['underdetermined'] or result['inconsistency']

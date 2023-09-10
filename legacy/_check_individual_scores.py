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

    if 'beta_minus' in scores_orig:
        scores['beta_minus'] = scores_orig['beta_minus']
    if 'beta_plus' in scores_orig:
        scores['beta_plus'] = scores_orig['beta_plus']

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
    evaluation['beta_plus'] = 2
    evaluation['beta_minus'] = 2

    score_values = {key: safe_call(functions[key], evaluation, scores[key].get('nans'))
                    for key in functions}
    score_values = {key: value for key, value in score_values.items() if value is not None}

    score_values['beta_plus'] = 2
    score_values['beta_minus'] = 2

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
    evaluation['beta_plus'] = 2
    evaluation['beta_minus'] = 2

    score_values = {key: safe_call(functions[key], evaluation, scores[key].get('nans'))
                    for key in functions}
    score_values = {key: value for key, value in score_values.items() if value is not None}

    score_values['beta_plus'] = 2
    score_values['beta_minus'] = 2

    result = check_individual_scores(score_values,
                                        evaluation['p']*2,
                                        evaluation['n']+50,
                                        eps=1e-4)

    # at least two non-edge cases are needed to ensure the discovery of inconsistency
    edges = (len(score_values) - len(result['edge_scores'])) < 2

    assert edges or result['underdetermined'] or result['inconsistency']

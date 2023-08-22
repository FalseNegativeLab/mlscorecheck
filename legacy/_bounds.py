def add_bounds_to_problems(problems,
                            problems_with_figures,
                            strategy=('mor', 'mor'),
                            bound_strategy=('min', 'min'),
                            bounds=('score', 'tptn')):
    """
    Adds bounds to problems

    Args:
        problems (list(dict)): a list of problem specifications
        problems_with_figures (list(dict)): a list of problems with random tp, tn figures
        strategy (tuple(str)): the aggregation strategy to be used
        bound_strategy (tuple(str)): the strategies to determine the bounds
        bounds (tuple(str)): the kinds of bounds to add ('score'/'tptn')

    Returns:
        list(dict): the datasets specifications with bounds
    """
    # expanding the datasets
    problems = _expand_datasets(problems)

    # calculating the scores and adding them to the datasets
    scores, evaluations = calculate_scores_datasets(problems_with_figures,
                                                    strategy=strategy,
                                                    return_populated=True)

    # initializing the scores
    min_scores = {'acc': 1, 'sens': 1, 'spec': 1, 'bacc': 1}
    min_tptn = {'tp': np.inf, 'tn': np.inf}

    for dataset in evaluations:
        for fold in dataset['folds']:
            for key, value in min_scores.items():
                if value < min_scores[key]:
                    min_scores[key] = value
            for key, value in min_tptn.items():
                if value < min_tptn[key]:
                    min_tptn[key] = value

    score_bounds = {key: (min_scores[key], 1) for key in min_scores}
    tptn_bounds = {key: (min_tptn[key], 1000000) for key in min_tptn}

    for dataset in problems:
        for fold in dataset['folds']:
            fold['score_bounds'] = copy.deepcopy(score_bounds)
            fold['tptn_bounds'] = copy.deepcopy(tptn_bounds)
        dataset['score_bounds'] = copy.deepcopy(score_bounds)
        dataset['tptn_bounds'] = copy.deepcopy(tptn_bounds)

    return problems

"""
This module implements the aggregation checks
"""

import copy

import pulp as pl

from ._calculate_scores import calculate_scores_datasets
from ._folds import _expand_datasets

__all__ = ['add_bounds',
            'calculate_scores_lp',
            '_initialize_folds',
            'create_lp_problem_inner',
            'create_target',
            'populate_solution',
            'check_aggregated_scores']

lp_scores = ['acc', 'sens', 'spec', 'bacc']

def add_bounds(pulp_problem, variables, bounds):
    """
    Adds bunds to the pulp problem

    Args:
        pulp_problem (pulp.LpProblem): the problem to add bounds to
        variables (dict): the variables to add bounds to
        bounds (dict(tuple)): the bounds for the variables
    """
    if bounds is None:
        return

    for key in bounds:
        pulp_problem += variables[key] <= bounds[key][1]
        pulp_problem += variables[key] >= bounds[key][0]

def calculate_scores_lp(*, tp, tn, p, n):
    """
    Calculates the scores applicable to the linear programming based check.
    The main difference compared to other score calculations is that the
    linear programming variables do not support division, so all scores need
    to be formulated in terms of multiplications by the inverses.

    Args:
        tp (pulp.LpVariable): the variable for true positives
        tn (pulp.LpVariable): the variable for true negatives
        p (int/float): the number of positives
        n (int/float): the number of negatives

    Returns:
        dict: the calculated scores
    """
    return {'acc': (tp + tn) * (1.0 / (p + n)),
            'sens': tp * (1.0/p),
            'spec': tn * (1.0/n),
            'bacc': (tp * (1.0/p) + tn * (1.0/n))*0.5}

def _initialize_folds(pulp_problem,
                    dataset,
                    dataset_idx):
    """
    Initializes the folds by creating and collecting the linear programming variables
    and adding constraints

    Args:
        pulp_problem (pl.LpProblem): the linear programming problem
        dataset (dict): the dataset
        dataset_idx (int): the index of the dataset

    Returns:
        list(dict(str:pl.LpVariable)): the variables for the folds
    """
    variables = []

    for idx, fold in enumerate(dataset['folds']):
        # iterating through all folds and creating the variables
        tp = pl.LpVariable(f'tp_{dataset_idx}_{idx}', 0, fold['p'], pl.LpInteger)
        tn = pl.LpVariable(f'tn_{dataset_idx}_{idx}', 0, fold['n'], pl.LpInteger)

        variables.append({'tp': tp, 'tn': tn, 'p': fold['p'], 'n': fold['n'],
                        **calculate_scores_lp(tp=tp, tn=tn, p=fold['p'], n=fold['n'])})

        # adding the constraints to the problem if specified
        add_bounds(pulp_problem, variables[-1], fold.get('score_bounds'))
        add_bounds(pulp_problem, variables[-1], fold.get('tptn_bounds'))

    return variables

def create_lp_problem_inner(pulp_problem,
                            dataset,
                            dataset_idx,
                            strategy):
    """
    Sets the problem for a dataset

    Args:
        pulp_problem (pl.LpProblem): the linear programming problem
        dataset (dict): the dataset specification
        dataset_idx (int): the index of the dataset
        strategy ('mor'/'rom'): the aggregation strategy

    Returns:
        list, dict: the list of all variables and the dict of the total sums and averages
    """

    variables = _initialize_folds(pulp_problem, dataset, dataset_idx)

    # calculating the totals
    total_variables = {'tp': sum(conf['tp'] for conf in variables),
                        'tn': sum(conf['tn'] for conf in variables),
                        'p': dataset['p'],
                        'n': dataset['n']}

    assert dataset['p'] == sum(conf['p'] for conf in variables)
    assert dataset['n'] == sum(conf['n'] for conf in variables)

    # depending on the strategy, determining the scores based on the totals
    # and adding the constraints to the totals
    if strategy == 'rom':
        # in the RoM case, the scores are calculated by the RoM scheme
        total_variables = {**total_variables,
                           **calculate_scores_lp(**total_variables)}
    elif strategy == 'mor':
        # in the MoR case, the scores are calculated by the MoR scheme
        for key in lp_scores:
            factor = 1.0 / len(variables)
            total_sum = sum(conf[key] for conf in variables)
            total_variables[key] = factor * total_sum

    # adding the bounds
    add_bounds(pulp_problem, total_variables, dataset.get('score_bounds'))
    add_bounds(pulp_problem, total_variables, dataset.get('tptn_bounds'))

    return variables, total_variables

def create_target(total_variables,
                    strategy):
    """
    Creates the total scores to be matched against the provided ones

    Args:
        total_variables (dict): the total sums and averages
        strategy ('rom'/'mor'): the aggregation strategy

    Returns:
        dict: the constructions to be matched with the scores
    """
    if strategy == 'rom':
        # the ratio of means variation
        sums = {key: sum(conf[key] for conf in total_variables)
                for key in ['tp', 'tn', 'p', 'n']}

        scores = calculate_scores_lp(**sums)
    if strategy == 'mor':
        # the mean of ratios variation
        scores = {}
        for key in lp_scores:
            # the simple average
            factor = 1.0 / len(total_variables)
            total_sum = sum(conf[key] for conf in total_variables)
            scores[key] = factor * total_sum

    return scores

def populate_solution(pulp_problem, problems, strategy, calculate_scores=True):
    """
    Populates the values of the variables into the problem structure

    Args:
        pulp_problem (pl.LpProblem): a solved problem
        problems (list(dict)): the original problems
        strategy (list('mor'/'rom')): the list of aggregation strategies

    Returns:
        list(dict): the problems with the variable values and scores populated
    """
    solutions = copy.deepcopy(problems)
    for variable in pulp_problem.variables():
        # extracts the name, the problem index and the fold index from the variable
        name, problem_idx, fold_idx = variable.name.split('_')
        problem_idx, fold_idx = int(problem_idx), int(fold_idx)
        solutions[problem_idx]['folds'][fold_idx][name] = variable.varValue

    if calculate_scores:
        solutions = calculate_scores_datasets(solutions,
                                                strategy=strategy,
                                                return_populated=True)[1]

    return solutions

def _check_bounds(problem):
    """
    Checking the bounds specified in the problem

    Args:
        problem (dict): a problem (dataset/fold) specification

    Returns:
        bool: True if the conditions are met
    """
    flag = True
    if 'score_bounds' in problem:
        problem['score_bounds_check'] = True
        for key, bounds in problem['score_bounds'].items():
            key_flag = bounds[0] <= problem[key] <= bounds[1]
            problem['score_bounds_check'] = problem['score_bounds_check'] and key_flag
            flag = flag and key_flag
    if 'tptn_bounds' in problem:
        problem['tptn_bounds_check'] = True
        for key, bounds in problem['tptn_bounds'].items():
            key_flag = bounds[0] <= problem[key] <= bounds[1]
            problem['tptn_bounds_check'] = problem['tptn_bounds_check'] and key_flag
            flag = flag and key_flag
    return flag

def _check_results(pulp_problem, problems, strategy):
    """
    Checking all the bound conditions in the results

    Args:
        pulp_problem (pl.LpProblem): the pl linear programming problem
        problems (list): problem/dataset specifications
        strategy (tuple(str)): the inner and outer aggregation strategies,
                                'mor'/'rom'
    Returns:
        dict: a summary of the results
    """
    solution = populate_solution(pulp_problem,
                                    problems,
                                    strategy,
                                    pulp_problem.status==1)

    if pulp_problem.status == 1:
        overall_check = True
        for dataset in solution:
            for fold in dataset['folds']:
                flag = _check_bounds(fold)
                overall_check = overall_check and flag
            flag = _check_bounds(dataset)
            overall_check = overall_check and flag
    else:
        overall_check = False

    return {'inconsistency': (pulp_problem.status != 1) or not overall_check,
            'pulp_solution': pulp_problem.status == 1,
            'bound_inconsistency': not overall_check,
            'configuration': solution}

def check_aggregated_scores(scores, eps, datasets, *, strategy, return_details=False):
    """
    Checks the consistency of the aggregated scores

    Args:
        scores (dict): the aggregated scores
        eps (dict/float): the numerical uncertainty
        datasets (list(dict)): the dataset specification to check against the scores
        strategy (list('mor'/'rom')): the list of aggregation strategies
        return_details (bool): whether to return the details of the check

    Returns:
        bool (, dict): the flag indicating the consistency and additionally the
                        details of the check
    """
    datasets = _expand_datasets(datasets)

    # creating the linear programming problem
    pulp_problem = pl.LpProblem('feasibility')

    # recording all variables and the totals for the datasets
    variables, total_variables = [], []

    for idx, problem in enumerate(datasets):
        # adding each dataset to the lp problem
        tmp = create_lp_problem_inner(pulp_problem, problem, idx, strategy[1])
        variables.append(tmp[0])
        total_variables.append(tmp[1])

    # creating the constructions to be matched with the published scores
    targets = create_target(total_variables, strategy[0])

    for key in scores:
        if key in lp_scores:
            # adding conditions for the numerical range of the supported scores
            epsilon = eps[key] if isinstance(eps, dict) else eps
            pulp_problem += targets[key] <= scores[key] + epsilon
            pulp_problem += targets[key] >= scores[key] - epsilon

    # adding an arbitrary variable as the objective of the problem
    pulp_problem += variables[0][0]['tp']

    # solving the problem
    pulp_problem.solve()

    details = _check_results(pulp_problem, datasets, strategy)

    # returning the results
    return ((details['inconsistency'], details) if return_details else
            details['inconsistency'])

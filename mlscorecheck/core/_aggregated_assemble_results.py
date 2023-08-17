"""
This module implements methods to assemble a detailed result for the
linear programming based consistency checks
"""

__all__ = [#'assemble_results',
            'assemble_results_problem',
            'assemble_results_problems']

#def assemble_results(pulp_problem):
#    """
#    Assembles a structured result for the problem
#
#    Args:
#        pulp_problem (pl.LpProblem): a solved pulp linear programming problem
#
#    Returns:
#        dict: the structured results
#    """
#
#    return {'overall_consistency': pulp_problem.status == 1,
#            'configuration': {var.name: var.varValue
#                                    for var in pulp_problem.variables()}}

def assemble_results_problem(pulp_problem, ps, ns):
    """
    Assembles a structured result for the problem

    Args:
        pulp_problem (pl.LpProblem): a solved pulp linear programming problem

    Returns:
        dict: the structured results
    """
    n_folds = int(len(pulp_problem.variables()) / 2)
    folds = [{'p': ps[idx], 'n': ns[idx]} for idx in range(n_folds)]

    for var in pulp_problem.variables():
        tokens = var.name.split('_')
        idx = int(tokens[1])
        folds[idx][tokens[0]] = var.varValue

    return {'overall_consistency': pulp_problem.status == 1,
            'configuration': folds}

def assemble_results_problems(pulp_problem, ps, ns, groups):
    """
    Assembles a structured result for the problem

    Args:
        pulp_problem (pl.LpProblem): a solved pulp linear programming problem

    Returns:
        dict: the structured results
    """
    problems = [[{} for _ in range(len(group))] for group in groups]

    for idx, var in enumerate(pulp_problem.variables()):
        tokens = var.name.split('_')
        pidx = int(tokens[1])
        fidx = int(tokens[2])
        problems[pidx][fidx][tokens[0]] = var.varValue

    idx = 0
    for pidx, group in enumerate(groups):
        for fidx, g in enumerate(group):
            problems[pidx][fidx]['p'] = ps[idx]
            problems[pidx][fidx]['n'] = ns[idx]
            idx += 1

    return {'overall_consistency': pulp_problem.status == 1,
            'configuration': problems}

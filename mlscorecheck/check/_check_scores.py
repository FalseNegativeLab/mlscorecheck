"""
This module implements the top level check functions
"""

from ..core import check
from ..core import consistency_1, consistency_grouped
from ..datasets import lookup_dataset, _resolve_pn
from ..core import determine_fold_configurations

__all__ = ['check_scores',
            'check_kfold_rom_scores',
            'check_multiple_datasets_rom_scores',
            'check_multiple_datasets_rom_kfold_rom_scores',
            'check_kfold_mor_scores',
            'check_multiple_datasets_mor_scores',
            'check_multiple_datasets_mor_kfold_rom_scores',
            'check_multiple_datasets_mor_kfold_mor_scores',
            '_check_problem_specification',
            'validate_problem_specification',
            'prepare_for_mor',
            'aggregate_problems',
            'expand_for_mor',
            'accumulate_dicts',
            'aggregate_dicts']

def _check_problem_specification(problem):
    """
    Check the problem specification
    """
    if isinstance(problem, dict):
        if not (('p' in problem and 'n' in problem) or ('dataset' in problem) or ('fold_configuration' in problem)):
            return False
        if 'fold_configuration' in problem:
            if not all('p' in fold and 'n' in fold for fold in problem['fold_configuration']):
                return False
        return True

    return all(_check_problem_specification(prob) for prob in problem)

def validate_problem_specification(problem):
    if not _check_problem_specification(problem):
        raise ValueError('the problem specification is incorrect')

def prepare_for_rom(problem):
    if isinstance(problem, dict):
        problem = _resolve_p_n(problem)
        if 'fold_configuration' in problem:
            p = 0
            n = 0
            for fold in problem['fold_configuration']:
                p += fold['p']
                n += fold['n']
            return {'p': p, 'n': n}
        else:
            return {'p': problem['p'] * problem.get('n_repeats', 1),
                    'n': problem['n'] * problem.get('n_repeats', 1)}
    else:
        resolved = [prepare_for_rom(subproblem) for subproblem in problem]
        return {'p': sum(tmp['p'] for tmp in resolved),
                'n': sum(tmp['n'] for tmp in resolved)}

def check_scores(scores,
                    eps,
                    dataset,
                    *,
                    return_details=False):
    """
    Use for one dataset - based on the solution formulas
    """
    validate_problem_specification(dataset)

    dataset = prepare_for_rom(dataset)

    results = check(scores, eps=eps, p=dataset['p'], n=dataset['n'])

    if return_details:
        return results['overall_consistency'], results

    return results['overall_consistency']

def check_kfold_rom_scores(scores,
                            eps,
                            dataset,
                            *,
                            return_details=False):
    """
    Use when scores are aggregated in the RoM manner
    """

    return check_scores(scores,
                        eps,
                        dataset=dataset,
                        return_details=return_details)

def check_multiple_datasets_rom_scores(scores,
                                        eps,
                                        datasets,
                                        *,
                                        return_details=False):

    return check_scores(scores,
                        eps,
                        datasets,
                        return_details=return_details)

def check_multiple_datasets_rom_kfold_rom_scores(scores,
                                                eps,
                                                datasets,
                                                *,
                                                return_details=False):
    """
    Use when scores are aggregated in the RoM manner
    """

    return check_scores(scores,
                            eps,
                            datasets,
                            return_details=return_details)



def expand_for_mor(problem):
    if isinstance(problem, dict):
        problem = _resolve_p_n(problem)
        if 'fold_configuration' in problem:
            result = problem['fold_configuration']
        else:
            result = determine_fold_configurations(problem['p'],
                                                    problem['n'],
                                                    problem.get('n_folds', 1),
                                                    problem.get('n_repeats', 1),
                                                    problem.get('folding', 'stratified_sklearn'))
            if 'score_bounds' in problem:
                for tmp in result:
                    tmp['score_bounds'] = {**problem['score_bounds']}
            if 'tptn_bounds' in problem:
                for tmp in result:
                    tmp['tptn_bounds'] = {**problem['tptn_bounds']}

        return result
    elif isinstance(problem, list):
        return [expand_for_mor(prob) for prob in problem]

def accumulate_tuples(tuple0, tuple1):
    result = []
    for idx in range(len(tuple0)):
        result.append(tuple0[idx] + tuple1[idx])

    return tuple(result)

def accumulate_dicts(dict0, dict1):
    result = {}
    for key in dict1:
        if key in dict0:
            if not isinstance(dict1[key], dict) and not isinstance(dict1[key], tuple):
                result[key] = dict0[key] + dict1[key]
            elif isinstance(dict1[key], tuple):
                result[key] = accumulate_tuples(dict0[key], dict1[key])
            else:
                result[key] = accumulate_dicts(dict0[key], dict1[key])
    return result

def aggregate_dicts(dicts, means=['score_bounds']):
    if not isinstance(dicts, list):
        return dicts
    if len(dicts) == 1:
        return dicts[0]
    result = accumulate_dicts(dicts[0], dicts[1])
    for tmp in dicts[2:]:
        result = accumulate_dicts(result, tmp)

    for key in means:
        if key in result:
            if isinstance(result[key], dict):
                for key2 in result[key]:
                    result[key][key2] = tuple([value/len(dicts) for value in result[key][key2]])
            elif isinstance(result[key], tuple):
                result[key] = tuple([value/len(dicts) for value in result[key]])
            else:
                result[key] = result[key] / len(dicts)

    return result

def aggregate_problems(problem):
    if isinstance(problem, list):
        p = 0
        n = 0
        for prob in problem:
            p += prob['p']
            n += prob['n']
        return {'p': p, 'n': n}
    return problem

def prepare_for_mor(problem, depth=None):
    problem = expand_for_mor(problem)

    if depth == 0:
        problem = [aggregate_dicts(prob) for prob in problem]
    if depth == 1:
        problem = [prepare_for_mor(prob, 0) for prob in problem]

    return problem

def check_kfold_mor_scores(scores,
                            eps,
                            dataset,
                            *,
                            return_details=False):
    """
    Use when a scores on a dataset are aggregated in the MoR manner

    By default, assumes stratified k-fold. If fold_setup is provided,
    it does overwrite all other arguments.
    """
    validate_problem_specification(dataset)

    dataset = _resolve_p_n(dataset)
    return consistency_1(scores=scores,
                            eps=eps,
                            problem=dataset,
                            return_details=return_details)

def check_multiple_datasets_mor_scores(scores,
                                        eps,
                                        datasets,
                                        *,
                                        return_details=False):
    validate_problem_specification(datasets)

    datasets = _resolve_p_n(datasets)
    return check_kfold_mor_scores(scores=scores,
                                    eps=eps,
                                    dataset={'fold_configuration': datasets},
                                    return_details=return_details)

def check_multiple_datasets_mor_kfold_rom_scores(scores,
                                                eps,
                                                datasets,
                                                *,
                                                return_details=False):
    validate_problem_specification(datasets)

    fold_configuration = []

    for dataset in datasets:
        dataset = _resolve_p_n(dataset)

        if 'fold_configuration' in dataset:
            fold_configuration.append({'p': sum(fold['p'] for fold in dataset['fold_configuration']),
                                        'n': sum(fold['n'] for fold in dataset['fold_configuration']),
                                        'score_bounds': dataset.get('score_bounds'),
                                        'tptn_bounds': dataset.get('tptn_bounds')})
        else:
            n_repeats = dataset['n_repeats']
            fold_configuration.append({'p': dataset['p']*n_repeats,
                                        'n': dataset['n']*n_repeats,
                                        'score_bounds': dataset.get('score_bounds'),
                                        'tptn_bounds': dataset.get('tptn_bounds')})

    return check_kfold_mor_scores(scores=scores,
                                    eps=eps,
                                    dataset={'fold_configuration': fold_configuration},
                                    return_details=return_details)

def check_multiple_datasets_mor_kfold_mor_scores(scores,
                                                eps,
                                                datasets,
                                                *,
                                                return_details=False):
    validate_problem_specification(datasets)

    datasets = _resolve_p_n(datasets)

    return consistency_grouped(scores=scores,
                                eps=eps,
                                problems=datasets,
                                return_details=return_details)


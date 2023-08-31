"""
This module implements some functionalities related to fold structures
"""

__all__ = ['stratified_configurations_sklearn',
            'determine_fold_configurations',
            '_create_folds']

def stratified_configurations_sklearn(p,
                                        n,
                                        n_splits):
    """
    The sklearn stratification strategy

    Args:
        p (int): number of positives
        n (int): number of negatives
        n_splits (int): the number of splits

    Returns:
        list(tuple): the list of the structure of the folds
    """
    p_base = p // n_splits
    n_base = n // n_splits
    p_remainder = p % n_splits
    n_remainder = n % n_splits

    results = [(n_base, p_base)] * n_splits

    idx = 0
    while n_remainder > 0:
        results[idx] = (results[idx][0] + 1, results[idx][1])
        n_remainder -= 1
        idx += 1
        idx %= n_splits
    while p_remainder > 0:
        results[idx] = (results[idx][0], results[idx][1] + 1)
        p_remainder -= 1
        idx += 1
        idx %= n_splits

    return results

def determine_fold_configurations(p,
                                    n,
                                    n_folds,
                                    n_repeats,
                                    folding='stratified_sklearn'):
    """
    Determine fold configurations according to a folding

    Args:
        p (int): the number of positives
        n (int): the number of negatives
        n_folds (int): the number of folds
        n_repeats (int): the number of repeats
        folding (str): 'stratified_sklearn' - the folding strategy

    Returns:
        list(dict): the list of folds

    Raises:
        ValueError: if the folding is not supported
    """
    if folding == 'stratified_sklearn':
        confs = stratified_configurations_sklearn(p=p, n=n, n_splits=n_folds)
        confs = [{'n': conf[0], 'p': conf[1]} for conf in confs]
        results = []
        for _ in range(n_repeats):
            for item in confs:
                results.append({**item})
    else:
        raise ValueError(f'folding strategy {folding} is not supported yet')

    return results

def _create_folds(p,
                    n,
                    *,
                    n_folds=None,
                    n_repeats=None,
                    folding=None,
                    score_bounds=None,
                    identifier=None):
    """
    Given a dataset, adds folds to it

    Args:
        p (int): the number of positives
        n (int): the number of negatives
        n_folds (int/None): the number of folds (defaults to 1)
        n_repeats (int|None): the number of repeats (defaults to 1)
        folding (str): the folding strategy ('stratified_sklearn')
        score_bounds (dict(str,tuple(float,float))): the score bounds
        identifier (str|None): the identifier

    Returns:
        list(dict): the list of fold specifications

    Raises:
        ValueError: if the folding is not supported
    """

    if n_folds == 1:
        folds = [{'p': p, 'n': n} for _ in range(n_repeats)]

    elif folding is None:
        folds = [{'p': p * n_repeats, 'n': n * n_repeats}]
    else:
        folds = determine_fold_configurations(p,
                                                n,
                                                n_folds,
                                                n_repeats,
                                                folding)
        n_fold = 0
        n_repeat = 0
        for _, fold in enumerate(folds):
            fold['identifier'] = f'{identifier}_{n_repeat}_{n_fold}'
            n_fold += 1
            if n_fold % n_folds == 0:
                n_fold = 0
                n_repeat += 1

    for _, fold in enumerate(folds):
        if score_bounds is not None:
            fold['score_bounds'] = {**score_bounds}

    return folds

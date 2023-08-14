"""
This module implements some functionalities related to fold structures
"""

__all__ = ['stratified_configurations',
            'determine_fold_configurations']

def stratified_configurations(n0, n1, n_splits):
    n0_base = n0 // n_splits
    n1_base = n1 // n_splits
    n0_remainder = n0 % n_splits
    n1_remainder = n1 % n_splits

    results = [(n0_base, n1_base)] * n_splits

    idx = 0
    while n0_remainder > 0:
        results[idx] = (results[idx][0] + 1, results[idx][1])
        n0_remainder -= 1
        idx += 1
        idx = idx % n_splits
    while n1_remainder > 0:
        results[idx] = (results[idx][0], results[idx][1] + 1)
        n1_remainder -= 1
        idx += 1
        idx = idx % n_splits

    return results

def determine_fold_configurations(p, n, n_folds, n_repeats):
    confs = stratified_configurations(p, n, n_folds)
    confs = [{'p': conf[0], 'n': conf[1]} for conf in confs]
    results = []
    for _ in range(n_repeats):
        for item in confs:
            results.append({**item})

    return results


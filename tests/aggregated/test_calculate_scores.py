"""
Testing the calculation of aggregated scores
"""

import numpy as np

from mlscorecheck.individual import calculate_scores

from mlscorecheck.aggregated import (calculate_scores_list,
                                        calculate_scores_dataset,
                                        calculate_scores_datasets,
                                        generate_1_problem_with_evaluations,
                                        generate_problems_with_evaluations)

def test_calculate_scores_list():
    """
    Testing the calculate_scores_list function
    """

    datasets = [{'p': 5, 'n': 10, 'tp': 2, 'tn': 8},
                {'p': 8, 'n': 20, 'tp': 4, 'tn': 5}]

    datasets = [calculate_scores(dataset, scores_only=False) for dataset in datasets]

    scores = calculate_scores_list(datasets, strategy='rom')

    assert scores['acc'] == (2 + 4 + 8 + 5) / (5 + 8 + 10 + 20)

    scores = calculate_scores_list(datasets, strategy='mor')

    assert scores['acc'] == (datasets[0]['acc'] + datasets[1]['acc']) / 2

def test_calculate_scores_dataset():
    """
    Testing the score calculation for a dataset
    """

    figures, problem = generate_1_problem_with_evaluations(n_repeats=2,
                                                            n_folds=3,
                                                            random_state=5)

    scores = calculate_scores_dataset(figures,
                                        strategy='rom')

    tp = sum(fold['tp'] for fold in figures['folds'])
    tn = sum(fold['tn'] for fold in figures['folds'])
    p = problem['p']
    n = problem['n']

    assert scores['acc'] == (tp + tn) / (p + n)

    scores, populated = calculate_scores_dataset(figures,
                                        strategy='mor',
                                        return_populated=True)

    assert scores['acc'] == np.mean([fold['acc'] for fold in populated['folds']])

def test_calculate_scores_datasets():
    """
    Testing the score calculation for datasets
    """

    figures, _ = generate_problems_with_evaluations(n_problems=2,
                                                    n_repeats=2,
                                                    n_folds=3,
                                                    random_state=5)

    for strategy in [('mor', 'mor'), ('mor', 'rom'), ('mor', 'mor')]:
        outer = []
        for dataset in figures:
            inner = []
            for fold in dataset['folds']:
                inner.append(calculate_scores(fold))
            outer.append(calculate_scores_list(inner, strategy=strategy[1]))
        scores = calculate_scores_list(outer, strategy=strategy[0])

        scores_orig = calculate_scores_datasets(figures, strategy=strategy)

        assert scores['acc'] == scores_orig['acc']

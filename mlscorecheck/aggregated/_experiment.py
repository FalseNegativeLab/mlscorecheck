"""
This module implements an abstraction for an experiment
"""

import numpy as np

from ._dataset import Dataset
from ._linear_programming import check_bounds

__all__ = ['Experiment']

class Experiment:
    def __init__(self,
                    *,
                    name=None,
                    datasets=None,
                    aggregation=None,
                    figures=None,
                    scores=None):
        self.name = name
        self.datasets = datasets
        self.aggregation = aggregation

        self.figures = figures
        self.scores = scores

        self.linear_programming = None

        self.initialize_datasets()

    def to_dict(self, raw_problem=False):
        results = {'name': self.name,
                    'aggregation': self.aggregation,
                    'datasets': [dataset.to_dict(raw_problem) for dataset in self.datasets]}

        if raw_problem:
            return results

        return {**results,
                'figures': self.figures,
                'scores': self.scores}

    def initialize_datasets(self):
        if not isinstance(self.datasets[0], Dataset):
            self.datasets = [Dataset(**dataset) for dataset in self.datasets]

    def sample(self, random_state=None):
        if random_state is None or not isinstance(random_state, np.random.RandomState):
            random_state = np.random.RandomState(random_state)

        for dataset in self.datasets:
            dataset.sample(random_state=random_state)

        self.figures = {'tp': sum(dataset.figures['tp'] for dataset in self.datasets),
                        'tn': sum(dataset.figures['tn'] for dataset in self.datasets),
                        'p': sum(dataset.figures['p'] for dataset in self.datasets),
                        'n': sum(dataset.figures['n'] for dataset in self.datasets)}

        return self

    def calculate_scores(self):
        for dataset in self.datasets:
            dataset.calculate_scores()

        if self.aggregation == 'rom':
            self.scores = calculate_scores_for_lp(self.figures)
        else:
            self.scores = {key: np.mean([dataset.scores[key] for dataset in self.datasets]) for key in ['acc', 'sens', 'spec', 'bacc']}

        return self.scores

    def init_lp(self, lp_program):
        for dataset in self.datasets:
            dataset.init_lp(lp_program)

        self.linear_programming = {'tp': sum(dataset.linear_programming['tp'] for dataset in self.datasets),
                                    'tn': sum(dataset.linear_programming['tn'] for dataset in self.datasets),
                                    'p': sum(dataset.linear_programming['p'] for dataset in self.datasets),
                                    'n': sum(dataset.linear_programming['n'] for dataset in self.datasets)}

        if self.aggregation == 'rom':
            self.linear_programming = calculate_scores_for_lp({**self.linear_programming})

        elif self.aggregation == 'mor':
            for key in ['acc', 'sens', 'spec', 'bacc']:
                self.linear_programming[key] = sum(dataset.linear_programming[key] for dataset in self.datasets) * (1.0 / len(self.datasets))

        return lp_program

    def populate_with_solution(self, lp_program):
        for dataset in self.datasets:
            dataset.populate_with_solution(lp_program)

        self.figures = {'tp': sum(dataset.figures['tp'] for dataset in self.datasets),
                        'tn': sum(dataset.figures['tn'] for dataset in self.datasets),
                        'p': sum(dataset.figures['p'] for dataset in self.datasets),
                        'n': sum(dataset.figures['n'] for dataset in self.datasets)}

        return self

    def check_bounds(self):
        downstream = [dataset.check_bounds() for dataset in self.datasets]
        flag = all(tmp['check_score_bounds'] and tmp['check_tptn_bounds'] and tmp['downstream_bounds'] for tmp in downstream)

        result = {'figures': self.figures,
                    'scores': self.scores,
                    'downstream_bounds': flag,
                    'datasets': downstream}

        return result

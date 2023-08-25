"""
This module implements an abstraction for an experiment
"""

import numpy as np

from ..core import logger
from ..individual import calculate_scores_for_lp

from ._dataset import (Dataset,
                        generate_dataset_specification)
from ._linear_programming import add_bounds, check_bounds

__all__ = ['Experiment',
            'generate_experiment_specification']

def generate_experiment_specification(max_n_datasets=10,
                                        max_p=1000,
                                        max_n=1000,
                                        max_n_folds=10,
                                        max_n_repeats=5,
                                        random_state=None):
    if random_state is None or not isinstance(random_state, np.random.RandomState):
        random_state = np.random.RandomState(random_state)

    n_datasets = random_state.randint(1, max_n_datasets+1)
    datasets = [generate_dataset_specification(max_p=max_p,
                                                max_n=max_n,
                                                max_n_folds=max_n_folds,
                                                max_n_repeats=max_n_repeats,
                                                random_state=random_state)
                for _ in range(n_datasets)]
    aggregation = random_state.choice(['rom', 'mor'])

    return {'aggregation': aggregation,
            'datasets': datasets}

class Experiment:
    def __init__(self,
                    *,
                    aggregation,
                    id=None,
                    datasets=None):
        """
        Constructor of the experiment object

        Args:
            aggregation (str): the aggregation strategy
            id (None/str): the id of the experiment
            datasets (list(dict)): the dataset specifications
        """
        self.id = id
        self.datasets = datasets

        if aggregation not in {'rom', 'mor'}:
            raise ValueError(f'aggregation {aggregation} is not supported yet')

        self.aggregation = aggregation

        self.linear_programming = None

        self.initialize_datasets()

    def to_dict(self, problem_only=False):
        """
        Returns a dict representation

        Args:
            problem_only (bool): whether to return the problem only (True) or add the
                                figuress and scores (False)

        Returns:
            dict: the dict representation
        """
        return {'id': self.id,
                    'aggregation': self.aggregation,
                    'datasets': [dataset.to_dict(problem_only) for dataset in self.datasets]}

    def __repr__(self):
        """
        Returning a string representation

        Returns:
            str: the string representation
        """
        return str(self.to_dict())

    def initialize_datasets(self):
        """
        Initialize all datasets
        """
        if not isinstance(self.datasets[0], Dataset):
            self.datasets = [Dataset(**dataset) for dataset in self.datasets]

    def sample(self, random_state=None):
        """
        Samples the problem, that is, generates random (but physical) tp and tn values

        Args:
            random_state (None/int/np.random.RandomState): the random state to use

        Returns:
            self: the sampled dataset
        """
        if random_state is None or not isinstance(random_state, np.random.RandomState):
            random_state = np.random.RandomState(random_state)

        datasets = [dataset.sample(random_state=random_state).to_dict(problem_only=False) for dataset in self.datasets]
        params = self.to_dict(problem_only=True)
        params['datasets'] = datasets

        return Experiment(**params)

    def calculate_figures(self):
        figures = {'tp': 0, 'tn': 0, 'p': 0, 'n': 0}
        for dataset in self.datasets:
            tmp = dataset.calculate_figures()
            figures['tp'] += tmp['tp']
            figures['tn'] += tmp['tn']
            figures['p'] += tmp['p']
            figures['n'] += tmp['n']
        return figures

    def calculate_scores(self, score_subset=None, rounding_decimals=None):
        """
        Calculates all scores for the fold

        Returns:
            dict(str,float): the scores
            rounding_decimals (None/float): how many digits to round the decimals to
        """

        score_subset = ['acc', 'sens', 'spec', 'bacc'] if score_subset is None else score_subset

        figures = self.calculate_figures()

        if self.aggregation == 'rom':
            scores = calculate_scores_for_lp(figures)
            scores = {key: scores[key] for key in score_subset}
        else:
            scores = {key: 0.0 for key in score_subset}
            for dataset in self.datasets:
                dataset_scores = dataset.calculate_scores(score_subset)
                for key in dataset_scores:
                    scores[key] += dataset_scores[key]

            for key in scores:
                scores[key] /= len(self.datasets)

        if rounding_decimals is not None:
            self.scores = {key: np.round(value, rounding_decimals)
                            for key, value in scores.items()}

        return scores

    def get_bounds(self, score_subset=None, feasible=True):
        scores = self.calculate_scores(score_subset)

        if feasible:
            score_bounds = {key: (scores[key] - 1e-3, scores[key] + 1e-3) for key in scores}
        else:
            score_bounds = {}
            for key in scores:
                if scores[key] > 0.2:
                    score_bounds[key] = (0.0, max(scores[key] - 2*1e-2, 0))
                else:
                    score_bounds[key] = (scores[key] + 2*1e-2, 1.0)

        return score_bounds

    def get_dataset_bounds(self, score_subset=None, feasible=True):
        return [dataset.get_bounds(score_subset, feasible) for dataset in self.datasets]

    def add_dataset_bounds(self, score_bounds):
        if isinstance(score_bounds, dict):
            score_bounds = [score_bounds] * len(self.datasets)

        return Experiment(id=self.id,
                            aggregation=self.aggregation,
                            datasets=[dataset.add_bounds(s_bounds).to_dict(problem_only=False)
                                    for dataset, s_bounds in zip(self.datasets, score_bounds)])

    def init_lp(self, lp_problem):
        """
        Initializes the linear programming problem for the dataset

        Args:
            lp_problem (pl.LpProblem): a linear programming problem by pulp

        Returns:
            pl.LpProblem: the updated linear programming problem
        """
        for dataset in self.datasets:
            dataset.init_lp(lp_problem)

        self.linear_programming = {'tp': sum(dataset.linear_programming['tp'] for dataset in self.datasets),
                                    'tn': sum(dataset.linear_programming['tn'] for dataset in self.datasets),
                                    'p': sum(dataset.linear_programming['p'] for dataset in self.datasets),
                                    'n': sum(dataset.linear_programming['n'] for dataset in self.datasets)}

        if self.aggregation == 'rom':
            self.linear_programming = {**self.linear_programming,
                                       **calculate_scores_for_lp({**self.linear_programming})}

        elif self.aggregation == 'mor':
            for key in ['acc', 'sens', 'spec', 'bacc']:
                self.linear_programming[key] = sum(dataset.linear_programming[key] for dataset in self.datasets) * (1.0 / len(self.datasets))

        return lp_problem

    def populate(self, lp_problem):
        """
        Populates the object by the elements of the solved/unsolved linear programming
        problem as a figures of tp and tn

        Args:
            lp_problem (pl.LpProblem): the linear programming problem after running the solve
                                        method

        Returns:
            self: the updated object
        """
        return Experiment(aggregation=self.aggregation,
                            id=self.id,
                            datasets=[dataset.populate(lp_problem).to_dict(problem_only=False) for dataset in self.datasets])

    def check_bounds(self):
        """
        Checks if the boundary conditions hold and returns a summary.
        The 'all_bounds' flag indicates the result of bound checks
        for each dataset.

        Returns:
            dict: a summary of the evaluation of the boundary conditions
        """

        downstream = [dataset.check_bounds() for dataset in self.datasets]
        flag = all(tmp['bounds_flag'] for tmp in downstream)

        return {'id': self.id,
                'figures': self.calculate_figures(),
                'scores': self.calculate_scores(),
                'bounds_flag': flag,
                'datasets': downstream}

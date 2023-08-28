"""
This module implements an abstraction for an experiment
"""

import numpy as np
import pulp as pl

from ..core import init_random_state, dict_mean, round_scores, dict_minmax
from ..individual import calculate_scores_for_lp

from ._dataset import (Dataset,
                        generate_dataset_specification)
from ._utils import aggregated_scores

__all__ = ['Experiment',
            'generate_experiment_specification']

def generate_experiment_specification(*,
                                        max_n_datasets=10,
                                        max_p=1000,
                                        max_n=1000,
                                        max_n_folds=10,
                                        max_n_repeats=5,
                                        random_state=None,
                                        aggregation=None,
                                        aggregation_ds=None):
    """
    Generate a random experiment specification

    Args:
        max_n_datasets (int): the maximum number of datasets
        max_p (int): the maximum number of positives
        max_n (int): the maximum number of negatives
        max_n_folds (int): the maximum number of folds
        max_n_repeats (int): the maximum number of repeats
        random_state (int/np.random.RandomState/None): the random state to use
        aggregation (str/None): 'mor'/'rom' - the aggregation of the experiment
        aggregation_ds (str/None): 'mor'/'rom' - the aggregation in the datasets
    """
    random_state = init_random_state(random_state)

    n_datasets = random_state.randint(1, max_n_datasets+1)
    datasets = [generate_dataset_specification(max_p=max_p,
                                                max_n=max_n,
                                                max_n_folds=max_n_folds,
                                                max_n_repeats=max_n_repeats,
                                                random_state=random_state,
                                                aggregation=aggregation_ds)
                for _ in range(n_datasets)]
    aggregation = (random_state.choice(['rom', 'mor'])
                    if aggregation is None else aggregation)

    return {'aggregation': aggregation,
            'datasets': datasets}

class Experiment:
    """
    The experiment class represents an experiment involving multiple datasets
    """
    def __init__(self,
                    *,
                    aggregation,
                    datasets=None,
                    identifier=None):
        """
        Constructor of the experiment object

        Args:
            aggregation (str): 'mor'/'rom' - the aggregation strategy
            datasets (list(dict)): the dataset specifications
            identifier (None/str): the id of the experiment
        """
        self.identifier = identifier
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
        return {'identifier': self.identifier,
                'aggregation': self.aggregation,
                'datasets': [dataset.to_dict(problem_only) for dataset in self.datasets]}

    def __repr__(self):
        """
        Returning a string representation

        Returns:
            str: the string representation
        """
        return str(self.to_dict())

    def has_downstream_bounds(self):
        """
        Checks if the experiment has score bounds specified

        Returns:
            bool: a flag indicating if the experiment has score bounds specified on
                    datasets or folds
        """
        return any(dataset.has_downstream_bounds() or dataset.score_bounds is not None
                    for dataset in self.datasets)

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
        random_state = init_random_state(random_state)

        return Experiment(identifier=self.identifier,
                    aggregation=self.aggregation,
                    datasets=[dataset.sample(random_state=random_state).to_dict(problem_only=False)
                                for dataset in self.datasets])

    def calculate_figures(self):
        """
        Calculates the raw figures tp, tn, p and n

        Returns:
            dict: the raw figures
        """
        figures = self.datasets[0].calculate_figures()

        for dataset in self.datasets[1:]:
            tmp = dataset.calculate_figures()
            for key in figures:
                figures[key] += tmp[key]

        return figures

    def calculate_scores(self, score_subset=None, rounding_decimals=None):
        """
        Calculates all scores for the fold

        Args:
            score_subset (None/list): the subset of scores to calculate
            rounding_decimals (None/float): how many digits to round the decimals to

        Returns:
            dict(str,float): the scores
        """

        score_subset = aggregated_scores if score_subset is None else score_subset

        figures = self.calculate_figures()

        if self.aggregation == 'rom':
            scores = calculate_scores_for_lp(figures, score_subset)
        else:
            scores = dict_mean([dataset.calculate_scores(score_subset)
                                        for dataset in self.datasets])

        return round_scores(scores, rounding_decimals)

    def get_minmax_bounds(self, score_subset=None):
        """
        Extract min-max bounds for scores in the datasets

        Args:
            score_subset (list/None): the subset of scores to extract bounds for

        Returns:
            dict (str,list(float,float)): the minimum and maximum values
        """

        score_bounds = dict_minmax([dataset.calculate_scores(score_subset)
                                    for dataset in self.datasets])

        return {key: [value[0] - 1e-3, value[1] + 1e-3] for key, value in score_bounds.items()}

    def get_dataset_bounds(self, score_subset=None, feasible=True):
        """
        Extract bounds for the datasets

        Args:
            score_subset (list/None): the subset of scores to extract bounds for
            feasible (bool): if True, the bounds will be feasible, else unfeasible

        Returns:
            list(dict): the list of bounds
        """
        return [dataset.get_bounds(score_subset, feasible) for dataset in self.datasets]

    def get_dataset_fold_bounds(self, score_subset=None, feasible=True):
        """
        Extract bounds for the folds of the datasets

        Args:
            score_subset (list/None): the subset of scores to extract bounds for
            feasible (bool): if True, the bounds will be feasible, else unfeasible

        Returns:
            list(list(dict)): the list of bounds
        """
        return [[fold.get_bounds(score_subset, feasible) for fold in dataset.folds]
                                                        for dataset in self.datasets]

    def add_dataset_fold_bounds(self, score_bounds):
        """
        Adds bounds to each fold of the datasets

        Args:
            score_bounds (list(list(dict))): the list of bounds

        Returns:
            Experiment: the updated experiment object
        """
        return Experiment(identifier=self.identifier,
                            aggregation=self.aggregation,
                            datasets=[dataset.add_fold_bounds(bounds).to_dict(problem_only=False)
                                        for dataset, bounds in zip(self.datasets, score_bounds)])

    def add_dataset_bounds(self, score_bounds):
        """
        Adds bounds to each dataset

        Args:
            score_bounds (dict/list(dict)): the bounds to add

        Returns:
            Experiment: the updated experiment object
        """
        if isinstance(score_bounds, dict):
            score_bounds = [score_bounds] * len(self.datasets)

        return Experiment(identifier=self.identifier,
                            aggregation=self.aggregation,
                            datasets=[dataset.add_bounds(s_bounds).to_dict(problem_only=False)
                                    for dataset, s_bounds in zip(self.datasets, score_bounds)])

    def init_lp(self, lp_problem, scores):
        """
        Initializes the linear programming problem for the dataset

        Args:
            lp_problem (pl.LpProblem): a linear programming problem by pulp
            scores (dict(str,float)): the scores are passed to provide a bases
                                        for the initial values of the variables

        Returns:
            pl.LpProblem: the updated linear programming problem
        """
        for dataset in self.datasets:
            dataset.init_lp(lp_problem, scores)

        self.linear_programming = {'tp': pl.lpSum(dataset.linear_programming['tp']
                                                    for dataset in self.datasets),
                                    'tn': pl.lpSum(dataset.linear_programming['tn']
                                                    for dataset in self.datasets),
                                    'p': sum(dataset.linear_programming['p']
                                                for dataset in self.datasets),
                                    'n': sum(dataset.linear_programming['n']
                                                for dataset in self.datasets)}

        if self.aggregation == 'rom':
            self.linear_programming = {**self.linear_programming,
                                       **calculate_scores_for_lp({**self.linear_programming})}

        elif self.aggregation == 'mor':
            norm = 1.0 / len(self.datasets)
            for key in aggregated_scores:
                self.linear_programming[key] = (pl.lpSum(dataset.linear_programming[key]
                                                         for dataset in self.datasets) * norm)

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
                            identifier=self.identifier,
                            datasets=[dataset.populate(lp_problem).to_dict(problem_only=False)
                                        for dataset in self.datasets])

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

        return {'identifier': self.identifier,
                'figures': self.calculate_figures(),
                'scores': self.calculate_scores(),
                'bounds_flag': flag,
                'datasets': downstream}

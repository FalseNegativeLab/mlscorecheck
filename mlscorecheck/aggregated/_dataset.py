"""
This module implements an abstraction for a dataset
"""

import numpy as np

from ..individual import calculate_scores_for_lp
from ..core import logger
from ..experiments import lookup_dataset

from ._fold import (Fold, random_identifier, generate_fold_specification)
from ._folds import _create_folds_pure
from ._linear_programming import add_bounds, check_bounds

from ..experiments import dataset_statistics

__all__ = ['Dataset',
            'generate_dataset_specification']

def generate_dataset_specification(max_p=1000,
                                    max_n=1000,
                                    max_n_folds=10,
                                    max_n_repeats=5,
                                    random_state=None):
    if random_state is None or not isinstance(random_state, np.random.RandomState):
        random_state = np.random.RandomState(random_state)

    aggregation = random_state.choice(['mor', 'rom'])
    id = random_state.choice([None, random_identifier(8)])

    spec_type = random_state.randint(3)
    if spec_type == 0:
        p = random_state.randint(1, max_p+1)
        n = random_state.randint(1, max_n+1)
        n_folds = random_state.randint(1, min(p+1, n+1, max_n_folds))
        n_repeats = random_state.randint(1, max_n_repeats+1)
        return {'p': p, 'n': n, 'n_folds': n_folds, 'n_repeats': n_repeats, 'aggregation': aggregation, 'id': id}
    elif spec_type == 1:
        name = random_state.choice(list(dataset_statistics.keys()))
        details = dataset_statistics[name]
        p = details['p']
        n = details['n']
        n_folds = random_state.randint(1, min(p+1, n+1, max_n_folds))
        n_repeats = random_state.randint(1, max_n_repeats+1)
        return {'name': name, 'n_folds': n_folds, 'n_repeats': n_repeats, 'aggregation': aggregation, 'id': id}
    elif spec_type == 2:
        n_folds = random_state.randint(1, max_n_folds+1) * random_state.randint(1, max_n_repeats+1)
        folds = [generate_fold_specification(random_state=random_state) for _ in range(n_folds)]
        return {'folds': folds, 'aggregation': aggregation, 'id': id}

class Dataset:
    """
    An abstraction for a dataset
    """
    def __init__(self,
                    *,
                    aggregation,
                    id=None,
                    name=None,
                    p=None,
                    n=None,
                    folds=None,
                    n_folds=None,
                    n_repeats=None,
                    folding='stratified_sklearn',
                    score_bounds=None,
                    fold_score_bounds=None):
        """
        Constructor of the dataset. Specify either p and n or a name
        to look up the p and n statistics. The name must be in the supported
        format like 'common_datasets.ADA'. Alternatively, one can pass
        a list of fold specifications.

        Examples:
            ds0 = Dataset(p=5, n=10, aggregation='rom')
            ds1 = Dataset(name='common_datasets.ADA', aggregation='mor')
            ds2 = Dataset(folds=[{p=5, n=10}, {p=2, n=8}], aggregation='rom')

        Args:
            aggregation (str): 'rom'/'mor - the aggregation strategy
            id (None/str): the identifier
            name (None/str): the name of the dataset to look-up
            p (None/int): the number of positives
            n (None/int): the number of negatives
            folds (None/list): the list of fold specifications
            n_folds (None/int): the number of folds
            n_repeats (None/int): the number of repetitions
            folding (str): the folding strategy
            score_bounds (None/dict(str,tuple)): the bound specification for scores
            fold_score_bounds (None/dict(str,tuple)): the bound specification for scores in
                                                        the folds
            figures (None/dict(str,int)): the already computed p, n, tp, and tn figures
        """

        # checking if the dataset is specified properly
        folds_provided = folds is not None
        pn_provided = p is not None and n is not None
        folds_repeats_provided = n_folds is not None and n_repeats is not None
        name_provided = name is not None

        if not ((folds_provided and not folds_repeats_provided and not name_provided and not pn_provided)\
            or (pn_provided and not folds_provided and not name_provided)\
            or (name_provided and not folds_provided and not pn_provided)):
                raise ValueError('Please specify either folds (without p, n, n_folds, n_repeats, name) '\
                                    'or p and n (without folds, name) '\
                                    'or name (without p, n, folds)')

        # the id of the dataset is set to the name or a random id is generated
        if id is None and name is not None:
            self.id = name.replace('-', '_')
        elif id is not None:
            self.id = id
        else:
            logger.info('generating a random id for the dataset')
            self.id = random_identifier(16)

        # the folds are generated
        if pn_provided or name_provided:
            if name_provided:
                logger.info('querying p and n from looking up the dataset')
                tmp = lookup_dataset(name)
                p, n = tmp['p'], tmp['n']
            logger.info('creating a folding based on the specification')
            folds = _create_folds_pure(p, n, n_folds, n_repeats, folding, fold_score_bounds, id=self.id.split('.')[-1])

        self.folds = folds

        # checking if the aggregation is specified properly
        if aggregation in ('rom', 'mor'):
            self.aggregation = aggregation
        else:
            raise ValueError(f'aggregation {aggregation} is not supported yet')

        self.score_bounds = score_bounds

        self.linear_programming = None

        # initializing the folds
        self.initialize_folds()

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
                'score_bounds': self.score_bounds,
                'folds': [fold.to_dict(problem_only) for fold in self.folds],
                'aggregation': self.aggregation
                }

    def __repr__(self):
        """
        Returning a string representation

        Returns:
            str: the string representation
        """
        return str(self.to_dict())

    def initialize_folds(self):
        """
        Initialize all folds by turning the specifications into fold objects
        """
        if not isinstance(self.folds[0], Fold):
            self.folds = [Fold(**fold) for fold in self.folds]

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

        # sampling each fold
        folds = [fold.sample(random_state=random_state).to_dict(problem_only=False) for fold in self.folds]
        params = self.to_dict(problem_only=True)
        params['folds'] = folds

        return Dataset(**params)

    def calculate_figures(self):
        figures = {'tp': 0, 'tn': 0, 'p': 0, 'n': 0}
        for fold in self.folds:
            figures['tp'] += fold.figures['tp']
            figures['tn'] += fold.figures['tn']
            figures['p'] += fold.p
            figures['n'] += fold.n
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
            for fold in self.folds:
                fold_scores = fold.calculate_scores(score_subset)
                for key in fold_scores:
                    scores[key] += fold_scores[key]
            for key in scores:
                scores[key] /= len(self.folds)

        if rounding_decimals is not None:
            scores = {key: np.round(value, rounding_decimals)
                            for key, value in scores.items()}

        return scores

    def init_lp(self, lp_problem):
        """
        Initializes the linear programming problem for the dataset

        Args:
            lp_problem (pl.LpProblem): a linear programming problem by pulp

        Returns:
            pl.LpProblem: the updated linear programming problem
        """

        for fold in self.folds:
            fold.init_lp(lp_problem)

        self.linear_programming = {'tp': sum(fold.linear_programming['tp'] for fold in self.folds),
                                    'tn': sum(fold.linear_programming['tn'] for fold in self.folds),
                                    'p': sum(fold.p for fold in self.folds),
                                    'n': sum(fold.n for fold in self.folds)}

        if self.aggregation == 'rom':
            self.linear_programming = {**self.linear_programming,
                                        **calculate_scores_for_lp({**self.linear_programming})}
        elif self.aggregation == 'mor':
            for key in ['acc', 'sens', 'spec', 'bacc']:
                self.linear_programming[key] = sum(fold.linear_programming[key] for fold in self.folds) * (1.0 / len(self.folds))

        add_bounds(lp_problem, self.linear_programming, self.score_bounds, f'dataset {self.id}')

        return lp_problem

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

    def add_bounds(self, score_bounds):
        params = self.to_dict(problem_only=False)
        params['score_bounds'] = score_bounds
        return Dataset(**params)

    def get_fold_bounds(self, score_subset=None, feasible=True):
        return [fold.get_bounds(score_subset, feasible) for fold in self.folds]

    def add_fold_bounds(self, score_bounds):
        if isinstance(score_bounds, dict):
            score_bounds = [score_bounds] * len(self.folds)

        folds = [fold.to_dict(problem_only=True) for fold in self.folds]
        for fold, sb in zip(folds, score_bounds):
            fold['score_bounds'] = sb
        params = self.to_dict()
        params['folds'] = folds

        return Dataset(**params)

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
        folds = [fold.populate(lp_problem).to_dict(problem_only=False) for fold in self.folds]
        params = self.to_dict(problem_only=True)
        params['folds'] = folds

        return Dataset(**params)

    def check_bounds(self):
        """
        Checks if the boundary conditions hold and returns a summary.
        The 'all_bounds' flag indicates the result of bound checks
        for each fold and the dataset.

        Returns:
            dict: a summary of the evaluation of the boundary conditions
        """
        scores = self.calculate_scores()
        figures = self.calculate_figures()

        downstream = [fold.check_bounds() for fold in self.folds]
        flag = all(tmp['bounds_flag'] for tmp in downstream)
        check_score_bounds = check_bounds(scores, self.score_bounds)

        all_bounds = (flag
                        and (check_score_bounds is None or check_score_bounds))

        return {'id': self.id,
                'figures': figures,
                'scores': scores,
                'score_bounds': self.score_bounds,
                'score_bounds_flag': check_score_bounds,
                'bounds_flag': all_bounds,
                'folds': downstream}

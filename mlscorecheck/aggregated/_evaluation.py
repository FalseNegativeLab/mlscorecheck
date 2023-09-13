"""
This module implements the abstraction for an Evaluation
"""

import pulp as pl

from ._fold import Fold
from ._dataset import Dataset
from ._folding import Folding
from ..core import (init_random_state, dict_mean, round_scores, dict_minmax,
                    NUMERICAL_TOLERANCE, round_scores)
from ..individual import calculate_scores_for_lp
from ._folding_utils import _create_folds
from ._utils import check_bounds
from ._linear_programming import add_bounds

class Evaluation:
    """
    Abstract representation of an evaluation
    """
    def __init__(self, dataset: dict, folding: dict, aggregation: str, fold_score_bounds: dict=None):
        """
        Constructor of the object

        Args:
            dataset (dict): the dataset specification
            folding (dict): the folding specification
            aggregation (str): the mode of aggregation over the folds ('mor'/'rom')
            fold_score_bounds(None|dict(str,tuple(float,float))): the bounds on the scores
                                                                for the folds
        """
        self.dataset = Dataset(**dataset)
        self.folding = Folding(**folding)
        self.fold_score_bounds = fold_score_bounds
        self.aggregation = aggregation

        if aggregation == 'rom' and fold_score_bounds is not None:
            raise ValueError('It is unlikely that fold score bounds are set for a RoM '\
                                'aggregation, therefore, it is not supported.')

        self.tp = None
        self.tn = None
        self.scores = None

        self.folds = self.folding.generate_folds(self.dataset, self.aggregation)
        self.p = sum(fold.p for fold in self.folds)
        self.n = sum(fold.n for fold in self.folds)

    def to_dict(self):
        """
        Returns the dictionary representation of the object

        Returns:
            dict: the dictionary representation
        """
        return {'dataset': self.dataset.to_dict(),
                'folding': self.folding.to_dict(),
                'fold_score_bounds': self.fold_score_bounds,
                'aggregation': self.aggregation}

    def sample_figures(self, random_state=None):
        """
        Samples the figures in the evaluation

        Args:
            random_state (None|int|np.random.RandomState): the random seed/state to use

        Returns:
            obj: the self object with the sampled figures
        """
        random_state = init_random_state(random_state)

        for fold in self.folds:
            fold.sample_figures(random_state)

        self.calculate_scores()

        return self

    def calculate_scores(self, rounding_decimals: int=None, score_subset: list=None):
        """
        Calculates the scores

        Args:
            rounding_decimals (int|None): the number of decimals to round the scores to

        Returns:
            dict: the calculated scores
        """
        for fold in self.folds:
            fold.calculate_scores()

        if isinstance(self.folds[0].tp, pl.LpVariable):
            self.tp = pl.lpSum(fold.tp for fold in self.folds)
            self.tn = pl.lpSum(fold.tn for fold in self.folds)
        else:
            self.tp = sum(fold.tp for fold in self.folds)
            self.tn = sum(fold.tn for fold in self.folds)

        if self.aggregation == 'rom':
            self.scores = calculate_scores_for_lp({'p': self.p,
                                                    'n': self.n,
                                                    'tp': self.tp,
                                                    'tn': self.tn})
        elif self.aggregation == 'mor':
            self.scores = dict_mean([fold.scores for fold in self.folds])

        self.scores = (self.scores if score_subset is None
                        else {key: value for key, value in self.scores.items()
                                            if key in score_subset})

        return self.scores if rounding_decimals is None else round_scores(self.scores,
                                                                            rounding_decimals)

    def init_lp(self, lp_problem: pl.LpProblem, scores=None):
        """
        Initializes a linear programming problem

        Args:
            lp_problem (pl.LpProblem): the linear programming problem to initialize
            scores (dict(str,float)): the scores used to estimate initial values

        Returns:
            pl.LpProblem: the updated linear programming problem
        """

        for fold in self.folds:
            fold.init_lp(scores=scores)

        self.calculate_scores()

        for fold in self.folds:
            add_bounds(lp_problem, fold.scores, self.fold_score_bounds, fold.identifier)

        return lp_problem

    def populate(self, lp_problem: pl.LpProblem):
        """
        Populates the evaluation with the figures in the solved linear programming problem

        Args:
            lp_problem (pl.LpProblem): the linear programming problem with ``solve()`` executed

        Returns:
            obj: the updated self object
        """
        for fold in self.folds:
            fold.populate(lp_problem)

        return self

    def check_bounds(self, numerical_tolerance: float=NUMERICAL_TOLERANCE):
        """
        Check the bounds in the problem

        Args:
            numerical_tolerance (float): the additional numerical tolerance to be used

        Returns:
            dict: a summary of the test, with the boolean flag under ``bounds_flag``
                    indicating the overall results
        """
        results = {'folds': []}
        for fold in self.folds:
            tmp = {'fold': fold.to_dict() | {'tp': fold.tp, 'tn': fold.tn},
                    'scores': fold.scores,
                    'score_bounds': self.fold_score_bounds}
            if self.fold_score_bounds is not None:
                tmp['bounds_flag'] = check_bounds(fold.scores,
                                            self.fold_score_bounds,
                                            numerical_tolerance)
            else:
                tmp['bounds_flag'] = True
            results['folds'].append(tmp)
        results['bounds_flag'] = all(fold['bounds_flag'] for fold in results['folds'])

        return results

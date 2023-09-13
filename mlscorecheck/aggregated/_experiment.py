"""
This module implements an abstraction for an experiment
"""

import pulp as pl

from ..core import (init_random_state, dict_mean, round_scores, dict_minmax,
                    NUMERICAL_TOLERANCE, round_scores)
from ..individual import calculate_scores_for_lp, calculate_scores

from ._fold import Fold
from ._dataset import (Dataset)
from ._evaluation import Evaluation
from ._utils import aggregated_scores

from ._utils import check_bounds

__all__ = ['Experiment']

class Experiment:
    """
    Abstract representation of an experiment
    """
    def __init__(self,
                    evaluations: list,
                    aggregation: str,
                    dataset_score_bounds: dict=None):
        """
        Constructor of the experiment

        Args:
            evaluations (list(dict)): a list of evaluation specifications
            aggregation (str): the mode of aggregation ('mor'/'rom')
            dataset_score_bounds (None|dict): the score bounds on the dataset scores
        """
        self.evaluations = [Evaluation(**evaluation) for evaluation in evaluations]
        self.dataset_score_bounds = dataset_score_bounds
        self.aggregation = aggregation

        self.p = sum(evaluation.p for evaluation in self.evaluations)
        self.n = sum(evaluation.n for evaluation in self.evaluations)

        self.tp = None
        self.tn = None

    def to_dict(self):
        """
        Returns a dictionary representation of the object

        Returns:
            dict: the dictionary representation of the object
        """
        return {'evaluations': [evaluation.to_dict() for evaluation in self.evaluations],
                'dataset_score_bounds': self.dataset_score_bounds,
                'aggregation': self.aggregation}

    def sample_figures(self, random_state=None):
        """
        Samples the ``tp`` and ``tn`` figures

        Args:
            random_state (None|int|np.random.RandomState): the random seed/state to use

        Returns:
            obj: the sampled self object
        """
        random_state = init_random_state(random_state)

        for evaluation in self.evaluations:
            evaluation.sample_figures(random_state)

        return self

    def calculate_scores(self, rounding_decimals: int=None, score_subset: list=None):
        """
        Calculates the scores

        Args:
            rounding_decimals (int|None): the number of decimals to round the scores to
            score_subset (list|None): the subset of scores to return

        Returns:
            dict(str,float): the scores
        """
        for evaluation in self.evaluations:
            evaluation.calculate_scores()

        print('AAA', self.evaluations[0].folds[0].tp.__class__)
        if isinstance(self.evaluations[0].folds[0].tp, pl.LpVariable):
            self.tp = pl.lpSum(evaluation.tp for evaluation in self.evaluations)
            self.tn = pl.lpSum(evaluation.tn for evaluation in self.evaluations)
        else:
            self.tp = sum(evaluation.tp for evaluation in self.evaluations)
            self.tn = sum(evaluation.tn for evaluation in self.evaluations)

        if self.aggregation == 'rom':
            self.scores = calculate_scores_for_lp({'p': self.p,
                                                    'n': self.n,
                                                    'tp': self.tp,
                                                    'tn': self.tn})
        elif self.aggregation == 'mor':
            self.scores = dict_mean([evaluation.scores for evaluation in self.evaluations])

        self.scores = (self.scores if score_subset is None
                        else {key: value for key, value in self.scores.items()
                                            if key in score_subset})

        return self.scores if rounding_decimals is None else round_scores(self.scores,
                                                                            rounding_decimals)

    def init_lp(self, lp_problem: pl.LpProblem, scores: dict=None):
        """
        Initializes a linear programming problem

        Args:
            lp_problem (pl.LpProblem): the linear programming problem to initialize
            scores (dict(str,float)): the scores used to estimate initial values

        Returns:
            pl.LpProblem: the updated linear programming problem
        """

        for evaluation in self.evaluations:
            evaluation.init_lp(lp_problem,
                                scores=scores)

        self.calculate_scores()

        if self.dataset_score_bounds is not None:
            for evaluation in self.evaluations:
                for key in self.dataset_score_bounds:
                    lp_problem += evaluation.scores[key] >= self.dataset_score_bounds[key][0]
                    lp_problem += evaluation.scores[key] <= self.dataset_score_bounds[key][1]

        return lp_problem

    def populate(self, lp_problem):
        """
        Populates the evaluation with the figures in the solved linear programming problem

        Args:
            lp_problem (pl.LpProblem): the linear programming problem with ``solve()`` executed

        Returns:
            obj: the updated self object
        """
        for evaluation in self.evaluations:
            evaluation.populate(lp_problem)

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

        results = {'evaluations': []}
        for evaluation in self.evaluations:
            tmp = {'folds': evaluation.check_bounds(numerical_tolerance),
                    'scores': evaluation.scores,
                    'score_bounds': self.dataset_score_bounds}
            if self.dataset_score_bounds is not None:
                tmp['bounds_flag'] = check_bounds(evaluation.scores,
                                                    self.dataset_score_bounds,
                                                    numerical_tolerance)
                tmp['bounds_flag'] = tmp['bounds_flag'] and tmp['folds']['bounds_flag']
            else:
                tmp['bounds_flag'] = tmp['folds']
            results['evaluations'].append(tmp)

        results['bounds_flag'] = all(evaluation['bounds_flag'] for evaluation in results['evaluations'])

        return results

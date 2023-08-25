"""
This module implements an abstraction for a fold

The abstraction is created to facilitate the testing and evaluation
of the aggregated checking of scores by enabling the easy creation,
sampling, and the calculation of scores and the assemblance of
the linear programming problem
"""
import string
import random
import copy

import numpy as np
import pulp as pl

from ..core import logger
from ..individual import calculate_scores_for_lp, calculate_scores

from ._linear_programming import add_bounds, check_bounds

__all__ = ['Fold',
            'random_identifier',
            'generate_fold_specification']

def random_identifier(length):
    """
    Generating a random identifier

    Args:
        length (int): the length of the string identifier

    Returns:
        str: the identifier
    """
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for _ in range(length))

def generate_fold_specification(max_p=100,
                            max_n=100,
                            random_state=None):
    """
    Generates a random fold

    Args:
        max_p (int): the maximum number of positives
        max_n (int): the maximum number of negatives
        random_state (None/int/np.random.RandomState): the random state/seed to use

    Returns:
        Fold: the fold object
    """
    if random_state is None or not isinstance(random_state, np.random.RandomState):
        random_state = np.random.RandomState(random_state)

    p = random_state.randint(1, max_p)
    n = random_state.randint(1, max_n)

    return {'p': p, 'n': n}

class Fold:
    """
    An abstraction for a fold
    """
    def __init__(self,
                    p,
                    n,
                    score_bounds=None,
                    id=None,
                    figures=None):
        """
        The constructor of the fold abstraction

        Args:
            p (int): the number of positives
            n (int): the number of negatives
            scores_bounds (None/dict(str,tuple)): a dictionary of bounds on the scores
                                                'acc', 'sens', 'spec', 'bacc'
            id (None/str): the identifier of the fold
            figures (dict(str,int)): a figures (tp and tn values)
            scores (dict(str,float)): the scores of the fold
        """
        self.p = p
        self.n = n

        if id is None:
            self.id = random_identifier(16)
            logger.info(f'a random identifier for the fold has been generated {self.id}')
        else:
            self.id = id

        self.score_bounds = score_bounds

        self.figures = figures

        self.linear_programming = None

        self.variable_names = {'tp': f'tp_{self.id}',
                                'tn': f'tn_{self.id}'}

    def to_dict(self, problem_only=False):
        """
        Returns a dict representation

        Args:
            problem_only (bool): whether to return the problem only (True) or add the
                                figuress and scores (False)

        Returns:
            dict: the dict representation
        """
        if problem_only:
            return {'p': self.p,
                    'n': self.n,
                    'id': self.id,
                    'score_bounds': copy.deepcopy(self.score_bounds)}
        else:
            return {**self.to_dict(problem_only=True),
                    'figures': copy.deepcopy(self.figures)}

    def __repr__(self):
        """
        Returning a string representation

        Returns:
            str: the string representation
        """
        return str(self.to_dict())

    def sample(self, random_state=None):
        """
        Samples the problem, that is, generates random (but physical) tp and tn values

        Args:
            random_state (None/int/np.random.RandomState): the random state to use

        Returns:
            self: the sampled fold
        """
        if random_state is None or not isinstance(random_state, np.random.RandomState):
            random_state = np.random.RandomState(random_state)

        figures = {'tp': int(random_state.randint(self.p+1)),
                    'tn': int(random_state.randint(self.n+1))}

        return Fold(**self.to_dict(problem_only=True), figures=figures)

    def calculate_scores(self, score_subset=None, rounding_decimals=None):
        """
        Calculates all scores for the fold

        Returns:
            dict(str,float): the scores
            rounding_decimals (None/float): how many digits to round the decimals to
        """
        if self.figures is None:
            raise ValueError('Call "sample" or "populate" first or specify '\
                                'figures when the object is instantiated.')

        score_subset = ['acc', 'sens', 'spec', 'bacc'] if score_subset is None else score_subset

        scores = calculate_scores_for_lp({'p': self.p,
                                            'n': self.n,
                                            **self.figures})

        scores = {key: value for key, value in scores.items() if key in score_subset}

        if rounding_decimals is not None:
            scores = {key: np.round(value, rounding_decimals) for key, value in scores.items()}

        return scores

    def init_lp(self, lp_problem):
        """
        Initializes the linear programming problem for the fold

        Args:
            lp_problem (pl.LpProblem): a linear programming problem by pulp

        Returns:
            pl.LpProblem: the updated linear programming problem
        """
        self.linear_programming = \
            {'tp': pl.LpVariable(self.variable_names['tp'], 0, self.p, pl.LpInteger),
                'tn': pl.LpVariable(self.variable_names['tn'], 0, self.n, pl.LpInteger)}

        self.linear_programming = {**self.linear_programming,
                                    **calculate_scores_for_lp({**self.linear_programming,
                                                                'p': self.p,
                                                                'n': self.n})}

        lp_problem = add_bounds(lp_problem, self.linear_programming, self.score_bounds, f'fold {self.id}')

        return lp_problem

    def get_bounds(self, score_subset=None, feasible=True):
        """
        Sets the score bounds according to the feasibility flag

        Args:
            feasible (bool): if True, sets feasible score bounds, sets infeasible score bounds otherwise

        Returns:
            self: the object itself
        """
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
        """
        A setter function for score bounds

        Args:
            score_bounds (dict(str,tuple(float,float))): the score bounds to set

        Returns:
            self: the adjusted object
        """
        params = self.to_dict(problem_only=False)
        params['score_bounds'] = score_bounds
        return Fold(**params)

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
        variable_names = list(self.variable_names.values())

        print(variable_names)
        print(lp_problem.variables())

        figures = {
            variable.name.split('_')[0]: variable.varValue
            for variable in lp_problem.variables()
            if variable.name in variable_names
        }

        return Fold(**self.to_dict(problem_only=True), figures=figures)

    def check_bounds(self):
        """
        Checks if the boundary conditions hold and returns a summary

        Returns:
            dict: a summary of the evaluation of the boundary conditions
        """
        if self.figures is None:
            raise ValueError('Call "sample" or "populate" first or specify '\
                                'figures when the object is instantiated.')

        scores = self.calculate_scores()

        score_flag = check_bounds(scores, self.score_bounds)
        bounds_flag = True
        bounds_flag = bounds_flag if score_flag is None else bounds_flag and score_flag

        return {'id': self.id,
                'figures': copy.deepcopy(self.figures),
                'scores': scores,
                'score_bounds': copy.deepcopy(self.score_bounds),
                'score_bounds_flag': score_flag,
                'bounds_flag': bounds_flag}

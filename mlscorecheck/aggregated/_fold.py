"""
This module implements an abstraction for a fold
"""
import string
import random

import numpy as np
import pulp as pl

from ..core import logger
from ..individual import calculate_scores_for_lp, calculate_scores

from ._linear_programming import add_bounds, check_bounds

__all__ = ['Fold', 'random_identifier']

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

class Fold:
    """
    An abstraction for a fold
    """
    def __init__(self,
                    p,
                    n,
                    score_bounds=None,
                    tptn_bounds=None,
                    id=None,
                    figures=None,
                    scores=None):
        """
        The constructor of the fold abstraction

        Args:
            p (int): the number of positives
            n (int): the number of negatives
            scores_bounds (None/dict(str,tuple)): a dictionary of bounds on the scores
                                                'acc', 'sens', 'spec', 'bacc'
            tptn_bounds (None/dict(str,tuple)): a dictionary of the bounds on the tp and n values
            id (None/str): the identifier of the fold
            figures (dict(str,int)): a figures (tp and tn values)
            scores (dict(str,float)): the scores of the fold
        """
        self.p = p
        self.n = n

        if id is None:
            logger.info('generating a random identifier for the fold')
            self.id = randomword(16)
        else:
            self.id = id

        self.score_bounds = score_bounds
        self.tptn_bounds = tptn_bounds

        self.figures = figures
        self.scores = scores

        self.linear_programming = None

        self.variable_names = {'tp': f'tp_{self.id}',
                                'tn': f'tn_{self.id}'}

    def to_dict(self, raw_problem=False):
        """
        Returns a dict representation

        Args:
            raw_problem (bool): whether to return the problem only (True) or add the
                                figuress and scores (False)

        Returns:
            dict: the dict representation
        """
        if raw_problem:
            return {'p': self.p,
                    'n': self.n,
                    'id': self.id,
                    'score_bounds': self.score_bounds,
                    'tptn_bounds': self.tptn_bounds}
        else:
            return {**self.to_dict(raw_problem=True),
                    'figures': self.figures,
                    'scores': self.scores}

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

        self.figures = {'tp': float(random_state.randint(self.p+1)),
                            'tn': float(random_state.randint(self.n+1))}
        return self

    def calculate_scores(self):
        """
        Calculates all scores for the fold

        Returns:
            dict(str,float): the scores
        """
        self.scores = calculate_scores({'p': self.p,
                                        'n': self.n,
                                        **self.figures})
        return self.scores

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

        lp_problem = add_bounds(lp_problem, self.linear_programming, self.score_bounds)
        lp_problem = add_bounds(lp_problem, self.linear_programming, self.tptn_bounds)

        return lp_problem

    def populate_with_solution(self, lp_problem):
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

        self.figures = {
            variable.name.split('_')[0]: variable.varValue
            for variable in lp_problem.variables()
            if variable.name in variable_names
        }

        return self

    def check_bounds(self):
        """
        Checks if the boundary conditions hold and returns a summary

        Returns:
            dict: a summary of the evaluation of the boundary conditions
        """
        return {'figures': self.figures,
                    'scores': self.scores,
                    'score_bounds': self.score_bounds,
                    'check_score_bounds': check_bounds(self.scores, self.score_bounds),
                    'tptn_bounds': self.tptn_bounds,
                    'check_tptn_bounds': check_bounds(self.figures, self.tptn_bounds)}

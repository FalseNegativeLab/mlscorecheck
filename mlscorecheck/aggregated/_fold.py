"""
This module implements an abstraction for a fold

The abstraction is created to facilitate the testing and evaluation
of the aggregated checking of scores by enabling the easy creation,
sampling, and the calculation of scores and the assemblance of
the linear programming problem
"""

import copy

import pulp as pl

from ..core import (logger, init_random_state, round_scores, NUMERICAL_TOLERANCE)
from ..individual import calculate_scores_for_lp

from ._linear_programming import add_bounds
from ._utils import check_bounds, random_identifier, aggregated_scores, create_bounds

__all__ = ['Fold',
            'random_identifier',
            'generate_fold_specification']

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
    random_state = init_random_state(random_state)

    p = random_state.randint(1, max_p)
    n = random_state.randint(1, max_n)

    return {'p': p, 'n': n}

class Fold:
    """
    An abstraction for a fold
    """
    def __init__(self,
                    *,
                    p,
                    n,
                    score_bounds=None,
                    identifier=None,
                    figures=None):
        """
        The constructor of the fold abstraction

        Args:
            p (int): the number of positives
            n (int): the number of negatives
            scores_bounds (None/dict(str,tuple)): a dictionary of bounds on the scores
                                                'acc', 'sens', 'spec', 'bacc'
            identifier (None/str): the identifier of the fold
            figures (dict(str,int)): a figures (tp and tn values)
        """
        self.p = p
        self.n = n

        if identifier is None:
            self.identifier = random_identifier(16)
            logger.info('a random identifier for the fold has been generated %s', self.identifier)
        else:
            self.identifier = identifier

        self.score_bounds = score_bounds

        self.figures = figures

        self.linear_programming = None

        self.variable_names = {'tp': f'tp_{self.identifier}',
                                'tn': f'tn_{self.identifier}'}

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
                    'identifier': self.identifier,
                    'score_bounds': copy.deepcopy(self.score_bounds)}

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
        random_state = init_random_state(random_state)

        return Fold(p=self.p,
                    n=self.n,
                    score_bounds=self.score_bounds,
                    identifier=self.identifier,
                    figures={'tp': int(random_state.randint(self.p+1)),
                                'tn': int(random_state.randint(self.n+1))})

    def has_bounds(self):
        """
        Checks if the fold has score bounds specified

        Returns:
            bool: a flag indicating if the fold has score bounds specified
        """
        return self.score_bounds is not None

    def calculate_scores(self, score_subset=None, rounding_decimals=None):
        """
        Calculates all scores for the fold

        Args:
            score_subset (list): the list of scores to compute
            rounding_decimals (None/float): how many digits to round the decimals to

        Returns:
            dict(str,float): the scores

        Raises:
            ValueError: if the fold instances is neither sampled nor populated
        """
        if self.figures is None:
            raise ValueError('Call "sample" or "populate" first or specify '\
                                'figures when the object is instantiated.')

        score_subset = aggregated_scores if score_subset is None else score_subset

        scores = calculate_scores_for_lp({'p': self.p,
                                            'n': self.n,
                                            **self.figures},
                                            score_subset)

        return round_scores(scores, rounding_decimals)

    def init_lp(self, lp_problem, scores):
        """
        Initializes the linear programming problem for the fold

        Args:
            lp_problem (pl.LpProblem): a linear programming problem by pulp
            scores (dict(str,float)): the scores to match are used to find
                                        suitable initial values for the variables

        Returns:
            pl.LpProblem: the updated linear programming problem
        """
        self.linear_programming = \
            {'tp': pl.LpVariable(self.variable_names['tp'], 0, self.p, pl.LpInteger),
                'tn': pl.LpVariable(self.variable_names['tn'], 0, self.n, pl.LpInteger)}

        if 'acc' in scores:
            tp_init = scores['acc'] * self.p
            tn_init = scores['acc'] * self.n
        if 'bacc' in scores:
            tp_init = scores['bacc'] * self.p
            tn_init = scores['bacc'] * self.n

        if 'sens' in scores:
            tp_init = scores['sens'] * self.p
        if 'spec' in scores:
            tn_init = scores['spec'] * self.n

        self.linear_programming['tp'].setInitialValue(int(tp_init))
        self.linear_programming['tn'].setInitialValue(int(tn_init))

        self.linear_programming = {**self.linear_programming,
                                    **calculate_scores_for_lp({**self.linear_programming,
                                                                'p': self.p,
                                                                'n': self.n})}

        lp_problem = add_bounds(lp_problem,
                                self.linear_programming,
                                self.score_bounds,
                                f'fold {self.identifier}')

        return lp_problem

    def get_bounds(self, score_subset=None, feasible=True):
        """
        Sets the score bounds according to the feasibility flag

        Args:
            scores_subset (None/list): the list of scores to get bounds for
            feasible (bool): if True, sets feasible score bounds, sets infeasible score
                                bounds otherwise

        Returns:
            self: the object itself
        """
        scores = self.calculate_scores(score_subset)
        return create_bounds(scores, feasible)

    def add_bounds(self, score_bounds):
        """
        A setter function for score bounds

        Args:
            score_bounds (dict(str,tuple(float,float))): the score bounds to set

        Returns:
            self: the adjusted object
        """
        return Fold(p=self.p,
                        n=self.n,
                    identifier=self.identifier,
                    figures=self.figures,
                    score_bounds=score_bounds)

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

        figures = {
            variable.name.split('_')[0]: variable.varValue
            for variable in lp_problem.variables()
            if variable.name in variable_names
        }

        return Fold(**self.to_dict(problem_only=True), figures=figures)

    def check_bounds(self, numerical_tolerance=NUMERICAL_TOLERANCE):
        """
        Checks if the boundary conditions hold and returns a summary

        Args:
            numerical_tolerance (float): the numerical tolerance

        Returns:
            dict: a summary of the evaluation of the boundary conditions

        Raises:
            ValueError: if the fold instance is neither sampled nor populated
        """
        if self.figures is None:
            raise ValueError('Call "sample" or "populate" first or specify '\
                                'figures when the object is instantiated.')

        scores = self.calculate_scores()

        score_flag = check_bounds(scores, self.score_bounds, numerical_tolerance)
        bounds_flag = True
        bounds_flag = bounds_flag if score_flag is None else bounds_flag and score_flag

        return {'identifier': self.identifier,
                'figures': copy.deepcopy(self.figures),
                'scores': scores,
                'score_bounds': copy.deepcopy(self.score_bounds),
                'score_bounds_flag': score_flag,
                'bounds_flag': bounds_flag}

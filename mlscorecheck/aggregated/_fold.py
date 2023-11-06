"""
This module implements an abstraction for a fold

The abstraction is created to facilitate the testing and evaluation
of the aggregated checking of scores by enabling the easy creation,
sampling, and the calculation of scores and the assemblance of
the linear programming problem
"""

import pulp as pl

from ..core import init_random_state, round_scores
from ..scores import calculate_scores_for_lp

from ._utils import random_identifier, aggregated_scores

__all__ = ["Fold"]


class Fold:
    """
    Abstract representation of a fold
    """

    def __init__(self, p: int, n: int, identifier: str = None):
        """
        Constructor of a fold

        Args:
            p (int): the number of positives
            n (int): the number of negatives
            identifier (None|str): identifier of the fold, randomly generated if None
        """
        self.p = p
        self.n = n
        self.identifier = random_identifier(5) if identifier is None else identifier

        self.tp = None
        self.tn = None
        self.scores = None

        self.variable_names = {
            "tp": f"tp_{self.identifier}".replace("-", "_"),
            "tn": f"tn_{self.identifier}".replace("-", "_"),
        }

    def to_dict(self) -> dict:
        """
        Dictionary representation of the fold

        Returns:
            dict: the dictionary representation
        """
        return {"p": self.p, "n": self.n, "identifier": self.identifier}

    def sample_figures(self, random_state=None):
        """
        Samples the ``tp`` and ``tn`` figures

        Args:
            random_state (None|int|np.random.RandomState): the random state/seed to use

        Returns:
            Fold: the self object after sampling
        """
        random_state = init_random_state(random_state)

        self.tp = random_state.randint(self.p + 1)
        self.tn = random_state.randint(self.n + 1)

        return self

    def calculate_scores(
        self, rounding_decimals: int = None, score_subset: list = None
    ) -> dict:
        """
        Calculate the scores for the fold

        Args:
            rounding_decimals (int|None): the number of decimals to round to
            score_subset (list): the subset of scores to calculate

        Returns:
            dict: the scores
        """
        score_subset = score_subset if score_subset is not None else aggregated_scores

        self.scores = calculate_scores_for_lp(
            {"p": self.p, "n": self.n, "tp": self.tp, "tn": self.tn},
            score_subset=score_subset,
        )

        return (
            self.scores
            if rounding_decimals is None
            else round_scores(self.scores, rounding_decimals)
        )

    def set_initial_values(self, scores):
        """
        Sets the initial values for the tp and tn variables

        Args:
            scores (dict): the dictionary of scores
        """
        if "acc" in scores:
            tp_init = scores["acc"] * self.p
            tn_init = scores["acc"] * self.n
        if "bacc" in scores:
            tp_init = scores["bacc"] * self.p
            tn_init = scores["bacc"] * self.n

        if "sens" in scores:
            tp_init = scores["sens"] * self.p
        if "spec" in scores:
            tn_init = scores["spec"] * self.n

        self.tp.setInitialValue(int(tp_init))
        self.tn.setInitialValue(int(tn_init))

    def init_lp(self, scores: dict = None):
        """
        Initialize a linear programming problem by creating the variables for the fold

        Args:
            scores (dict|None): the score values to be used to set initial values

        Returns:
            pl.LpProblem: the updated problem
        """
        self.tp = pl.LpVariable(self.variable_names["tp"], 0, self.p, pl.LpInteger)
        self.tn = pl.LpVariable(self.variable_names["tn"], 0, self.n, pl.LpInteger)

        if scores is not None:
            self.set_initial_values(scores)

        score_subset = aggregated_scores
        if scores is not None:
            score_subset = list(set(scores.keys()).intersection(set(aggregated_scores)))

        self.calculate_scores(score_subset=score_subset)

    def populate(self, lp_problem: pl.LpProblem) -> pl.LpProblem:
        """
        Populate the fold with the ``tp`` and ``tn`` values from the linear program

        Args:
            lp_problem (pl.LpProblem): the linear programming problem

        Returns:
            obj: the self object populated with the ``tp`` and ``tn`` scores
        """
        for variable in lp_problem.variables():
            if variable.name == self.variable_names["tp"]:
                self.tp = variable.varValue
            if variable.name == self.variable_names["tn"]:
                self.tn = variable.varValue

        return self

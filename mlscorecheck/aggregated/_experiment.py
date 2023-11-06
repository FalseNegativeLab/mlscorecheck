"""
This module implements an abstraction for an experiment
"""

import pulp as pl

from ..core import init_random_state, dict_mean, round_scores, NUMERICAL_TOLERANCE
from ..scores import calculate_scores_for_lp

from ._evaluation import Evaluation

from ._utils import check_bounds, aggregated_scores
from ._linear_programming import add_bounds

__all__ = ["Experiment"]


class Experiment:
    """
    Abstract representation of an experiment
    """

    def __init__(
        self, evaluations: list, aggregation: str, dataset_score_bounds: dict = None
    ):
        """
        Constructor of the experiment

        Args:
            evaluations (list(dict)): a list of evaluation specifications
            aggregation (str): the mode of aggregation ('mos'/'som')
            dataset_score_bounds (None|dict): the score bounds on the dataset scores
        """
        self.evaluations = [Evaluation(**evaluation) for evaluation in evaluations]
        self.dataset_score_bounds = dataset_score_bounds
        self.aggregation = aggregation

        if aggregation == "som" and dataset_score_bounds is not None:
            raise ValueError(
                "It is unlikely that fold score bounds are set for a SoM "
                "aggregation, therefore, it is not supported."
            )

        self.figures = {
            "tp": None,
            "tn": None,
            "p": sum(evaluation.figures["p"] for evaluation in self.evaluations),
            "n": sum(evaluation.figures["n"] for evaluation in self.evaluations),
        }

        self.scores = None

    def to_dict(self) -> dict:
        """
        Returns a dictionary representation of the object

        Returns:
            dict: the dictionary representation of the object
        """
        return {
            "evaluations": [evaluation.to_dict() for evaluation in self.evaluations],
            "dataset_score_bounds": self.dataset_score_bounds,
            "aggregation": self.aggregation,
        }

    def sample_figures(self, random_state=None, score_subset: list = None):
        """
        Samples the ``tp`` and ``tn`` figures

        Args:
            random_state (None|int|np.random.RandomState): the random seed/state to use

        Returns:
            obj: the sampled self object
        """
        random_state = init_random_state(random_state)

        for evaluation in self.evaluations:
            evaluation.sample_figures(random_state, score_subset=score_subset)

        self.calculate_scores(score_subset=score_subset)

        return self

    def calculate_scores(
        self, rounding_decimals: int = None, score_subset: list = None
    ) -> dict:
        """
        Calculates the scores

        Args:
            rounding_decimals (int|None): the number of decimals to round the scores to
            score_subset (list|None): the subset of scores to return

        Returns:
            dict(str,float): the scores
        """
        score_subset = (
            ["acc", "sens", "spec", "bacc"] if score_subset is None else score_subset
        )
        score_subset = [
            score for score in score_subset if score in ["acc", "sens", "spec", "bacc"]
        ]

        for evaluation in self.evaluations:
            evaluation.calculate_scores(score_subset=score_subset)

        if isinstance(self.evaluations[0].folds[0].tp, pl.LpVariable):
            self.figures["tp"] = pl.lpSum(
                evaluation.figures["tp"] for evaluation in self.evaluations
            )
            self.figures["tn"] = pl.lpSum(
                evaluation.figures["tn"] for evaluation in self.evaluations
            )
        else:
            self.figures["tp"] = sum(
                evaluation.figures["tp"] for evaluation in self.evaluations
            )
            self.figures["tn"] = sum(
                evaluation.figures["tn"] for evaluation in self.evaluations
            )

        if self.aggregation == "som":
            self.scores = calculate_scores_for_lp(
                self.figures, score_subset=score_subset
            )
        elif self.aggregation == "mos":
            self.scores = dict_mean(
                [evaluation.scores for evaluation in self.evaluations]
            )

        return (
            self.scores
            if rounding_decimals is None
            else round_scores(self.scores, rounding_decimals)
        )

    def init_lp(self, lp_problem: pl.LpProblem, scores: dict = None) -> pl.LpProblem:
        """
        Initializes a linear programming problem

        Args:
            lp_problem (pl.LpProblem): the linear programming problem to initialize
            scores (dict(str,float)): the scores used to estimate initial values

        Returns:
            pl.LpProblem: the updated linear programming problem
        """

        for evaluation in self.evaluations:
            evaluation.init_lp(lp_problem, scores=scores)

        score_subset = aggregated_scores
        if scores is not None:
            score_subset = list(set(scores.keys()).intersection(set(aggregated_scores)))

        self.calculate_scores(score_subset=score_subset)

        for evaluation in self.evaluations:
            add_bounds(
                lp_problem,
                evaluation.scores,
                self.dataset_score_bounds,
                evaluation.dataset.identifier,
            )

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

    def check_bounds(self, numerical_tolerance: float = NUMERICAL_TOLERANCE) -> dict:
        """
        Check the bounds in the problem

        Args:
            numerical_tolerance (float): the additional numerical tolerance to be used

        Returns:
            dict: a summary of the test, with the boolean flag under ``bounds_flag``
                    indicating the overall results
        """

        results = {"evaluations": []}
        for evaluation in self.evaluations:
            tmp = {
                "folds": evaluation.check_bounds(numerical_tolerance),
                "scores": evaluation.scores,
                "score_bounds": self.dataset_score_bounds,
            }
            if self.dataset_score_bounds is not None:
                tmp["bounds_flag"] = check_bounds(
                    evaluation.scores, self.dataset_score_bounds, numerical_tolerance
                )
                tmp["bounds_flag"] = tmp["bounds_flag"] and tmp["folds"]["bounds_flag"]
            else:
                tmp["bounds_flag"] = tmp["folds"]
            results["evaluations"].append(tmp)

        results["bounds_flag"] = all(
            evaluation["bounds_flag"] for evaluation in results["evaluations"]
        )

        return results

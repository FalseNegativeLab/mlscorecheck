"""
This module implements the abstraction for an Evaluation
"""

import pulp as pl

from ._dataset import Dataset
from ._folding import Folding
from ..core import init_random_state, dict_mean, round_scores, NUMERICAL_TOLERANCE
from ..scores import calculate_scores_for_lp
from ._utils import check_bounds, aggregated_scores
from ._linear_programming import add_bounds


class Evaluation:
    """
    Abstract representation of an evaluation
    """

    def __init__(
        self,
        dataset: dict,
        folding: dict,
        aggregation: str,
        fold_score_bounds: dict = None,
    ):
        """
        Constructor of the object

        Args:
            dataset (dict): the dataset specification
            folding (dict): the folding specification
            aggregation (str): the mode of aggregation over the folds ('mos'/'som')
            fold_score_bounds(None|dict(str,tuple(float,float))): the bounds on the scores
                                                                for the folds
        """
        self.dataset = Dataset(**dataset)
        self.folding = Folding(**folding)
        self.fold_score_bounds = fold_score_bounds
        self.aggregation = aggregation

        if aggregation == "som" and fold_score_bounds is not None:
            raise ValueError(
                "It is unlikely that fold score bounds are set for a SoM "
                "aggregation, therefore, it is not supported."
            )

        self.folds = self.folding.generate_folds(self.dataset, self.aggregation)

        self.figures = {
            "tp": None,
            "tn": None,
            "p": sum(fold.p for fold in self.folds),
            "n": sum(fold.n for fold in self.folds),
        }

        self.scores = None

    def to_dict(self) -> dict:
        """
        Returns the dictionary representation of the object

        Returns:
            dict: the dictionary representation
        """
        return {
            "dataset": self.dataset.to_dict(),
            "folding": self.folding.to_dict(),
            "fold_score_bounds": self.fold_score_bounds,
            "aggregation": self.aggregation,
        }

    def sample_figures(self, random_state=None, score_subset: list = None):
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

        self.calculate_scores(score_subset=score_subset)

        return self

    def calculate_scores(
        self, rounding_decimals: int = None, score_subset: list = None
    ) -> dict:
        """
        Calculates the scores

        Args:
            rounding_decimals (int|None): the number of decimals to round the scores to
            score_subset (list): the list of scores to calculate scores for

        Returns:
            dict: the calculated scores
        """

        score_subset = aggregated_scores if score_subset is None else score_subset

        for fold in self.folds:
            fold.calculate_scores(score_subset=score_subset)

        if isinstance(self.folds[0].tp, pl.LpVariable):
            self.figures["tp"] = pl.lpSum(fold.tp for fold in self.folds)
            self.figures["tn"] = pl.lpSum(fold.tn for fold in self.folds)
        else:
            self.figures["tp"] = sum(fold.tp for fold in self.folds)
            self.figures["tn"] = sum(fold.tn for fold in self.folds)

        if self.aggregation == "som":
            self.scores = calculate_scores_for_lp(
                self.figures, score_subset=score_subset
            )
        elif self.aggregation == "mos":
            self.scores = dict_mean([fold.scores for fold in self.folds])

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
            scores (dict(str,float)|None): the scores used to estimate initial values

        Returns:
            pl.LpProblem: the updated linear programming problem
        """

        for fold in self.folds:
            fold.init_lp(scores=scores)

        score_subset = aggregated_scores
        if scores is not None:
            score_subset = list(set(scores.keys()).intersection(set(aggregated_scores)))

        self.calculate_scores(score_subset=score_subset)

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

    def check_bounds(self, numerical_tolerance: float = NUMERICAL_TOLERANCE) -> dict:
        """
        Check the bounds in the problem

        Args:
            numerical_tolerance (float): the additional numerical tolerance to be used

        Returns:
            dict: a summary of the test, with the boolean flag under ``bounds_flag``
                    indicating the overall results
        """
        results = {"folds": []}
        for fold in self.folds:
            tmp = {
                "fold": fold.to_dict() | {"tp": fold.tp, "tn": fold.tn},
                "scores": fold.scores,
                "score_bounds": self.fold_score_bounds,
            }
            if self.fold_score_bounds is not None:
                tmp["bounds_flag"] = check_bounds(
                    fold.scores, self.fold_score_bounds, numerical_tolerance
                )
            else:
                tmp["bounds_flag"] = True
            results["folds"].append(tmp)
        results["bounds_flag"] = all(fold["bounds_flag"] for fold in results["folds"])

        return results

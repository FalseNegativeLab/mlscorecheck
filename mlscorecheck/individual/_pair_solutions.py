"""
This module loads the solutions.

The Solution abstractions enable the evaluation of the solution
formulas with scalars and intervals too.
"""

import os
import json

from importlib.resources import files

from ._interval import Interval, IntervalUnion
from ._expression import Expression

__all__ = ["load_solutions", "Solution", "Solutions", "solution_specifications"]


class Solution:
    """
    Represents one single solution (expressions for tp and tn) and corresponding
    non-zero and non-negative conditions as expressions
    """

    def __init__(self, solution: dict, conditions: list = None):
        """
        Constructor of the solution

        Args:
            solution (dict(dict)): the solutions ({'expressions': , 'symbols': })
            conditions (list(dict)): the conditions to check
        """
        self.solution = solution
        self.conditions = conditions

        # extracting all symbols
        self.all_symbols = set()

        for item in self.solution.values():
            self.all_symbols = self.all_symbols.union(item["symbols"])

        for cond in self.conditions:
            self.all_symbols = self.all_symbols.union(cond["symbols"])

        self.conditions = sorted(self.conditions, key=lambda x: -x["depth"])

    def to_dict(self):
        """
        Returning a dictionary representation

        Returns:
            dict: the dictionary representation
        """
        return {"solution": self.solution, "conditions": self.conditions}

    def check_non_negative(self, value) -> bool:
        """
        Checks the non-negativity condition

        Args:
            value (numeric|Interval|IntervalUnion): the value to check

        Returns:
            bool: True if the condition fails, False otherwise
        """
        if isinstance(value, (Interval, IntervalUnion)):
            if isinstance(value, Interval):
                if value.upper_bound < 0:
                    return True
            elif all(interval.upper_bound < 0 for interval in value.intervals):
                return True
        elif value < 0:
            return True

        return False

    def check_non_zero(self, value) -> bool:
        """
        Checks the non-zero condition

        Args:
            value (numeric|Interval|IntervalUnion): the value to check

        Returns:
            bool: True if the condition fails, False otherwise
        """
        return (isinstance(value, (Interval, IntervalUnion)) and value.contains(0)) or (
            not isinstance(value, (Interval, IntervalUnion)) and abs(value) < 1e-8
        )

    def evaluate(self, subs):
        """
        Evaluate the solution with a substitution

        Args:
            subs (dict): a substitution

        Returns:
            dict: the results
        """
        subs = {key: subs[key] for key in self.all_symbols}

        message = None
        term = None
        for condition in self.conditions:
            if condition["mode"] == "non-negative":
                value = Expression(**condition).evaluate(subs)
                if self.check_non_negative(value):
                    message = "negative base"
                    term = condition["expression"]
                    break
            elif condition["mode"] == "non-zero":
                value = Expression(**condition).evaluate(subs)
                if self.check_non_zero(value):
                    message = "zero division"
                    term = condition["expression"]
                    break

        if message is not None:
            return {"tp": None, "tn": None, "message": message, "term": term}

        res = {
            key: Expression(**value).evaluate(subs)
            for key, value in self.solution.items()
        }
        if "tp" in self.solution:
            res["tp_formula"] = self.solution["tp"]["expression"]

        if "tn" in self.solution:
            res["tn_formula"] = self.solution["tn"]["expression"]

        return res


class Solutions:
    """
    Represents all solutions to a particular problem
    """

    def __init__(self, scores: list, solutions: list):
        """
        The constructor of the object

        Args:
            scores (list): the list of the names of the scores the solution is for
            solutions (list): the list of the individual solutions
        """
        self.scores = scores
        self.solutions = [Solution(**sol) for sol in solutions]

    def to_dict(self) -> dict:
        """
        Returns a dictionary representation

        Returns:
            dict: the dictionary representation
        """

        return {
            "scores": self.scores,
            "solutions": [sol.to_dict() for sol in self.solutions],
        }

    def evaluate(self, subs):
        """
        Evaluate the solutions with a substitution

        Args:
            subs (dict): a substitution

        Returns:
            dict: the results
        """
        results = []

        for sol in self.solutions:
            res = sol.evaluate(subs)
            results.append({**res})

        return results


def load_solutions():
    """
    Load the solutions

    Returns:
        dict: the dictionary of the solutions
    """
    sio = (
        files("mlscorecheck")
        .joinpath(os.path.join("individual", "solutions.json"))
        .read_text()
    )  # pylint: disable=unspecified-encoding

    solutions_dict = json.loads(sio)

    results = {}

    for sol in solutions_dict["solutions"]:
        scores = list(sol["scores"])
        if "p4" not in scores:
            results[tuple(sorted(scores))] = Solutions(**sol)

    # removing the solutions containing complex values
    del results[("fm", "gm")]
    del results[("fm", "mk")]
    # del results[('fm', 'p4')] # goes to complex
    del results[("fm", "upm")]  # goes to complex
    # del results[('dor', 'p4')] # goes to complex
    del results[("dor", "upm")]  # goes to complex
    del results[("gm", "mk")]  # goes to complex
    del results[("gm", "mcc")]  # goes to complex when tn = 0 (maybe other times too)

    return results


solution_specifications = load_solutions()

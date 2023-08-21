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

__all__ = ['load_solutions',
            'Solution',
            'Solutions']

def load_solutions():
    """
    Load the solutions

    Returns:
        dict: the dictionary of the solutions
    """
    sio = files('mlscorecheck').joinpath(os.path.join('individual', 'solutions.json')).read_text() # pylint: disable=unspecified-encoding

    solutions = json.loads(sio)

    results = {}

    for sol in solutions['solutions']:
        scores = list(sol['scores'])
        results[tuple(sorted(scores))] = Solutions(**sol)

    return results

class Solution:
    """
    Represents one single solution (expressions for tp and tn) and corresponding
    non-zero and non-negative conditions as expressions
    """
    def __init__(self,
                    solution,
                    non_zero,
                    non_negative):
        """
        Constructor of the solution

        Args:
            solution (dict(dict)): the solutions ({'expressions': , 'symbols': })
            non_zero (list(dict)): the non-zero conditions ([{'expression': , 'symbols':}])
            non_negative (list(dict)): the non-negative conditions ([{'expression': , 'symbols':}])
        """
        self.solution = solution
        self.non_zero = non_zero
        self.non_negative = non_negative

        # extracting all symbols
        self.all_symbols = set()

        for _, item in self.solution.items():
            self.all_symbols = self.all_symbols.union(item['symbols'])

        for non_zero_val in self.non_zero:
            self.all_symbols = self.all_symbols.union(non_zero_val['symbols'])

        for non_negative_val in self.non_negative:
            self.all_symbols = self.all_symbols.union(non_negative_val['symbols'])

    def to_dict(self):
        """
        Returning a dictionary representation

        Returns:
            dict: the dictionary representation
        """
        return {'solution': self.solution,
                'non_zero': self.non_zero,
                'non_negative': self.non_negative}

    def check_non_zeros(self, evals):
        """
        Check if the non-zero conditions hold

        Args:
            evals (dict): evaluations

        Returns:
            dict/None: if any of the conditions hold, the condition
        """
        for key, value in evals.items():
            if isinstance(value, (Interval, IntervalUnion)):
                if value.contains(0):
                    return {key: value}
            else:
                if abs(value) < 1e-8:
                    return {key: value}
        return None

    def non_zero_conditions(self, subs):
        """
        Checking the non-zero condition with a substitution

        Args:
            subs (dict): a substitution

        Returns:
            bool: the result of the check
        """
        non_zeros = {non_zero['expression']: Expression(**non_zero).evaluate(subs)
                                for non_zero in self.non_zero}

        return self.check_non_zeros(non_zeros)

    def check_non_negatives(self, evals):
        """
        Check if the non-negative conditions hold

        Args:
            evals (dict): evaluations

        Returns:
            dict/None: if any of the conditions hold, the condition
        """
        for key, value in evals.items():
            if isinstance(value, (Interval, IntervalUnion)):
                if isinstance(value, Interval):
                    if value.upper_bound < 0:
                        return {key: value}
                elif any(interval.upper_bound < 0 for interval in value.intervals):
                    return {key: value}
            elif value < 0:
                return {key: value}
        return None

    def non_negative_conditions(self, subs):
        """
        Checking the non-negativity condition with a substitution

        Args:
            subs (dict): a substitution

        Returns:
            bool: the result of the check
        """
        non_negatives = {non_negative['expression']: Expression(**non_negative).evaluate(subs)
                                for non_negative in self.non_negative}

        return self.check_non_negatives(non_negatives)

    def evaluate(self, subs):
        """
        Evaluate the solution with a substitution

        Args:
            subs (dict): a substitution

        Returns:
            dict: the results
        """
        subs = {key: subs[key] for key in self.all_symbols}

        if non_zero := self.non_zero_conditions(subs):
            return {'tp': None,
                    'tn': None,
                    'message': 'zero division',
                    'denominator': non_zero}

        if non_negative := self.non_negative_conditions(subs):
            return {'tp': None,
                    'tn': None,
                    'message': 'negative base',
                    'base': non_negative}

        res = {key: Expression(**value).evaluate(subs) for key, value in self.solution.items()}
        if 'tp' in self.solution:
            res['tp_formula'] = self.solution['tp']['expression']

        if 'tn' in self.solution:
            res['tn_formula'] = self.solution['tn']['expression']

        return res

class Solutions:
    """
    Represents all solutions to a particular problem
    """
    def __init__(self,
                    scores,
                    solutions):
        """
        The constructor of the object

        Args:
            scores (list): the list of the score descriptors
            solutions (list): the list of the individual solutions
        """
        self.scores = scores
        self.solutions = [Solution(**sol) for sol in solutions]

    def to_dict(self):
        """
        Returns a dictionary representation

        Returns:
            dict: the dictionary representation
        """

        return {'scores': self.scores,
                'solutions': [sol.to_dict() for sol in self.solutions]}

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

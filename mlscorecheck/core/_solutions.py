"""
This module imports the solutions
"""

from io import StringIO

from importlib.resources import files

import json

from ._interval import Interval, IntervalUnion

__all__ = ['load_solutions',
            'load_scores',
            'Solution',
            'Solutions',
            'ZeroDivision',
            'NegativeBase']

def load_solutions():
    sio = files('mlscorecheck').joinpath('core/solutions.json').read_text() # pylint: disable=unspecified-encoding

    solutions = json.loads(sio)

    results = {}

    for sol in solutions['solutions']:
        scores = [score['descriptor']['abbreviation'] for score in sol['scores']]
        results[tuple(sorted(scores))] = Solutions(**sol)

    return results

def load_scores():
    sio = files('mlscorecheck').joinpath('core/scores.json').read_text() # pylint: disable=unspecified-encoding

    scores = json.loads(sio)

    return scores['scores']

class ZeroDivision(Exception):
    """
    An exception indicating zero division
    """
    def __init__(self, expression):
        """
        The constructor of the exception

        Args:
            expression (dict): the expression and its value
        """
        self.expression = expression

class NegativeBase(BaseException):
    """
    An exception indicating the root of a negative number
    """
    def __init__(self, expression):
        """
        The constructor of the exception

        Args:
            expression (dict): the expression and its value
        """
        self.expression = expression

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

        #print(self.solution)
        #print(self.non_zero)
        #print(self.non_negative)

        # extracting all symbols
        self.all_symbols = set()

        for _, item in self.solution['symbols'].items():
            self.all_symbols = self.all_symbols.union(set(item))

        for non_zero in self.non_zero:
            self.all_symbols = self.all_symbols.union(non_zero['symbols'])

        for non_negative in self.non_negative:
            self.all_symbols = self.all_symbols.union(non_negative['symbols'])

        #print(self.all_symbols)

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
        for key, value in evals.items():
            if isinstance(value, (Interval, IntervalUnion)):
                if value.contains(0):
                    return {key: value}
            else:
                if value == 0:
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
        non_zeros = {non_zero['expression']: eval(non_zero['expression'], subs)
                                for non_zero in self.non_zero}

        #print(non_zeros)

        return self.check_non_zeros(non_zeros)

    def check_non_negatives(self, evals):
        for key, value in evals.items():
            if isinstance(value, (Interval, IntervalUnion)):
                if isinstance(value, Interval):
                    if value.lower_bound < 0:
                        return {key: value}
                else:
                    if any(interval.lower_bound < 0 for interval in value.intervals):
                        return {key: value}
            else:
                if value < 0:
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
        non_negatives = {non_negative['expression']: eval(non_negative['expression'], subs)
                                for non_negative in self.non_negative}

        return self.check_non_negatives(non_negatives)

    def evaluate(self, subs):
        """
        Evaluate the solution with a substitution

        Args:
            subs (dict): a substitution

        Returns:
            dict: the results

        Throws:
            ZeroDivision: when zero division occurs
            NegativeBase: when the root of a negative number is taken
        """
        subs = {key: subs[key] for key in self.all_symbols}

        non_zero = self.non_zero_conditions(subs)

        if non_zero:
            raise ZeroDivision(expression=non_zero)

        non_negative = self.non_negative_conditions(subs)

        if non_negative:
            raise NegativeBase(expression=non_negative)

        return {key: eval(value, subs) for key, value in self.solution['expressions'].items()}

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
            try:
                res = sol.evaluate(subs)
                res['message'] = None
            except ZeroDivision as exc:
                res = {'tp': None,
                        'tn': None,
                        'message': 'zero division error',
                        'denominator': exc.expression}
            except NegativeBase as exc:
                res = {'tp': None,
                        'tn': None,
                        'message': 'negative base',
                        'base': exc.expression}
            tmp = {'formula': sol.solution['expressions'],
                    'results': res}
            results.append(tmp)

        return results

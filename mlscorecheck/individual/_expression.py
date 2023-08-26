"""
This module implements a symbolic expression abstraction for the
ease of handling symbolic formulas
"""

from ..core import safe_eval
from ._interval import sqrt

__all__ = ['Expression']

class Expression:
    """
    The class represents a formal expression
    """
    def __init__(self, expression, symbols, functional_symbols=None):
        """
        The constructor of the expression

        Args:
            expression (str): the formal expression
            symbols (list(str)): the symbols in the expression
            functional_symbols (list(str)): the function symbols
        """
        if functional_symbols is None:
            self.functional_symbols = {'sqrt': sqrt}

        self.expression = expression
        self.symbols = symbols

    def evaluate(self, subs):
        """
        Evaluates the expression

        Args:
            subs (dict): the substitution

        Returns:
            numeric/Interval/IntervalUnion: the result of the evaluation
        """
        subs = {**{symbol: subs[symbol] for symbol in self.symbols},
                'sqrt': sqrt}
        return safe_eval(self.expression, subs)

    def to_dict(self):
        """
        Converts the expression into a dictionary representation

        Returns:
            dict: the dictionary representation
        """
        return {'expression': self.expression,
                'symbols': self.symbols,
                'functional_symbols': self.functional_symbols}
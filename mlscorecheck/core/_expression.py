"""
This module implements a symbolic expression abstraction for the
ease of handling symbolic formulas
"""

from ._interval import sqrt

__all__ = ['Expression']

class Expression:
    def __init__(self, expression, symbols, functional_symbols=None):
        if functional_symbols is None:
            self.functional_symbols = {'sqrt': sqrt}

        self.expression = expression
        self.symbols = symbols

    def evaluate(self, subs):
        subs = {**{symbol: subs[symbol] for symbol in self.symbols},
                #**{symbol: subs[symbol] for symbol in self.functional_symbols}
                'sqrt': sqrt}
        return eval(self.expression, subs)

    def to_dict(self):
        return {'expression': self.expression,
                'symbols': self.symbols,
                'functional_symbols': self.functional_symbols}

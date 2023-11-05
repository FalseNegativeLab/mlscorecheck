"""
This module implements the wrapper for the sympy algebra
"""

__all__ = ["SympyAlgebra"]

import importlib

from ._algebra import Algebra


class SympyAlgebra(Algebra):
    """
    The required algebra driven by sympy
    """

    def __init__(self):
        """
        Constructor of the algebra
        """
        Algebra.__init__(self)

        self.algebra = importlib.import_module("sympy")
        self.sqrt = self.algebra.sqrt

    def create_symbol(self, name: str, **kwargs):
        """
        Create a symbol in the algebra with the specified name and assumptions

        Args:
            name (str): the name of the symbol
            kwargs (dict): the assumptions

        Returns:
            object: the symbol
        """
        if "upper_bound" in kwargs:
            del kwargs["upper_bound"]
        if "lower_bound" in kwargs:
            del kwargs["lower_bound"]
        return self.algebra.Symbol(name, **kwargs)

    def num_denom(self, expression):
        """
        Extract the numerator and denominator

        Args:
            expression (object): the expression to process

        Returns:
            object, object: the numerator and denominator
        """
        return expression.as_numer_denom()

    def simplify(self, expression):
        """
        Simplify the expression

        Args:
            expression (object): the expression to simplify

        Returns:
            object: the symplified expression
        """
        return self.algebra.simplify(expression)

    def solve(self, equation, var, **kwargs):
        """
        Solve an equation for a variable

        Args:
            equation (object): the equation to solve
            var (object): the variable to solve for
            kwargs (dict): additional parameters to the solver

        Returns:
            list(dict): the solutions
        """
        results = self.algebra.solve(equation, var, **kwargs)
        solutions = []
        for res in results:
            solutions.append({var: res})
        return solutions

    def subs(self, expression, subs_dict):
        """
        Substitute a substitution into the expression

        Args:
            expression (object): the expression to substitute into
            subs_dict (dict): the substitution

        Returns:
            object: the result of the substitution
        """
        return expression.subs(subs_dict)

    def args(self, expression) -> list:
        """
        The list of arguments

        Args:
            expression (object): the expression to process

        Returns:
            list: the list of arguments
        """
        return expression.free_symbols

    def is_trivial(self, expression) -> bool:
        """
        Checks if the expression is trivial
        TODO: checking other constants

        Args:
            expression (object): the expression to check

        Returns:
            bool: True if the expression is trivial, False otherwise
        """
        return True if expression is None else expression == 1

    def is_root(self, expression) -> bool:
        """
        Checks if the expression is a root

        Args:
            expression (object): the expression to check if it is a root

        Returns:
            bool: True if the expression is a root, False otherwise
        """
        if self.is_power(expression):
            _, exponent = expression.args
            if 0 < exponent < 1:
                return True
        return False

    def is_power(self, expression) -> bool:
        """
        Checks whether the expression is a power

        Args:
            expression (object): the expression to check

        Returns:
            bool: whether the expression is a power
        """
        return isinstance(expression, self.algebra.core.power.Pow)

    def is_division(self, expression) -> bool:
        """
        Checks whether the expression is a division

        Args:
            expression (object): the expression to check

        Returns:
            bool: whether the expression is a division
        """
        if self.is_power(expression):
            _, power = expression.args
            if power < 0:
                return True

        if isinstance(expression, self.algebra.core.power.Mul):
            args = expression.args
            if len(args) == 2 and self.is_power(args[1]):
                _, power = args[1].args
                if power < 0:
                    return True
        return False

    def operands(self, expression) -> list:
        """
        Returns the list of operands

        Args:
            expression (object): the expression to return the operands of

        Returns:
            list: the operands
        """
        return expression.args

    def free_symbols(self, expression) -> list:
        """
        Get all free symbols in an expression

        Args:
            expression (object): the expression to get the free symbols of

        Returns:
            list: the list of free symbols
        """
        return [str(var) for var in list(expression.free_symbols)]

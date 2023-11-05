"""
This module implements the joint interface to the algebraic systems
to be used.
"""

import abc

__all__ = ["Algebra"]


class Algebra(metaclass=abc.ABCMeta):
    """
    The base class of the algebra abstractions
    """

    @abc.abstractmethod
    def __init__(self):
        """
        The constructor of the algebra
        """

    @abc.abstractmethod
    def create_symbol(self, name: str, **kwargs):
        """
        Create a symbol in the algebra with the specified name and assumptions

        Args:
            name (str): the name of the symbol
            kwargs (dict): the assumptions

        Returns:
            object: the symbol
        """

    @abc.abstractmethod
    def num_denom(self, expression):
        """
        Extract the numerator and denominator

        Args:
            expression (object): the expression to process

        Returns:
            object, object: the numerator and denominator
        """

    @abc.abstractmethod
    def simplify(self, expression):
        """
        Simplify the expression

        Args:
            expression (object): the expression to simplify

        Returns:
            object: the symplified expression
        """

    @abc.abstractmethod
    def solve(self, equation, var):
        """
        Solve an equation for a variable

        Args:
            equation (object): the equation to solve
            var (object): the variable to solve for

        Returns:
            list(dict): the solutions
        """

    @abc.abstractmethod
    def subs(self, expression, subs_dict):
        """
        Substitute a substitution into the expression

        Args:
            expression (object): the expression to substitute into
            subs_dict (dict): the substitution

        Returns:
            object: the result of the substitution
        """

    @abc.abstractmethod
    def args(self, expression) -> list:
        """
        The list of arguments

        Args:
            expression (object): the expression to process

        Returns:
            list: the list of arguments
        """

    @abc.abstractmethod
    def is_trivial(self, expression) -> bool:
        """
        Checks if the expression is trivial

        Args:
            expression (object): the expression to check

        Returns:
            bool: True if the expression is trivial, False otherwise
        """

    @abc.abstractmethod
    def is_root(self, expression) -> bool:
        """
        Checks if the expression is a root

        Args:
            expression (object): the expression to check if it is a root

        Returns:
            bool: True if the expression is a root, False otherwise
        """

    @abc.abstractmethod
    def is_division(self, expression) -> bool:
        """
        Checks if the expression is a division

        Args:
            expression (object): the expression to check if it is a division

        Returns:
            bool: True if the expression is a division, False otherwise
        """

    @abc.abstractmethod
    def is_power(self, expression) -> bool:
        """
        Checks if the expression is a power

        Args:
            expression (object): the expression to check if it is a power

        Returns:
            bool: True if the expression is a power, False otherwise
        """

    @abc.abstractmethod
    def operands(self, expression) -> list:
        """
        Returns the list of operands

        Args:
            expression (object): the expression to return the operands of

        Returns:
            list: the operands
        """

    @abc.abstractmethod
    def free_symbols(self, expression) -> list:
        """
        Get all free symbols in an expression

        Args:
            expression (object): the expression to get the free symbols of

        Returns:
            list: the list of free symbols
        """

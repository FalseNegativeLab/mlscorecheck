"""
This module implements the sage algebra wrapper

Sage works only on Linux and MacOS, also requires some preparation there,
so the sage-related codes are excluded from the coverage report.
"""

__all__ = ["SageAlgebra"]

import importlib

from ._algebra import Algebra


class SageAlgebra(Algebra):  # pragma: no cover
    """
    The required algebra driven by sage
    """

    def __init__(self):  # pragma: no cover
        """
        Constructor of the algebra
        """
        Algebra.__init__(self)  # pragma: no cover

        self.algebra = importlib.import_module("sage.all")  # pragma: no cover
        self.sqrt = self.algebra.sqrt  # pragma: no cover

    def create_symbol(self, name, **kwargs):  # pragma: no cover
        """
        Create a symbol in the algebra with the specified name and assumptions

        Args:
            name (str): the name of the symbol
            kwargs (dict): the assumptions

        Returns:
            object: the symbol
        """
        var = self.algebra.var(name)  # pragma: no cover
        if kwargs.get("nonnegative", False):  # pragma: no cover
            self.algebra.assume(var >= 0)  # pragma: no cover
        if kwargs.get("positive", False):  # pragma: no cover
            self.algebra.assume(var > 0)  # pragma: no cover
        if kwargs.get("negative", False):  # pragma: no cover
            self.algebra.assume(var < 0)  # pragma: no cover
        if kwargs.get("nonpositive", False):  # pragma: no cover
            self.algebra.assume(var <= 0)  # pragma: no cover
        if kwargs.get("real", False):  # pragma: no cover
            self.algebra.assume(var, "real")  # pragma: no cover
        if kwargs.get("upper_bound", None) is not None:  # pragma: no cover
            self.algebra.assume(var <= kwargs["upper_bound"])  # pragma: no cover
        if kwargs.get("lower_bound", None) is not None:  # pragma: no cover
            self.algebra.assume(var >= kwargs["lower_bound"])  # pragma: no cover

        return var  # pragma: no cover

    def num_denom(self, expression):  # pragma: no cover
        """
        Extract the numerator and denominator

        Args:
            expression (object): the expression to process

        Returns:
            object, object: the numerator and denominator
        """
        return expression.numerator(), expression.denominator()  # pragma: no cover

    def simplify(self, expression):  # pragma: no cover
        """
        Simplify the expression

        Args:
            expression (object): the expression to simplify

        Returns:
            object: the symplified expression
        """
        return self.algebra.factor(expression)  # pragma: no cover

    def solve(self, equation, var, **kwargs):  # pragma: no cover
        """
        Solve an equation for a variable

        Args:
            equation (object): the equation to solve
            var (object): the variable to solve for
            kwargs (dict): additional parameters to the solver

        Returns:
            list(dict): the solutions
        """
        results = self.algebra.solve(equation, var, **kwargs)  # pragma: no cover
        solutions = []  # pragma: no cover
        for sol in results:  # pragma: no cover
            solution = {sol.lhs(): self.algebra.factor(sol.rhs())}  # pragma: no cover
            solutions.append(solution)  # pragma: no cover
        return solutions  # pragma: no cover

    def subs(self, expression, subs_dict):  # pragma: no cover
        """
        Substitute a substitution into the expression

        Args:
            expression (object): the expression to substitute into
            subs_dict (dict): the substitution

        Returns:
            object: the result of the substitution
        """
        return expression.subs(subs_dict)  # pragma: no cover

    def args(self, expression) -> list:  # pragma: no cover
        """
        The list of arguments

        Args:
            expression (object): the expression to process

        Returns:
            list: the list of arguments
        """
        return set(expression.args())  # pragma: no cover

    def is_trivial(self, expression) -> bool:  # pragma: no cover
        """
        Checks if the expression is trivial

        Args:
            expression (object): the expression to check

        Returns:
            bool: True if the expression is trivial, False otherwise
        """
        return (
            True if expression is None else expression.is_trivially_equal(1)
        )  # pragma: no cover

    def is_root(self, expression) -> bool:  # pragma: no cover
        """
        Checks if the expression is a root

        Args:
            expression (object): the expression to check if it is a root

        Returns:
            bool: True if the expression is a root, False otherwise
        """
        if self.is_power(expression):  # pragma: no cover
            _, exponent = expression.operands()  # pragma: no cover
            if 0 < exponent < 1:  # pragma: no cover
                return True  # pragma: no cover
        return False  # pragma: no cover

    def is_power(self, expression) -> bool:  # pragma: no cover
        """
        Checks whether the expression is a power

        Args:
            expression (object): the expression to check

        Returns:
            bool: whether the expression is a power
        """
        return bool(
            hasattr(expression.operator(), "__qualname__")
            and expression.operator().__qualname__ == "pow"
        )  # pragma: no cover

    def is_division(self, expression) -> bool:  # pragma: no cover
        """
        Checks whether the expression is a division

        Args:
            expression (object): the expression to check

        Returns:
            bool: whether the expression is a division
        """
        if self.is_power(expression):  # pragma: no cover
            operands = expression.operands()  # pragma: no cover
            if operands[1] < 0:  # pragma: no cover
                return True  # pragma: no cover

        if (
            hasattr(expression.operator(), "__qualname__")
            and expression.operator().__qualname__ == "mul_vararg"
        ):  # pragma: no cover
            operands = expression.operands()  # pragma: no cover

            if len(operands) == 2:  # pragma: no cover
                if self.is_power(operands[1]):  # pragma: no cover
                    _, power = operands[1].operands()  # pragma: no cover
                    if power < 0:  # pragma: no cover
                        return True  # pragma: no cover
                elif self.is_power(operands[0]):  # pragma: no cover
                    _, power = operands[0].operands()  # pragma: no cover
                    if power < 0:  # pragma: no cover
                        return True  # pragma: no cover
        return False  # pragma: no cover

    def operands(self, expression) -> list:  # pragma: no cover
        """
        Returns the list of operands

        Args:
            expression (object): the expression to return the operands of

        Returns:
            list: the operands
        """
        return tuple(expression.operands())  # pragma: no cover

    def free_symbols(self, expression) -> list:  # pragma: no cover
        """
        Get all free symbols in an expression

        Args:
            expression (object): the expression to get the free symbols of

        Returns:
            list: the list of free symbols
        """
        return [
            str(var) for var in list(expression.free_variables())
        ]  # pragma: no cover

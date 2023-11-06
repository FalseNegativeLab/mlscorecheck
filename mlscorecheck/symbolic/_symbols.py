"""
This module implements the wrapper for the symbols
"""

__all__ = ["Symbols"]

from ._sympy_algebra import SympyAlgebra

from ._sage_algebra import SageAlgebra


class Symbols:  # pylint: disable=too-many-instance-attributes
    """
    A symbols class representing the basic symbols to be used
    """

    def __init__(self, algebraic_system):
        """
        The constructor of the object

        Args:
            algebraic_system ('sympy'/'sage'): the algebraic system to be used
        """
        self.algebraic_system = algebraic_system
        if algebraic_system == "sympy":
            self.algebra = SympyAlgebra()
        elif algebraic_system == "sage":  # pragma: no cover
            self.algebra = SageAlgebra()  # pragma: no cover

        self.tp = self.algebra.create_symbol("tp", nonnegative=True, real=True)
        self.tn = self.algebra.create_symbol("tn", nonnegative=True, real=True)
        self.p = self.algebra.create_symbol("p", positive=True, real=True)
        self.n = self.algebra.create_symbol("n", positive=True, real=True)
        self.beta_positive = self.algebra.create_symbol(
            "beta_positive", positive=True, real=True
        )
        self.beta_negative = self.algebra.create_symbol(
            "beta_negative", positive=True, real=True
        )
        self.sqrt = self.algebra.sqrt

    def get_algebra(self):
        """
        Returns the algebra

        Returns:
            Algebra: the algebra object
        """
        return self.algebra

    def to_dict(self) -> dict:
        """
        Returns a dictionary representation

        Returns:
            dict: the dictionary representation
        """
        return {
            "tp": self.tp,
            "tn": self.tn,
            "p": self.p,
            "n": self.n,
            "beta_positive": self.beta_positive,
            "beta_negative": self.beta_negative,
            "sqrt": self.sqrt,
        }

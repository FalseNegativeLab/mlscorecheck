"""
This module implements the complex interval arithmetics
"""

from ._interval import Interval, IntervalUnion

__all__ = ["ComplexInterval"]


class ComplexInterval:
    """
    The class represents a complex interval
    """

    def __init__(self, real, imag):
        """
        Constructor of the complex interval

        Args:
            real (float|Interval|IntervalUnion): the real part
            imag (float|Interval|IntervalUnion): the imaginary part
        """
        self.real = real
        self.imag = imag

    def __add__(self, other):
        """
        Addition operator

        Args:
            other (float|Interval|IntervalUnion|ComplexInterval): the object to add

        Returns:
            ComplexInterval: the sum of the objects
        """
        if not isinstance(other, ComplexInterval):
            return ComplexInterval(self.real + other, self.imag)

        return ComplexInterval(self.real + other.real, self.imag + other.imag)

    def __radd__(self, other):
        """
        Right addition operator

        Args:
            other (float|Interval|IntervalUnion|ComplexInterval): the object to add

        Returns:
            ComplexInterval: the sum of the objects
        """
        return self + other

    def __sub__(self, other):
        """
        Subtraction operator

        Args:
            other (float|Interval|IntervalUnion|ComplexInterval): the object to subtract

        Returns:
            ComplexInterval: the difference of the objects
        """
        if not isinstance(other, ComplexInterval):
            return ComplexInterval(self.real - other, self.imag)

        return ComplexInterval(self.real - other.real, self.imag - other.imag)

    def __rsub__(self, other):
        """
        Right subtraction operator

        Args:
            other (float|Interval|IntervalUnion|ComplexInterval): the object to subtract from

        Returns:
            ComplexInterval: the difference of the objects
        """
        return (-1) * self + other

    def __mul__(self, other):
        """
        Multiplication operator

        Args:
            other (float|Interval|IntervalUnion|ComplexInterval): the object to multiply by

        Returns:
            ComplexInterval: the product of the objects
        """
        if not isinstance(other, ComplexInterval):
            return ComplexInterval(self.real * other, self.imag * other)

        return ComplexInterval(
            self.real * other.real - self.imag * other.imag,
            self.real * other.imag + self.imag * other.real,
        )

    def __rmul__(self, other):
        """
        Right multiplication operator

        Args:
            other (float|Interval|IntervalUnion|ComplexInterval): the object to multiply

        Returns:
            ComplexInterval: the product of the objects
        """
        return self.__mul__(other)

    def __truediv__(self, other):
        """
        Division operator

        Args:
            other (float|Interval|IntervalUnion|ComplexInterval): the object to divide by

        Returns:
            ComplexInterval: the ratio of the objects
        """
        if not isinstance(other, ComplexInterval):
            return ComplexInterval(self.real / other, self.imag / other)

        norm = other.real**2 + other.imag**2

        return ComplexInterval(
            (self.real * other.real + self.imag * other.imag) / norm,
            (self.imag * other.real - self.real * other.imag) / norm,
        )

    def __rtruediv__(self, other):
        """
        Right division operator

        Args:
            other (float|Interval|IntervalUnion|ComplexInterval): the object to divide

        Returns:
            ComplexInterval: the ratio of the objects
        """
        if not isinstance(other, ComplexInterval):
            if not isinstance(other, (Interval, IntervalUnion)):
                other = Interval(other, other)
            other = ComplexInterval(other, Interval(0, 0))

        return other / self

    def __repr__(self):
        """
        The representation dunder

        Returns:
            str: the string representation of the complex interval
        """
        return str(f"[{str(self.real)}, {str(self.imag)}]")

    def __eq__(self, other) -> bool:
        """
        The equality operator

        Args:
            other (float|Interval|IntervalUnion|ComplexInterval): the object to compare to

        Returns:
            bool: True if the objects are equal, False otherwise
        """
        if not isinstance(other, ComplexInterval):
            return False

        return self.real == other.real and self.imag == other.imag

    def __ne__(self, other) -> bool:
        """
        The non-equality operator

        Args:
            other (float|Interval|IntervalUnion|ComplexInterval): the object to compare to

        Returns:
            bool: False if the objects are equal, True otherwise
        """
        return not self.__eq__(other)

    def __neg__(self):
        """
        The negation operator

        Returns:
            ComplexInterval: the negated interval
        """
        return ComplexInterval(-self.real, -self.imag)

    def __pow__(self, other):
        """
        The power operator

        Args:
            other (float|Interval|IntervalUnion|ComplexInterval): the exponent

        Returns:
            ComplexInterval: the complex interval raised to the given power
        """

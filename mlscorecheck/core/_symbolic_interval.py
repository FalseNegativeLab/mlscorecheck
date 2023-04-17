"""
This module implements the interval arithmetics
"""

import numpy as np
import sympy

from ._interval import Interval, IntervalUnion

__all__ = ['SymbolicInterval',
           'SymbolicIntervalUnion']

class SymbolicInterval:
    """
    The symbolic interval abstraction
    """

    def __init__(self, lower_bound, upper_bound, condition=True):
        """
        Constructor of the interval

        Args:
            lower_bound (Symbol/Expression/int/float): the lower bound
            upper_bound (Symbol/Expression/int/float): the upper bound
            condition (bool/Expression): the condition for the interval
        """
        self.lower_bound = sympy.simplify(lower_bound)
        self.upper_bound = sympy.simplify(upper_bound)
        self.condition = condition

    def to_interval(self, subst):
        """
        Convert the symbolic interval to a real interval by substitution

        Args:
            subst (dict): the substitution

        Returns:
            Interval: the real interval
        """
        return Interval(self.lower_bound.subs(subst) if self.lower_bound != -np.inf else -np.inf,
                        self.upper_bound.subs(subst) if self.upper_bound != np.inf else np.inf)

    def check_condition(self, subst):
        """
        Check if the condition is true with the substitution

        Args:
            subst (dict): the substitution

        Returns:
            bool: whether the condition is True
        """
        if isinstance(self.condition, (bool, sympy.logic.boolalg.BooleanTrue)):
            return (self.condition == True)
        return self.condition.subs(subst)

    def __add__(self, other):
        """
        The addition operator

        Args:
            other (int/float/Symbol/Expression/SymbolicInterval): an object to add

        Returns:
            SymbolicInterval: the sum
        """

        if isinstance(other, SymbolicIntervalUnion):
            return other + self

        if not isinstance(other, SymbolicInterval):
            return SymbolicInterval(lower_bound=self.lower_bound + other,
                                    upper_bound=self.upper_bound + other)

        return SymbolicInterval(lower_bound=self.lower_bound + other.lower_bound,
                                upper_bound=self.upper_bound + other.upper_bound,
                                condition=self.condition & other.condition)

    def __radd__(self, other):
        """
        The right hand addition

        Args:
            other (int/float/Interval): an object to add

        Returns:
            SymbolicInterval: the sum
        """

        return self + other

    def __sub__(self, other):
        """
        The subtraction operator

        Args:
            other (int/float/Symbol/Expression/SymbolicInterval): an object to subtract

        Returns:
            SymbolicInterval: the difference
        """

        if isinstance(other, SymbolicIntervalUnion):
            return (-1)*other + self

        if not isinstance(other, SymbolicInterval):
            return SymbolicInterval(lower_bound=self.lower_bound - other,
                                    upper_bound=self.upper_bound - other)

        return SymbolicInterval(lower_bound=self.lower_bound - other.upper_bound,
                                upper_bound=self.upper_bound - other.lower_bound,
                                condition=self.condition & other.condition)

    def __rsub__(self, other):
        """
        The right hand subtraction

        Args:
            other (int/float/Symbol/Expression/SymbolicInterval): an object to subtract from

        Returns:
            Interval: the difference
        """

        return SymbolicInterval(-self.upper_bound,
                                -self.lower_bound) + other

    def __mul__(self, other):
        """
        The multiplication operator

        Args:
            other (int/float/Interval): an object to multiply

        Returns:
            SymbolicInterval: the product
        """

        if isinstance(other, SymbolicIntervalUnion):
            return other * self

        if not isinstance(other, SymbolicInterval):
            return SymbolicIntervalUnion([SymbolicInterval(lower_bound=self.lower_bound * other,
                                                        upper_bound=self.upper_bound * other,
                                                        condition = self.condition & (other >= 0)),
                                         SymbolicInterval(lower_bound=self.upper_bound * other,
                                                        upper_bound=self.lower_bound * other,
                                                        condition = self.condition & (other < 0))])

        terms = [self.lower_bound * other.lower_bound,
                 self.lower_bound * other.upper_bound,
                 self.upper_bound * other.lower_bound,
                 self.upper_bound * other.upper_bound]

        terms = list(set(terms))

        intervals = []

        for lower_idx, lower in enumerate(terms):
            for upper_idx, upper in enumerate(terms):
                if lower_idx != upper_idx:
                    lower_constraint = sympy.simplify(True)
                    upper_constraint = sympy.simplify(True)
                    for lower_tmp_idx, lower_tmp in enumerate(terms):
                        if lower_tmp_idx != lower_idx:
                            lower_constraint = lower_constraint & (lower <= lower_tmp)
                    for upper_tmp_idx, upper_tmp in enumerate(terms):
                        if upper_tmp_idx != upper_idx:
                            upper_constraint = upper_constraint & (upper >= upper_tmp)

                    condition = self.condition & other.condition & \
                                (lower_constraint & upper_constraint)

                    intervals.append(SymbolicInterval(lower,
                                                      upper,
                                                      condition))
        return SymbolicIntervalUnion(intervals)

    def __rmul__(self, other):
        """
        The right hand multiplication operator

        Args:
            other (int/float/Symbol/Expression/SymbolicInterval): an object to multiply

        Returns:
            SymbolicIntervalUnion: the product
        """

        return self.__mul__(other)

    def __truediv__(self, other):
        """
        The division operator

        Args:
            other (int/float/Symbol/Expression/SymbolicInterval): an object to divide with

        Returns:
            SymbolicIntervalUnion: the ratio
        """

        if isinstance(other, SymbolicIntervalUnion):
            return (1.0/other) * self

        if not isinstance(other, SymbolicInterval):
            pos = SymbolicInterval(lower_bound=self.lower_bound/other,
                                   upper_bound=self.upper_bound/other,
                                   condition=self.condition & (other >= 0))

            neg = SymbolicInterval(lower_bound=self.upper_bound/other,
                                   upper_bound=self.lower_bound/other,
                                   condition=self.condition & (other < 0))

            return SymbolicIntervalUnion([pos, neg])

        intervals = []

        condition = self.condition & other.condition & \
                    ((other.lower_bound > 0) | (other.upper_bound < 0))
        intervals.extend((self * SymbolicInterval(1.0/other.upper_bound,
                                                 1.0/other.lower_bound,
                                                 condition=condition)).intervals)

        condition = (self.condition \
                    & other.condition \
                    & sympy.Eq(other.upper_bound, 0) \
                    & ((other.lower_bound < 0) | (other.lower_bound > 0)))
        intervals.extend((self * SymbolicInterval(-np.inf,
                                                 1.0/other.lower_bound,
                                                 condition=condition)).intervals)

        condition = (self.condition \
                    & other.condition\
                    & sympy.Eq(other.lower_bound, 0) \
                    & ((other.upper_bound < 0) | (other.upper_bound > 0)))
        intervals.extend((self * SymbolicInterval(1.0/other.upper_bound,
                                                 np.inf,
                                                 condition=condition)).intervals)

        condition = (self.condition \
                    & other.condition \
                    & (other.lower_bound < 0)\
                    & (other.upper_bound > 0))
        intervals.extend((self * SymbolicInterval(-np.inf,
                                                 1.0/other.lower_bound,
                                                 condition=condition)).intervals)
        intervals.extend((self * SymbolicInterval(1.0/other.upper_bound,
                                                 np.inf,
                                                 condition=condition)).intervals)

        return SymbolicIntervalUnion(intervals)

    def __rtruediv__(self, other):
        """
        The right hand division operator

        Args:
            other (int/float/Symbol/Expression/SymbolicInterval): an object to divide

        Returns:
            SymbolicInterval: the ratio
        """

        if not isinstance(other, (SymbolicInterval, SymbolicIntervalUnion)):
            other = SymbolicInterval(other, other)

        return other.__truediv__(self)

    def __repr__(self):
        """
        String representation of the object

        Returns:
            str: the representation
        """
        return str(f"[{str(self.lower_bound)},\n{str(self.upper_bound)}]"
                   f"\nsubject to {self.condition}")

    def __eq__(self, other):
        """
        Equality operator

        Args:
            other (obj): the object to compare to

        Returns:
            bool: whether the objects equal
        """
        if not isinstance(other, SymbolicInterval):
            return False
        return self.lower_bound == other.lower_bound \
                and self.upper_bound == other.upper_bound \
                and self.condition == other.condition

    def __ne__(self, other):
        """
        Non-equality operator

        Args:
            other (obj): the object to compare to

        Returns:
            bool: whether the objects are not equal
        """
        return not self.__eq__(other)

    def __hash__(self):
        return self.lower_bound.__hash__() + self.upper_bound.__hash__()

class SymbolicIntervalUnion:
    """
    Interval union for symbolic intervals
    """

    def __init__(self, intervals):
        """
        Constructor of the object

        Args:
            intervals (list(SymbolicInterval)): a list of symbolic intervals
        """
        intervals = list(set(intervals))
        self.intervals = [interval for interval in intervals
                          if interval.condition != False]

    def to_interval_union(self, subst):
        """
        Convert the symbolic interval union into a real interval union by substitution

        Args:
            subst (dict): the substitution

        Returns:
            IntervalUnion: the real interval union
        """
        return IntervalUnion([interval.to_interval(subst)
                              for interval in self.intervals
                              if ((not isinstance(interval.condition, bool))
                                  and interval.condition.subs(subst))
                              or (interval.condition == True)
                              ])

    def condition_mask(self, subst):
        """
        Return the mask of conditions fulfilled by the substitution

        Args:
            subst (dict): a substitution

        Returns:
            list(bool): the list of flags
        """
        return [interval.condition.subs(subst)
                if (not isinstance(interval.condition, bool))
                else interval.condition
                for interval in self.intervals]

    def __add__(self, other):
        """
        The addition operator

        Args:
            other (int/float/SymbolicInterval/SymbolicIntervalUnion): an object to add

        Returns:
            IntervalUnion: the sum
        """

        if not isinstance(other, (SymbolicIntervalUnion, )):
            return SymbolicIntervalUnion([interval + other for interval in self.intervals])

        intervals = []
        for int0 in self.intervals:
            for int1 in other.intervals:
                intervals.append(int0 + int1)

        return SymbolicIntervalUnion(intervals)

    def __radd__(self, other):
        """
        The right hand addition

        Args:
            other (int/float/SymbolicInterval/SymbolicIntervalUnion): an object to add

        Returns:
            IntervalUnion: the sum
        """

        return self.__add__(other)

    def __sub__(self, other):
        """
        The subtraction operator

        Args:
            other (int/float/SymbolicInterval/SymbolicIntervalUnion): an object to subtract

        Returns:
            SymbolicIntervalUnion: the difference
        """

        if not isinstance(other, (SymbolicIntervalUnion,)):
            return SymbolicIntervalUnion([interval - other for interval in self.intervals])

        intervals = []
        for int0 in self.intervals:
            for int1 in other.intervals:
                intervals.append(int0 - int1)

        return SymbolicIntervalUnion(intervals)

    def __rsub__(self, other):
        """
        The right hand subtraction

        Args:
            other (int/float/SymbolicInterval/SymbolicIntervalUnion): an object to subtract from

        Returns:
            SymbolicIntervalUnion: the difference
        """

        return -1 * self + other

    def __mul__(self, other):
        """
        The multiplication operator

        Args:
            other (int/float/SymbolicInterval/SymbolicIntervalUnion): an object to multiply

        Returns:
            SymbolicIntervalUnion: the product
        """

        if not isinstance(other, (SymbolicIntervalUnion)):
            intervals = []
            for interval in self.intervals:
                tmp = interval * other
                intervals.extend(tmp.intervals)
            return SymbolicIntervalUnion(intervals)

        intervals = []
        for int0 in self.intervals:
            for int1 in other.intervals:
                tmp = int0 * int1
                intervals.extend(tmp.intervals)

        return SymbolicIntervalUnion(intervals)

    def __rmul__(self, other):
        """
        The right hand multiplication operator

        Args:
            other (int/float/SymbolicInterval/SymbolicIntervalUnion): an object to multiply

        Returns:
            SymbolicIntervalUnion: the product
        """

        return self.__mul__(other)

    def __truediv__(self, other):
        """
        The division operator

        Args:
            other (int/float/SymbolicInterval/SymbolicIntervalUnion): an object to divide with

        Returns:
            SymbolicIntervalUnion: the ratio
        """

        if not isinstance(other, (SymbolicIntervalUnion)):
            intervals = []
            for interval in self.intervals:
                tmp = interval / other
                intervals.extend(tmp.intervals)

            return SymbolicIntervalUnion(intervals)

        intervals = []
        for int0 in self.intervals:
            for int1 in other.intervals:
                tmp = int0 / int1
                intervals.extend(tmp.intervals)

        return SymbolicIntervalUnion(intervals)

    def __rtruediv__(self, other):
        """
        The right hand division operator

        Args:
            other (int/float/SymbolicInterval/SymbolicIntervalUnion): an object to divide

        Returns:
            SymbolicIntervalUnion: the ratio
        """

        if not isinstance(other, (SymbolicInterval, SymbolicIntervalUnion)):
            other = SymbolicInterval(other, other)

        if not isinstance(other, (SymbolicIntervalUnion)):
            other = SymbolicIntervalUnion([other])

        return other / self

    def __repr__(self):
        """
        String representation of the object

        Returns:
            str: the representation
        """
        return ',\n'.join([f'[{interval.lower_bound}, {interval.upper_bound}]'
                           f'subject to {interval.condition}'
                           for interval in self.intervals])

    def __eq__(self, other):
        """
        Equality operator

        Args:
            other (obj): the object to compare to

        Returns:
            bool: whether the objects equal
        """
        if not isinstance(other, SymbolicIntervalUnion):
            return False

        int0 = [(sympy.simplify(interval.lower_bound), sympy.simplify(interval.upper_bound))
                for interval in self.intervals
                if (interval.condition == True) or not isinstance(interval.condition, bool)]
        int1 = [(sympy.simplify(interval.lower_bound), sympy.simplify(interval.upper_bound))
                for interval in other.intervals
                if (interval.condition == True) or not isinstance(interval.condition, bool)]

        int0 = list(set(int0))
        int1 = list(set(int1))

        res0 = [interval in int1 for interval in int0]
        res1 = [interval in int0 for interval in int1]

        return all(res0) and all(res1)

    def __ne__(self, other):
        """
        Non-equality operator

        Args:
            other (obj): the object to compare to

        Returns:
            bool: whether the objects are not equal
        """
        return not self.__eq__(other)

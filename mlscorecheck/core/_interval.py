"""
This module implements the interval arithmetics
"""

import numpy as np

__all__ = ['Interval',
            'IntervalUnion',
            'union']

def union(intervals):
    if len(intervals) == 1:
        return intervals[0]

    union_tmp = intervals[0]

    for idx in range(1, len(intervals)):
        union_tmp = union_tmp.union(intervals[idx])

    return union_tmp

class Interval:
    """
    The interval abstraction
    """
    def __init__(self, lower_bound, upper_bound):
        """
        Constructor of the interval

        Args:
            lower_bound (int/float): the lower bound
            upper_bound (int/float): the upper bound
        """
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def to_tuple(self):
        """
        Convert to tuple representation

        Returns:
            tuple: the interval
        """
        return (self.lower_bound, self.upper_bound)

    def contains(self, value):
        """
        Check if the interval contains the value

        Args:
            value (float/int): the value to check

        Returns:
            bool: True if the interval contains the value, otherwise False
        """
        return value >= self.lower_bound and value <= self.upper_bound

    def intersection(self, other):
        """
        Returns the intersection of two intervals

        Args:
            other (Interval): the other interval

        Returns:
            Interval: the intersection, [1, 0] if the intersection is empty
        """

        if self.lower_bound >= other.lower_bound and self.upper_bound <= other.upper_bound:
            return Interval(self.lower_bound, self.upper_bound)
        if self.lower_bound <= other.lower_bound and self.upper_bound >= other.upper_bound:
            return Interval(other.lower_bound, other.upper_bound)

        tmp_lower = self
        tmp_upper = other
        if tmp_upper.lower_bound < tmp_lower.lower_bound:
            tmp_lower, tmp_upper = tmp_upper, tmp_lower

        assert tmp_lower.lower_bound <= tmp_upper.lower_bound

        if tmp_lower.upper_bound >= tmp_upper.lower_bound:
            return Interval(tmp_upper.lower_bound, tmp_lower.upper_bound)

        return Interval(1, 0)

    def integer(self):
        """
        Check whether the interval fulfills integer conditions

        Returns:
            bool: whether the interval contains an integer
        """
        if self.upper_bound - self.lower_bound >= 1.0:
            return True

        return np.ceil(self.lower_bound) == np.floor(self.upper_bound)

    def shrink_to_integers(self):
        return Interval(np.ceil(self.lower_bound), np.floor(self.upper_bound))

    def is_empty(self):
        return self.upper_bound < self.lower_bound

    def __add__(self, other):
        """
        The addition operator

        Args:
            other (int/float/Interval): an object to add

        Returns:
            Interval: the sum
        """
        if isinstance(other, IntervalUnion):
            return other + self

        if not isinstance(other, Interval):
            return Interval(lower_bound=self.lower_bound + other,
                            upper_bound=self.upper_bound + other)

        return Interval(lower_bound=self.lower_bound + other.lower_bound,
                        upper_bound=self.upper_bound + other.upper_bound)

    def __radd__(self, other):
        """
        The right hand addition

        Args:
            other (int/float/Interval): an object to add

        Returns:
            Interval: the sum
        """
        return self + other

    def __sub__(self, other):
        """
        The subtraction operator

        Args:
            other (int/float/Interval): an object to subtract

        Returns:
            Interval: the difference
        """
        if isinstance(other, IntervalUnion):
            return (-1) * other + self

        if not isinstance(other, Interval):
            return Interval(lower_bound=self.lower_bound - other,
                            upper_bound=self.upper_bound - other)

        return Interval(lower_bound=self.lower_bound - other.upper_bound,
                        upper_bound=self.upper_bound - other.lower_bound)

    def __rsub__(self, other):
        """
        The right hand subtraction

        Args:
            other (int/float/Interval): an object to subtract from

        Returns:
            Interval: the difference
        """
        return (-1) * self + other

    def __mul__(self, other):
        """
        The multiplication operator

        Args:
            other (int/float/Interval): an object to multiply

        Returns:
            Interval: the product
        """
        if isinstance(other, IntervalUnion):
            return other * self

        if not isinstance(other, Interval):
            return Interval(lower_bound=min(self.lower_bound * other, self.upper_bound*other),
                            upper_bound=max(self.upper_bound * other, self.lower_bound*other))

        term0 = self.lower_bound * other.lower_bound
        term1 = self.lower_bound * other.upper_bound
        term2 = self.upper_bound * other.lower_bound
        term3 = self.upper_bound * other.upper_bound

        return Interval(np.min([term0, term1, term2, term3]),
                        np.max([term0, term1, term2, term3]))

    def __rmul__(self, other):
        """
        The right hand multiplication operator

        Args:
            other (int/float/Interval): an object to multiply

        Returns:
            Interval: the product
        """
        return self.__mul__(other)

    def __truediv__(self, other):
        """
        The division operator

        Args:
            other (int/float/Interval): an object to divide with

        Returns:
            Interval: the ratio
        """
        if isinstance(other, IntervalUnion):
            return (1.0/other) * self

        if not isinstance(other, Interval):
            if other >= 0:
                lower_bound = self.lower_bound / other
                upper_bound = self.upper_bound / other
            else:
                lower_bound = self.upper_bound / other
                upper_bound = self.lower_bound / other

            return Interval(lower_bound=lower_bound,
                            upper_bound=upper_bound)

        if (other.lower_bound > 0) or (other.upper_bound < 0):
            return self * Interval(1.0/other.upper_bound,
                                   1.0/other.lower_bound)

        if (other.upper_bound == 0) and (other.lower_bound != 0):
            return self * Interval(-np.inf, 1.0/other.lower_bound)

        if (other.lower_bound == 0) and (other.upper_bound != 0):
            return self * Interval(1.0/other.upper_bound, np.inf)

        res_0 = Interval(-np.inf, 1.0/other.lower_bound)
        res_1 = Interval(1.0/other.upper_bound, np.inf)

        return IntervalUnion([self * res_0, self * res_1])

    def __rtruediv__(self, other):
        """
        The right hand division operator

        Args:
            other (int/float/Interval): an object to divide

        Returns:
            Interval: the ratio
        """
        if not isinstance(other, (Interval, IntervalUnion)):
            other = Interval(other, other)

        return other / self

    def __repr__(self):
        """
        String representation of the object

        Returns:
            str: the representation
        """
        return str(f"({str(self.lower_bound)}, {str(self.upper_bound)})")

    def __eq__(self, other):
        """
        Equality operator

        Args:
            other (obj): the object to compare to

        Returns:
            bool: whether the objects equal
        """
        if not isinstance(other, Interval):
            return False
        return self.lower_bound == other.lower_bound and self.upper_bound == other.upper_bound

    def __ne__(self, other):
        """
        Non-equality operator

        Args:
            other (obj): the object to compare to

        Returns:
            bool: whether the objects are not equal
        """
        return not self.__eq__(other)

    def __neg__(self):
        return (-1)*self


class IntervalUnion:
    """
    The interval union abstraction
    """
    def __init__(self, intervals):
        """
        Constructor of the object

        Args:
            intervals (list(Interval)): a list of intervals
        """
        self.intervals = intervals
        if len(intervals) > 0:
            self.simplify()

    def to_tuple(self):
        """
        Convert to tuple representation

        Returns:
            list(tuple): the interval tuples
        """
        return [interval.to_tuple() for interval in self.intervals]

    def simplify(self):
        """
        Simplify the union of intervals

        TODO: a more efficient implementation would be desired
        """

        intervals = self.intervals

        # removing those intervals that are entirely contained by another one
        to_drop = []
        for idx, int0 in enumerate(intervals):
            for jdx, int1 in enumerate(intervals):
                if idx < jdx:
                    if (int0.lower_bound <= int1.lower_bound) \
                            and (int0.upper_bound >= int1.upper_bound):
                        # the interval widh index jdx is contained in the interval with index idx
                        to_drop.append(jdx)

        intervals = [interval for idx, interval in enumerate(intervals) if not idx in to_drop]

        # chaining the intervals
        sorted_intervals = sorted(intervals, key=lambda x: x.lower_bound)

        final_intervals = []
        interval = sorted_intervals[0]

        for idx in range(1, len(sorted_intervals)):
            if interval.upper_bound >= sorted_intervals[idx].lower_bound:
                interval = Interval(interval.lower_bound, sorted_intervals[idx].upper_bound)
            else:
                final_intervals.append(interval)
                interval = sorted_intervals[idx]
        final_intervals.append(interval)

        self.intervals = final_intervals

    def contains(self, value):
        """
        Check if the interval contains the value

        Args:
            value (float/int): the value to check

        Returns:
            bool: True if the interval contains the value, otherwise False
        """
        return any([interval.contains(value) for interval in self.intervals])

    def intersection(self, other):
        """
        Returns the intersection of two intervals

        Args:
            other (Interval/IntervalUnion): the other interval

        Returns:
            IntervalUnion: the intersection, empty interval union if the intersection
                           is empty
        """
        if isinstance(other, Interval):
            new_intervals = [other.intersection(interval) for interval in self.intervals]
            new_intervals = [interval for interval in new_intervals
                                      if not interval == Interval(1, 0)]
            return IntervalUnion(new_intervals)

        intersections = []
        for int0 in self.intervals:
            for int1 in other.intervals:
                intersections.append(int0.intersection(int1))

        return IntervalUnion([interval for interval in intersections
                                        if not interval == Interval(1, 0)])

    def integer(self):
        """
        Check whether the interval fulfills integer conditions

        Returns:
            bool: whether the interval contains an integer
        """

        return any([interval.integer() for interval in self.intervals])

    def shrink_to_integer(self):
        return IntervalUnion([interval.shrink_to_integer() for interval in self.intervals])

    def is_empty(self):
        return all(interval.is_empty() for interval in self.intervals)

    def __add__(self, other):
        """
        The addition operator

        Args:
            other (int/float/Interval/IntervalUnion): an object to add

        Returns:
            IntervalUnion: the sum
        """
        if not isinstance(other, (IntervalUnion,)):
            return IntervalUnion([interval + other for interval in self.intervals])

        intervals = []
        for int0 in self.intervals:
            for int1 in other.intervals:
                intervals.append(int0 + int1)

        return IntervalUnion(intervals)

    def __radd__(self, other):
        """
        The right hand addition

        Args:
            other (int/float/Interval/IntervalUnion): an object to add

        Returns:
            IntervalUnion: the sum
        """
        return self + other

    def __sub__(self, other):
        """
        The subtraction operator

        Args:
            other (int/float/Interval/IntervalUnion): an object to subtract

        Returns:
            IntervalUnion: the difference
        """
        if not isinstance(other, (IntervalUnion,)):
            return IntervalUnion([interval - other for interval in self.intervals])

        intervals = []
        for int0 in self.intervals:
            for int1 in other.intervals:
                intervals.append(int0 - int1)

        return IntervalUnion(intervals)

    def __rsub__(self, other):
        """
        The right hand subtraction

        Args:
            other (int/float/IntervalUnion): an object to subtract from

        Returns:
            IntervalUnion: the difference
        """
        return -1 * self + other

    def __mul__(self, other):
        """
        The multiplication operator

        Args:
            other (int/float/IntervalUnion): an object to multiply

        Returns:
            IntervalUnion: the product
        """

        if not isinstance(other, (IntervalUnion)):
            return IntervalUnion([interval * other for interval in self.intervals])

        intervals = []
        for int0 in self.intervals:
            for int1 in other.intervals:
                intervals.append(int0 * int1)

        return IntervalUnion(intervals)

    def __rmul__(self, other):
        """
        The right hand multiplication operator

        Args:
            other (int/float/IntervalUnion): an object to multiply

        Returns:
            IntervalUnion: the product
        """
        return self.__mul__(other)

    def __truediv__(self, other):
        """
        The division operator

        Args:
            other (int/float/IntervalUnion): an object to divide with

        Returns:
            IntervalUnion: the ratio
        """

        if not isinstance(other, (IntervalUnion)):
            intervals = []
            for interval in self.intervals:
                tmp = interval / other
                if isinstance(tmp, Interval):
                    intervals.append(tmp)
                else:
                    intervals.extend(tmp.intervals)
            return IntervalUnion(intervals)

        intervals = []
        for int0 in self.intervals:
            for int1 in other.intervals:
                tmp = int0 / int1
                if isinstance(tmp, Interval):
                    intervals.append(tmp)
                else:
                    intervals.extend(tmp.intervals)

        return IntervalUnion(intervals)

    def __rtruediv__(self, other):
        """
        The right hand division operator

        Args:
            other (int/float/IntervalUnion): an object to divide

        Returns:
            IntervalUnion: the ratio
        """
        if not isinstance(other, (Interval, IntervalUnion)):
            other = Interval(other, other)

        if not isinstance(other, IntervalUnion):
            other = IntervalUnion([other])

        return other / self

    def __neg__(self):
        return (-1)*self

    def __repr__(self):
        """
        String representation of the object

        Returns:
            str: the representation
        """
        return ' | '.join([f'({interval.lower_bound}, {interval.upper_bound})'
                                                for interval in self.intervals])

    def __eq__(self, other):
        """
        Equality operator

        Args:
            other (obj): the object to compare to

        Returns:
            bool: whether the objects equal
        """
        if not isinstance(other, IntervalUnion):
            return False

        if len(self.intervals) != len(other.intervals):
            return False

        for int0, int1 in zip(self.intervals, other.intervals):
            if int0 != int1:
                return False

        return True

    def __ne__(self, other):
        """
        Non-equality operator

        Args:
            other (obj): the object to compare to

        Returns:
            bool: whether the objects are not equal
        """
        return not self.__eq__(other)

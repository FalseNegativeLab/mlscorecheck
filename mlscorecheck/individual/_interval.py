"""
This module implements the interval arithmetics
"""

import numpy as np

__all__ = ["Interval", "IntervalUnion", "sqrt"]


def sqrt(obj):
    """
    Square root of an interval or interval union

    Args:
        obj (Interval|IntervalUnion|numeric): the object to take the square root of

    Returns:
        Interval|IntervalUnion|numeric: the square root of the parameter
    """

    result = obj**0.5 if isinstance(obj, (Interval, IntervalUnion)) else obj ** 0.5

    return result


class Interval:
    """
    The interval abstraction
    """

    def __init__(self, lower_bound, upper_bound):
        """
        Constructor of the interval

        Args:
            lower_bound (int|float): the lower bound
            upper_bound (int|float): the upper bound
        """
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        if np.isnan(self.lower_bound) or np.isnan(self.upper_bound):
            self.lower_bound = 1
            self.upper_bound = 0

        if self.lower_bound > self.upper_bound:
            self.lower_bound = 1
            self.upper_bound = 0

    def to_tuple(self) -> tuple:
        """
        Convert to tuple representation

        Returns:
            tuple: the interval
        """
        return (self.lower_bound, self.upper_bound)

    def contains(self, value) -> bool:
        """
        Check if the interval contains the value

        Args:
            value (float|int): the value to check

        Returns:
            bool: True if the interval contains the value, otherwise False
        """
        return self.lower_bound <= value <= self.upper_bound

    def intersection(self, other):
        """
        Returns the intersection of two intervals

        Args:
            other (Interval): the other interval

        Returns:
            Interval: the intersection, [1, 0] if the intersection is empty
        """
        if isinstance(other, IntervalUnion):
            return IntervalUnion([self]).intersection(other)

        if (
            self.lower_bound >= other.lower_bound
            and self.upper_bound <= other.upper_bound
        ):
            return Interval(self.lower_bound, self.upper_bound)
        if (
            self.lower_bound <= other.lower_bound
            and self.upper_bound >= other.upper_bound
        ):
            return Interval(other.lower_bound, other.upper_bound)

        tmp_lower = self
        tmp_upper = other
        if tmp_upper.lower_bound < tmp_lower.lower_bound:
            tmp_lower, tmp_upper = tmp_upper, tmp_lower

        assert tmp_lower.lower_bound <= tmp_upper.lower_bound

        if tmp_lower.upper_bound >= tmp_upper.lower_bound:
            return Interval(tmp_upper.lower_bound, tmp_lower.upper_bound)

        return Interval(1, 0)

    def integer(self) -> int:
        """
        Check whether the interval fulfills integer conditions

        Returns:
            bool: whether the interval contains an integer
        """
        if self.upper_bound - self.lower_bound >= 1.0:
            return True

        return np.ceil(self.lower_bound) == np.floor(self.upper_bound)

    def shrink_to_integers(self):
        """
        Shrinks the interval to integers

        Returns:
            Interval: the interval shrunk to integers
        """
        return Interval(int(np.ceil(self.lower_bound)), int(np.floor(self.upper_bound)))

    def integer_counts(self) -> int:
        """
        Returns the number of integers in the interval

        Returns:
            int: the number of integers in the interval
        """
        if not self.is_empty():
            integer = self.shrink_to_integers()
            return integer.upper_bound - integer.lower_bound + 1
        return 0

    def is_empty(self) -> bool:
        """
        Checks if the interval is empty

        Returns:
            bool: True if the interval is empty, False otherwise
        """
        return self.upper_bound < self.lower_bound

    def __add__(self, other):
        """
        The addition operator

        Args:
            other (int|float|Interval): an object to add

        Returns:
            Interval: the sum
        """
        if isinstance(other, IntervalUnion):
            return other + self

        if not isinstance(other, Interval):
            return Interval(
                lower_bound=self.lower_bound + other,
                upper_bound=self.upper_bound + other,
            )

        return Interval(
            lower_bound=self.lower_bound + other.lower_bound,
            upper_bound=self.upper_bound + other.upper_bound,
        )

    def __radd__(self, other):
        """
        The right hand addition

        Args:
            other (int|float|Interval): an object to add

        Returns:
            Interval: the sum
        """
        return self + other

    def __sub__(self, other):
        """
        The subtraction operator

        Args:
            other (int|float|Interval): an object to subtract

        Returns:
            Interval: the difference
        """
        if isinstance(other, IntervalUnion):
            return (-1) * other + self

        if not isinstance(other, Interval):
            return Interval(
                lower_bound=self.lower_bound - other,
                upper_bound=self.upper_bound - other,
            )

        return Interval(
            lower_bound=self.lower_bound - other.upper_bound,
            upper_bound=self.upper_bound - other.lower_bound,
        )

    def __rsub__(self, other):
        """
        The right hand subtraction

        Args:
            other (int|float|Interval): an object to subtract from

        Returns:
            Interval: the difference
        """
        return (-1) * self + other

    def __mul__(self, other):
        """
        The multiplication operator

        Args:
            other (int|float|Interval): an object to multiply

        Returns:
            Interval: the product
        """
        if isinstance(other, IntervalUnion):
            return other * self

        if not isinstance(other, Interval):
            return Interval(
                lower_bound=min(self.lower_bound * other, self.upper_bound * other),
                upper_bound=max(self.upper_bound * other, self.lower_bound * other),
            )

        term0 = self.lower_bound * other.lower_bound
        term1 = self.lower_bound * other.upper_bound
        term2 = self.upper_bound * other.lower_bound
        term3 = self.upper_bound * other.upper_bound

        return Interval(
            min(term0, term1, term2, term3), max(term0, term1, term2, term3)
        )

    def __rmul__(self, other):
        """
        The right hand multiplication operator

        Args:
            other (int|float|Interval): an object to multiply

        Returns:
            Interval: the product
        """
        return self.__mul__(other)

    def __truediv__(self, other):
        """
        The division operator

        Args:
            other (int|float|Interval): an object to divide with

        Returns:
            Interval: the ratio
        """
        if isinstance(other, IntervalUnion):
            return (1.0 / other) * self

        if not isinstance(other, Interval):
            if other >= 0:
                lower_bound = self.lower_bound / other
                upper_bound = self.upper_bound / other
            else:
                lower_bound = self.upper_bound / other
                upper_bound = self.lower_bound / other

            return Interval(lower_bound=lower_bound, upper_bound=upper_bound)

        if (other.upper_bound == 0) and (other.lower_bound != 0):
            return self * Interval(-np.inf, 1.0 / other.lower_bound)

        if (other.lower_bound == 0) and (other.upper_bound != 0):
            return self * Interval(1.0 / other.upper_bound, np.inf)

        if (other.lower_bound > 0) or (other.upper_bound < 0):
            return self * Interval(1.0 / other.upper_bound, 1.0 / other.lower_bound)

        res_0 = Interval(-np.inf, 1.0 / other.lower_bound)
        res_1 = Interval(1.0 / other.upper_bound, np.inf)

        return IntervalUnion([self * res_0, self * res_1])

    def __rtruediv__(self, other):
        """
        The right hand division operator

        Args:
            other (int|float|Interval): an object to divide

        Returns:
            Interval: the ratio
        """
        if not isinstance(other, (Interval, IntervalUnion)):
            other = Interval(other, other)

        return other / self

    def __repr__(self) -> str:
        """
        String representation of the object

        Returns:
            str: the representation
        """
        return str(f"({str(self.lower_bound)}, {str(self.upper_bound)})")

    def __eq__(self, other) -> bool:
        """
        Equality operator

        Args:
            other (obj): the object to compare to

        Returns:
            bool: whether the objects equal
        """
        if not isinstance(other, Interval):
            return False
        return (
            self.lower_bound == other.lower_bound
            and self.upper_bound == other.upper_bound
        )

    def __ne__(self, other) -> bool:
        """
        Non-equality operator

        Args:
            other (obj): the object to compare to

        Returns:
            bool: whether the objects are not equal
        """
        return not self.__eq__(other)

    def __neg__(self):
        """
        The negation operator

        Returns:
            Interval: the negated interval
        """
        return (-1) * self

    def __pow__(self, other):
        """
        The power operation on the interval

        Args:
            other (numeric): the exponent

        Returns:
            Interval: the power of the interval
        """
        tmp = self
        if other < 1:
            tmp = self.intersection(Interval(0, np.inf))
            res = Interval(tmp.lower_bound**other, tmp.upper_bound**other)
        elif int(other) % 2 == 0:
            if self.lower_bound > 0 or self.upper_bound < 0:
                lower_bound = self.lower_bound**other
                upper_bound = self.upper_bound**other
                res = Interval(
                    min(lower_bound, upper_bound), max(lower_bound, upper_bound)
                )
            else:
                res = Interval(
                    0, max(self.lower_bound**other, self.upper_bound**other)
                )
        elif int(other) % 2 == 1:
            lower_bound = self.lower_bound**other
            upper_bound = self.upper_bound**other
            res = Interval(min(lower_bound, upper_bound), max(lower_bound, upper_bound))

        return res

    def representing_int(self):
        """
        Returns a representative integer

        Returns:
            int: a representative element of the interval
        """
        shrunk = self.shrink_to_integers()
        if not shrunk.is_empty():
            return shrunk.lower_bound
        return None


class IntervalUnion:
    """
    The interval union abstraction
    """

    def __init__(self, intervals):
        """
        Constructor of the object

        Args:
            intervals (Interval|tuple|list(Interval)): a specification of one interval
                                                        or a list of intervals
        """

        if isinstance(intervals, Interval):
            self.intervals = [intervals]
        elif isinstance(intervals, tuple):
            self.intervals = [Interval(intervals[0], intervals[1])]
        elif isinstance(intervals, (list)) and len(intervals) > 0:
            if isinstance(intervals[0], Interval):
                self.intervals = intervals
            else:
                self.intervals = [
                    Interval(interval[0], interval[1]) for interval in intervals
                ]
        else:
            self.intervals = intervals

        if len(self.intervals) > 0:
            self.simplify()

    def to_tuple(self) -> list:
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
                if idx != jdx:
                    if (
                        (int0.lower_bound == int1.lower_bound)
                        and (int0.upper_bound == int1.upper_bound)
                        and idx > jdx
                    ):
                        continue
                    if (int0.lower_bound <= int1.lower_bound) and (
                        int0.upper_bound >= int1.upper_bound
                    ):
                        # the interval widh index jdx is contained in the interval with index idx
                        to_drop.append(jdx)

        intervals = [
            interval for idx, interval in enumerate(intervals) if not idx in to_drop
        ]

        # chaining the intervals
        sorted_intervals = sorted(intervals, key=lambda x: x.lower_bound)

        final_intervals = []
        interval = sorted_intervals[0]

        for idx in range(1, len(sorted_intervals)):
            if interval.upper_bound >= sorted_intervals[idx].lower_bound:
                interval = Interval(
                    interval.lower_bound, sorted_intervals[idx].upper_bound
                )
            else:
                final_intervals.append(interval)
                interval = sorted_intervals[idx]
        final_intervals.append(interval)

        self.intervals = final_intervals

    def contains(self, value) -> bool:
        """
        Check if the interval contains the value

        Args:
            value (float|int): the value to check

        Returns:
            bool: True if the interval contains the value, otherwise False
        """
        return any(interval.contains(value) for interval in self.intervals)

    def intersection(self, other):
        """
        Returns the intersection of two intervals

        Args:
            other (Interval|IntervalUnion): the other interval

        Returns:
            IntervalUnion: the intersection, empty interval union if the intersection
            is empty
        """
        if isinstance(other, Interval):
            new_intervals = [
                other.intersection(interval) for interval in self.intervals
            ]
            new_intervals = [
                interval for interval in new_intervals if interval != Interval(1, 0)
            ]
            return IntervalUnion(new_intervals)

        intersections = []
        for int0 in self.intervals:
            for int1 in other.intervals:
                intersections.append(int0.intersection(int1))

        return IntervalUnion(
            [interval for interval in intersections if interval != Interval(1, 0)]
        )

    def integer(self) -> bool:
        """
        Check whether the interval fulfills integer conditions

        Returns:
            bool: whether the interval contains an integer
        """

        return any(interval.integer() for interval in self.intervals)

    def shrink_to_integers(self):
        """
        Shrinking the interval to integer boundaries

        Returns:
            IntervalUnion: the shrinked interval union
        """
        return IntervalUnion(
            [interval.shrink_to_integers() for interval in self.intervals]
        )

    def integer_counts(self) -> int:
        """
        Returns the count of integers in the interval union

        Returns:
            int: the count of integers in the interval union
        """
        return sum(interval.integer_counts() for interval in self.intervals)

    def is_empty(self) -> bool:
        """
        Checking if the interval union is empty

        Returns:
            bool: whether the interval union is empty
        """
        if len(self.intervals) == 0:
            return True
        return all(interval.is_empty() for interval in self.intervals)

    def __add__(self, other):
        """
        The addition operator

        Args:
            other (int|float|Interval|IntervalUnion): an object to add

        Returns:
            IntervalUnion: the sum
        """
        if not isinstance(other, (IntervalUnion,)):
            return IntervalUnion([interval + other for interval in self.intervals])

        intervals = []
        for int0 in self.intervals:
            intervals.extend(int0 + int1 for int1 in other.intervals)

        return IntervalUnion(intervals)

    def __radd__(self, other):
        """
        The right hand addition

        Args:
            other (int|float|Interval|IntervalUnion): an object to add

        Returns:
            IntervalUnion: the sum
        """
        return self + other

    def __sub__(self, other):
        """
        The subtraction operator

        Args:
            other (int|float|Interval|IntervalUnion): an object to subtract

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
            other (int|float|IntervalUnion): an object to subtract from

        Returns:
            IntervalUnion: the difference
        """
        return -1 * self + other

    def __mul__(self, other):
        """
        The multiplication operator

        Args:
            other (int|float|IntervalUnion): an object to multiply

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
            other (int|float|IntervalUnion): an object to multiply

        Returns:
            IntervalUnion: the product
        """
        return self.__mul__(other)

    def __truediv__(self, other):
        """
        The division operator

        Args:
            other (int|float|IntervalUnion): an object to divide with

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
            other (int|float|IntervalUnion): an object to divide

        Returns:
            IntervalUnion: the ratio
        """
        if not isinstance(other, (Interval, IntervalUnion)):
            other = Interval(other, other)

        if not isinstance(other, IntervalUnion):
            other = IntervalUnion([other])

        return other / self

    def __neg__(self):
        """
        The negation operator

        Returns:
            IntervalUnion: the negated interval union
        """
        return (-1) * self

    def __repr__(self) -> str:
        """
        String representation of the object

        Returns:
            str: the representation
        """
        return " | ".join(
            [
                f"({interval.lower_bound}, {interval.upper_bound})"
                for interval in self.intervals
            ]
        )

    def __eq__(self, other) -> bool:
        """
        Equality operator

        Args:
            other (obj): the object to compare to

        Returns:
            bool: whether the objects equal
        """
        if not isinstance(other, IntervalUnion):
            if isinstance(other, Interval):
                if len(self.intervals) == 1:
                    return self.intervals[0] == other
            return False

        if len(self.intervals) != len(other.intervals):
            return False

        for int0, int1 in zip(self.intervals, other.intervals):
            if int0 != int1:
                return False

        return True

    def __ne__(self, other) -> bool:
        """
        Non-equality operator

        Args:
            other (obj): the object to compare to

        Returns:
            bool: whether the objects are not equal
        """
        return not self.__eq__(other)

    def __pow__(self, other):
        """
        The power operation on the interval union

        Args:
            other (numeric): the exponent

        Returns:
            IntervalUnion: the return of the power operation
        """
        return IntervalUnion([interval**other for interval in self.intervals])

    def representing_int(self):
        """
        Returns a representative integer

        Returns:
            int: a representative element of the interval
        """
        if len(self.intervals) == 0:
            return None

        for interval in self.intervals:
            integer = interval.representing_int()
            if integer is not None:
                return integer

        return None

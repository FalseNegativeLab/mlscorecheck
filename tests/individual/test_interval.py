"""
This module tests the interval computing
"""

import numpy as np

from mlscorecheck.individual import Interval, IntervalUnion

def test_interval_constructor():
    """
    Testing the interval constructor
    """

    interval = Interval(np.nan, 1)
    assert interval == Interval(1, 0)

    assert Interval(2, 1) == Interval(1, 0)

def test_interval_equality():
    """
    Testing the interval equality
    """
    interval = Interval(-2, 5)

    assert interval != -2
    assert interval == Interval(-2, 5)
    assert interval != Interval(-2, 6)

def test_interval_repr():
    """
    Testing the representation
    """
    assert str(Interval(0, 1)) == '(0, 1)'

def test_to_tuple():
    """
    Testing the tuple conversion
    """
    assert Interval(0, 1).to_tuple() == (0, 1)

def test_interval_addition():
    """
    Testing the interval addition
    """

    int0 = Interval(-2, 3)
    int1 = Interval(-1, 4)

    add = int0 + int1

    assert add.lower_bound == int0.lower_bound + int1.lower_bound
    assert add.upper_bound == int0.upper_bound + int1.upper_bound

    add = int0 + 2

    assert add.lower_bound == int0.lower_bound + 2

    add = 2 + int0

    assert add.lower_bound == int0.lower_bound + 2

def test_interval_subtraction():
    """
    Testing the interval subtraction
    """
    int0 = Interval(-2, 3)
    int1 = Interval(-1, 4)

    sub = int0 - int1

    assert sub.lower_bound == int0.lower_bound - int1.upper_bound
    assert sub.upper_bound == int0.upper_bound - int1.lower_bound

    sub = int0 - 2

    assert sub.lower_bound == int0.lower_bound - 2

    sub = 2 - int0

    assert sub.lower_bound == 2 - int0.upper_bound

def test_interval_multiplication():
    """
    Testing the interval multiplication
    """
    int0 = Interval(-2, 3)
    int1 = Interval(-1, 4)

    mult = int0 * int1

    terms = [int0.lower_bound * int1.lower_bound,
             int0.lower_bound * int1.upper_bound,
             int0.upper_bound * int1.lower_bound,
             int0.upper_bound * int1.upper_bound]

    assert mult.lower_bound == min(terms)
    assert mult.upper_bound == max(terms)

    mult = int0 * 2

    assert mult.lower_bound == 2 * int0.lower_bound

    mult = 2 * int0

    assert mult.lower_bound == 2 * int0.lower_bound

def test_interval_division_scalar():
    """
    Testing the interval division with scalar
    """
    int0 = Interval(-2, 3)

    div = 1.0 / int0

    assert isinstance(div, IntervalUnion)
    assert div.intervals[0].lower_bound == -np.inf
    assert div.intervals[0].upper_bound == 1.0/int0.lower_bound
    assert div.intervals[1].lower_bound == 1.0/int0.upper_bound
    assert div.intervals[1].upper_bound == np.inf

    div = 1.0 / Interval(1, 2)
    assert div.lower_bound == 0.5
    assert div.upper_bound == 1

    div = 1.0 / Interval(0, 2)
    assert div.lower_bound == 0.5
    assert div.upper_bound == np.inf

    div = 1.0 / Interval(-2, 0)
    assert div.lower_bound == -np.inf
    assert div.upper_bound == -0.5

    div = int0 / 1.0

    assert int0.lower_bound == div.lower_bound and int0.upper_bound == div.upper_bound

    div = int0 / (-1.0)

    assert int0.lower_bound == -div.upper_bound and int0.upper_bound == -div.lower_bound

def test_interval_division_composite():
    """
    Testing the interval division with intervals
    """
    int0 = Interval(-2, 3)

    int1 = Interval(1, 2)
    inverse = 1.0 / int1

    div1 = int0 / int1

    terms = [int0.lower_bound * inverse.lower_bound,
             int0.lower_bound * inverse.upper_bound,
             int0.upper_bound * inverse.lower_bound,
             int0.upper_bound * inverse.upper_bound]

    assert div1.lower_bound == min(terms)
    assert div1.upper_bound == max(terms)

    int1 = Interval(1, 2)
    inverse = 1.0 / int1

    div1 = int0 / int1

    terms = [int0.lower_bound * inverse.lower_bound,
             int0.lower_bound * inverse.upper_bound,
             int0.upper_bound * inverse.lower_bound,
             int0.upper_bound * inverse.upper_bound]

    assert div1.lower_bound == min(terms)
    assert div1.upper_bound == max(terms)

    int1 = Interval(-5, -1)
    inverse = 1.0 / int1

    div1 = int0 / int1

    terms = [int0.lower_bound * inverse.lower_bound,
             int0.lower_bound * inverse.upper_bound,
             int0.upper_bound * inverse.lower_bound,
             int0.upper_bound * inverse.upper_bound]

    assert div1.lower_bound == min(terms)
    assert div1.upper_bound == max(terms)

    int0 = Interval(2, 5)
    int1 = Interval(-5, 7)
    inverse = 1.0 / int1

    div1 = int0 / int1

    terms = [int0.lower_bound * inverse.intervals[0].lower_bound,
             int0.lower_bound * inverse.intervals[0].upper_bound,
             int0.upper_bound * inverse.intervals[0].lower_bound,
             int0.upper_bound * inverse.intervals[0].upper_bound]

    assert div1.intervals[0].lower_bound == min(terms)
    assert div1.intervals[0].upper_bound == max(terms)

    terms = [int0.lower_bound * inverse.intervals[1].lower_bound,
             int0.lower_bound * inverse.intervals[1].upper_bound,
             int0.upper_bound * inverse.intervals[1].lower_bound,
             int0.upper_bound * inverse.intervals[1].upper_bound]

    assert div1.intervals[1].lower_bound == min(terms)
    assert div1.intervals[1].upper_bound == max(terms)

def test_interval_union_constructor():
    """
    Testing the interval union constructor
    """

    assert IntervalUnion(Interval(0, 1)) == IntervalUnion([Interval(0, 1)])
    assert IntervalUnion((0, 1)) == IntervalUnion([Interval(0, 1)])
    assert IntervalUnion([(0, 1)]) == IntervalUnion([Interval(0, 1)])

def test_interval_union_pow():
    """
    Testing the power operation on interval unions
    """

    intu = IntervalUnion([(-1, 2)])
    intu = intu ** 2

    assert intu == IntervalUnion([(0, 4)])

def test_interval_union_to_tuple():
    """
    Testing the tuple conversion of interval union
    """

    intu = IntervalUnion([Interval(0, 1), Interval(3, 4)])

    assert intu.to_tuple() == [(0, 1), (3, 4)]

def test_interval_union_simplify():
    """
    Testing the interval union simplification
    """

    # simple
    intun = IntervalUnion([Interval(1, 2),
                            Interval(3, 5),
                            Interval(4, 6),
                            Interval(5, 8),
                            Interval(10, 11)])

    assert len(intun.intervals) == 3
    assert intun.intervals[1].lower_bound == 3
    assert intun.intervals[1].upper_bound == 8

    # shuffled
    intun = IntervalUnion([Interval(1, 2),
                            Interval(3, 5),
                            Interval(10, 11),
                            Interval(5, 8),
                            Interval(4, 6)])

    assert len(intun.intervals) == 3
    assert intun.intervals[1].lower_bound == 3
    assert intun.intervals[1].upper_bound == 8

    # subsets
    intun = IntervalUnion([Interval(1.5, 1.6),
                            Interval(1, 2),
                            Interval(3, 5),
                            Interval(4, 6),
                            Interval(5, 8),
                            Interval(10, 11),
                            Interval(4, 4)])

    assert len(intun.intervals) == 3
    assert intun.intervals[1].lower_bound == 3
    assert intun.intervals[1].upper_bound == 8

    # first union
    intun = IntervalUnion([Interval(3, 5),
                            Interval(4, 6),
                            Interval(5, 8),
                            Interval(10, 11)])

    assert len(intun.intervals) == 2
    assert intun.intervals[0].lower_bound == 3
    assert intun.intervals[0].upper_bound == 8

    # last union
    intun = IntervalUnion([Interval(1, 2),
                            Interval(3, 5),
                            Interval(4, 6),
                            Interval(5, 8)])

    assert len(intun.intervals) == 2
    assert intun.intervals[1].lower_bound == 3
    assert intun.intervals[1].upper_bound == 8

    # duplicate
    intun = IntervalUnion([Interval(1, 2),
                            Interval(3, 5),
                            Interval(1, 2),
                            Interval(5, 8),
                            Interval(10, 11)])

    assert len(intun.intervals) == 3
    assert intun.intervals[1].lower_bound == 3
    assert intun.intervals[1].upper_bound == 8

    # all one
    intun = IntervalUnion([Interval(4, 4),
                            Interval(3, 5),
                            Interval(4, 6),
                            Interval(5, 8),
                            Interval(4, 7)])

    assert len(intun.intervals) == 1
    assert intun.intervals[0].lower_bound == 3
    assert intun.intervals[0].upper_bound == 8

def test_interval_union_add():
    """
    Testing addition with interval unions
    """
    intun0 = IntervalUnion([Interval(1, 2),
                            Interval(10, 20),
                            Interval(100, 200)])
    intun1 = IntervalUnion([Interval(-1, 1),
                            Interval(2, 8)])

    add = intun0 + intun1

    results = []
    for int0 in intun0.intervals:
        for int1 in intun1.intervals:
            results.append(int0 + int1)
    tmp = IntervalUnion(results)

    assert len(add.intervals) <= len(intun0.intervals) * len(intun1.intervals)
    assert len(add.intervals) == len(tmp.intervals)
    assert add.intervals[0] == tmp.intervals[0]

    add = intun0 + 2

    assert len(add.intervals) == len(intun0.intervals)
    assert add.intervals[0] == intun0.intervals[0] + 2

    add = 2 + intun0

    assert len(add.intervals) == len(intun0.intervals)
    assert add.intervals[0] == intun0.intervals[0] + 2

def test_interval_union_subtract():
    """
    Testing subtraction with interval unions
    """
    intun0 = IntervalUnion([Interval(1, 2),
                            Interval(10, 20),
                            Interval(100, 200)])
    intun1 = IntervalUnion([Interval(-1, 1),
                            Interval(2, 8)])

    sub = intun0 - intun1

    results = []
    for int0 in intun0.intervals:
        for int1 in intun1.intervals:
            results.append(int0 - int1)
    tmp = IntervalUnion(results)

    assert len(sub.intervals) <= len(intun0.intervals) * len(intun1.intervals)
    assert len(sub.intervals) == len(tmp.intervals)
    assert sub.intervals[0] == tmp.intervals[0]

    sub = intun0 - 2

    assert len(sub.intervals) == len(intun0.intervals)
    assert sub.intervals[0] == intun0.intervals[0] - 2

    sub = 2 - intun0

    assert len(sub.intervals) == len(intun0.intervals)
    assert sub.intervals[2] == (2 - intun0.intervals[0])

def test_interval_union_multiply():
    """
    Testing addition with interval unions
    """
    intun0 = IntervalUnion([Interval(1, 2),
                            Interval(10, 20),
                            Interval(100, 200)])
    intun1 = IntervalUnion([Interval(-1, 1),
                            Interval(2, 8)])

    mult = intun0 * intun1

    results = []
    for int0 in intun0.intervals:
        for int1 in intun1.intervals:
            results.append(int0 * int1)
    tmp = IntervalUnion(results)

    assert len(mult.intervals) <= len(intun0.intervals) * len(intun1.intervals)
    assert len(mult.intervals) == len(tmp.intervals)
    assert mult.intervals[0] == tmp.intervals[0]

    mult = intun0 * 2

    assert len(mult.intervals) == len(intun0.intervals)
    assert mult.intervals[0] == intun0.intervals[0] * 2

    mult = 2 * intun0

    assert len(mult.intervals) == len(intun0.intervals)
    assert mult.intervals[0] == intun0.intervals[0] * 2

def test_interval_union_divide():
    """
    Testing addition with interval unions
    """
    intun0 = IntervalUnion([Interval(1, 2),
                            Interval(10, 20),
                            Interval(100, 200)])
    intun1 = IntervalUnion([Interval(-1, 1),
                            Interval(2, 8),
                            Interval(0, 1.5),
                            Interval(-2.2, 0)])

    div = intun0 / intun1

    results = []
    for int0 in intun0.intervals:
        for int1 in intun1.intervals:
            tmp = int0 / int1
            if isinstance(tmp, Interval):
                results.append(tmp)
            else:
                results.extend(tmp.intervals)
    tmp = IntervalUnion(results)

    assert len(div.intervals) <= len(intun0.intervals) * len(intun1.intervals)
    assert len(div.intervals) == len(tmp.intervals)
    assert div.intervals[0] == tmp.intervals[0]

    div = intun0 / 2

    assert len(div.intervals) == len(intun0.intervals)
    assert div.intervals[0] == intun0.intervals[0] / 2

    div = 2 / intun0

    assert len(div.intervals) == len(intun0.intervals)
    assert div.intervals[0] == 2 / intun0.intervals[2]

def test_interval_union_equality():
    """
    Testing the interval union equality
    """
    intun = IntervalUnion([Interval(-2, 5)])

    assert intun != -2
    assert intun == IntervalUnion([Interval(-2, 5)])
    assert intun == IntervalUnion([Interval(-2, 4), Interval(4, 5)])
    assert intun != IntervalUnion([Interval(-2, 6)])
    assert intun != IntervalUnion([Interval(0, 1), Interval(2, 3)])

def test_interval_union_repr():
    """
    Testing the representation
    """
    assert str(IntervalUnion([Interval(0, 1), Interval(2, 3)])) == '(0, 1) | (2, 3)'

def test_cross_interval_intervalunion():
    """
    Testing cross operations between intervals and interval unions
    """
    int0 = Interval(-1, 2)
    intun = IntervalUnion([Interval(3, 5)])

    res0 = int0 + intun
    res1 = intun + int0

    assert res0 == res1

    res0 = int0 - intun
    res1 = (-1)*(intun - int0)

    assert res0 == res1

    res0 = int0 * intun
    res1 = intun * int0

    assert res0 == res1

    res0 = int0 / intun
    res1 = intun / int0

    assert (1.0 / res0) == res1

def test_intersection():
    """
    Testing the interval intersection
    """

    assert Interval(0, 1).intersection(Interval(0.5, 2)) == Interval(0.5, 1)
    assert Interval(0, 1).intersection(Interval(1, 2)) == Interval(1, 1)
    assert Interval(0, 1).intersection(Interval(2, 3)) == Interval(1, 0)
    assert Interval(0, 10).intersection(Interval(3, 4)) == Interval(3, 4)
    assert Interval(3, 4).intersection(Interval(0, 10)) == Interval(3, 4)

def test_integer():
    """
    Testing the integer condition
    """

    assert Interval(0, 1).integer()
    assert Interval(-0.1, 0.9).integer()
    assert Interval(0.9, 1.1).integer()
    assert Interval(1.0, 1.1).integer()
    assert Interval(1.5, 3.5).integer()
    assert Interval(-np.inf, 5).integer()
    assert Interval(-2, np.inf).integer()

    assert not Interval(0.5, 0.6).integer()
    assert not Interval(-0.6, -0.5).integer()

def test_contains():
    """
    Testing the contain function
    """

    assert Interval(0, 1).contains(0)
    assert Interval(0, 1).contains(1)
    assert Interval(0, 1).contains(0.5)
    assert not Interval(0, 1).contains(-np.inf)
    assert not Interval(0, 1).contains(np.inf)
    assert not Interval(0, 1).contains(6)
    assert Interval(-np.inf, 5).contains(-np.inf)
    assert Interval(5, np.inf).contains(10)

def test_union_intersection():
    """
    Testing the intersections of unions
    """

    assert IntervalUnion([Interval(0, 1), Interval(1, 2)])\
            .intersection(IntervalUnion([Interval(1, 2), Interval(2, 3)])) == \
            IntervalUnion([Interval(1, 2)])

    assert IntervalUnion([Interval(0, 1), Interval(2, 3)])\
            .intersection(IntervalUnion([Interval(1, 2), Interval(2, 3)])) == \
            IntervalUnion([Interval(1, 1), Interval(2, 2), Interval(2, 3)])

    assert IntervalUnion([Interval(0, 1), Interval(2, 3)])\
            .intersection(IntervalUnion([Interval(1.5, 1.8), Interval(3.2, 4)])) == \
            IntervalUnion([])

    assert IntervalUnion([Interval(0, 1), Interval(2, 3)])\
            .intersection(IntervalUnion([Interval(0.5, 2.5), Interval(2.8, 3.2)])) == \
            IntervalUnion([Interval(0, 1).intersection(Interval(0.5, 2.5)),
                            Interval(0, 1).intersection(Interval(2.8, 3.2)),
                            Interval(2, 3).intersection(Interval(0.5, 2.5)),
                            Interval(2, 3).intersection(Interval(2.8, 3.2))])

    assert IntervalUnion([Interval(0, 1), Interval(1, 2)])\
            .intersection(Interval(0.5, 1.5)) == Interval(0.5, 1.5)

def test_union_integer():
    """
    Testing the integer condition for interval union
    """

    assert IntervalUnion([Interval(0, 1), Interval(0.5, 0.6)]).integer()
    assert IntervalUnion([Interval(-0.1, 0.9), Interval(0.5, 0.6)]).integer()
    assert IntervalUnion([Interval(0.9, 1.1), Interval(0.5, 0.6)]).integer()
    assert IntervalUnion([Interval(1.0, 1.1), Interval(0.5, 0.6)]).integer()
    assert IntervalUnion([Interval(1.5, 3.5), Interval(0.5, 0.6)]).integer()
    assert IntervalUnion([Interval(-np.inf, 5), Interval(0.5, 0.6)]).integer()
    assert IntervalUnion([Interval(-2, np.inf), Interval(0.5, 0.6)]).integer()

    assert not IntervalUnion([Interval(-0.6, -0.5)]).integer()
    assert not IntervalUnion([Interval(0.5, 0.6)]).integer()
    assert not IntervalUnion([Interval(-0.6, -0.5), Interval(0.5, 0.6)]).integer()

def test_union_contains():
    """
    Testing the contain function for interval union
    """

    assert IntervalUnion([Interval(0, 1), Interval(0.5, 0.6)]).contains(0.55)
    assert IntervalUnion([Interval(-0.1, 0.9), Interval(0.5, 0.6)]).contains(0)
    assert IntervalUnion([Interval(0.9, 1.1), Interval(0.5, 0.6)]).contains(1)
    assert IntervalUnion([Interval(1.0, 1.1), Interval(0.5, 0.6)]).contains(1)
    assert IntervalUnion([Interval(1.5, 3.5), Interval(0.5, 0.6)]).contains(2)
    assert IntervalUnion([Interval(-np.inf, 5), Interval(0.5, 0.6)]).contains(0)
    assert IntervalUnion([Interval(-2, np.inf), Interval(0.5, 0.6)]).contains(0)
    assert IntervalUnion([Interval(-np.inf, 5), Interval(0.5, 0.6)]).contains(-np.inf)
    assert IntervalUnion([Interval(-2, np.inf), Interval(0.5, 0.6)]).contains(np.inf)

    assert not IntervalUnion([Interval(-0.6, -0.5)]).contains(0)
    assert not IntervalUnion([Interval(0.5, 0.6)]).contains(0)
    assert not IntervalUnion([Interval(-0.6, -0.5), Interval(0.5, 0.6)]).contains(0)

def test_interval_shrink_to_integer():
    """
    Testing the shrinking of intervals to integer boundaries
    """

    assert Interval(0.2, 4.5).shrink_to_integers() == Interval(1, 4)
    assert Interval(0.2, 1.2).shrink_to_integers() == Interval(1, 1)
    assert Interval(0.2, 0.3).shrink_to_integers() == Interval(1, 0)

def test_interval_union_shrink_to_integer():
    """
    Testing shrinking the interval union to integer boundaries
    """

    assert IntervalUnion([Interval(0.2, 4.5), Interval(6.3, 6.8)])\
            .shrink_to_integers() == IntervalUnion([Interval(1, 4), Interval(7, 6)])

def test_is_empty():
    """
    Testing the is_empty functionalities
    """
    assert not Interval(1, 2).is_empty()
    assert Interval(1, 0).is_empty()
    assert not IntervalUnion([Interval(1, 2)]).is_empty()
    assert IntervalUnion([Interval(1, 0)]).is_empty()
    assert IntervalUnion([]).is_empty()

def test_negation():
    """
    Testing the negation operators
    """

    assert -Interval(0, 1) == Interval(-1, 0)
    assert -IntervalUnion([Interval(0, 1)]) == IntervalUnion([Interval(-1, 0)])

def test_integer_counts():
    """
    Testing the integer counts
    """

    assert Interval(0, 1).integer_counts() == 2
    assert Interval(1, 0).integer_counts() == 0
    assert IntervalUnion([Interval(0, 1), Interval(1, 0)]).integer_counts() == 2
    assert IntervalUnion([]).integer_counts() == 0

def test_odd_power():
    """
    Testing the odd powers
    """

    assert Interval(0, 2)**3 == Interval(0, 8)
    assert Interval(-1, 2)**3 == Interval(-1, 8)
    assert Interval(-2, -1)**3 == Interval(-8, -1)

def test_representing_integer():
    """
    Testing the extraction of a representing integer
    """

    assert Interval(0, 1).representing_int() == 0
    assert Interval(1, 0).representing_int() is None

    assert IntervalUnion([Interval(0, 1)]).representing_int() == 0
    assert IntervalUnion([Interval(1, 0)]).representing_int() is None
    assert IntervalUnion([]).representing_int() is None

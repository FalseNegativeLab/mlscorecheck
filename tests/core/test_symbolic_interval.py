"""
This module tests symbolic interval computing
"""

import numpy as np

from sympy import Symbol

from mlscorecheck.core import (Interval, IntervalUnion,
                               SymbolicInterval, SymbolicIntervalUnion)

def test_equality():
    """
    Testing the equality operator
    """

    a = Symbol('a')
    b = Symbol('b')
    c = Symbol('c')

    interval = SymbolicInterval(a + b, b - c)

    assert interval == SymbolicInterval(a + b, b - c)
    assert interval != -2
    assert interval != SymbolicInterval(a + b, b)

def test_repr():
    """
    Testing the representation
    """

    interval = SymbolicInterval(Symbol('a'), Symbol('b'))

    assert len(str(interval)) > 0

def test_addition():
    """
    Testing the interval addition
    """

    a = Symbol('a')
    b = Symbol('b')
    c = Symbol('c')
    d = Symbol('d')

    int0 = SymbolicInterval(c - d, a + b)
    int1 = SymbolicInterval(c/d, a*b)

    add = int0 + int1

    assert add.lower_bound == int0.lower_bound + int1.lower_bound
    assert add.upper_bound == int0.upper_bound + int1.upper_bound

    add = int0 + 2

    assert add.lower_bound == int0.lower_bound + 2

    add = 2 + int0

    assert add.lower_bound == 2 + int0.lower_bound

def test_subtraction():
    """
    Testing the interval subtraction
    """

    a = Symbol('a')
    b = Symbol('b')
    c = Symbol('c')
    d = Symbol('d')

    int0 = SymbolicInterval(c - d, a + b)
    int1 = SymbolicInterval(c/d, a*b)

    sub = int0 - int1

    assert sub.lower_bound == int0.lower_bound - int1.upper_bound
    assert sub.upper_bound == int0.upper_bound - int1.lower_bound

    sub = int0 - 2

    assert sub.lower_bound == int0.lower_bound - 2

    sub = 2 - int0

    assert sub.lower_bound == 2 - int0.upper_bound

def test_multiplication():
    """
    Testing interval multiplication
    """

    a = Symbol('a')
    b = Symbol('b')
    c = Symbol('c')
    d = Symbol('d')

    int0 = SymbolicInterval(c - d, a + b)
    int1 = SymbolicInterval(c/d, a*b)

    mult = int0 * int1

    assert len(mult.intervals) == 12

    subst = {'a': 1, 'b': 2, 'c': 3, 'd': 4}

    value = mult.to_interval_union(subst).intervals[0]
    int0_s = int0.to_interval(subst)
    int1_s = int1.to_interval(subst)

    mult_i = int0_s * int1_s

    assert value == mult_i

    mult = int0 * 2
    tmp = 2 * int0.lower_bound

    assert [interval for interval in mult.intervals if interval.condition != False][0].lower_bound == tmp

    mult = 2 * int0
    tmp = 2 * int0.lower_bound

    assert [interval for interval in mult.intervals if interval.condition != False][0].lower_bound == tmp

def test_division_scalar():
    """
    Testing the scalar division
    """

    a = Symbol('a')
    b = Symbol('b')
    c = Symbol('c')
    d = Symbol('d')

    int0 = SymbolicInterval(c - d, a + b)

    div = 1 / int0

    assert isinstance(div, SymbolicIntervalUnion)
    # there is some ambiguity in the order due to the set(list) reduction of terms

    def compare(int0, int1):
        return int0.lower_bound == int1.lower_bound and int0.upper_bound == int1.upper_bound
    tmp = SymbolicInterval(1.0 / (c - d), 1.0 / (a + b))

    assert any([compare(tmp, interval) for interval in div.intervals])

    subst = {'a': 1, 'b': 2, 'c': 3, 'd': 4}

    value = div.to_interval_union(subst)
    int0_s = int0.to_interval(subst)

    assert value == (1 / int0_s)

def test_division_composite():
    """
    Testing the composite division
    """

    a = Symbol('a')
    b = Symbol('b')
    c = Symbol('c')
    d = Symbol('d')

    int0 = SymbolicInterval(c - d, a + b)
    int1 = SymbolicInterval(a*b, c*d)

    div = int0 / int1

    subst = {'a': 1, 'b': 2, 'c': 3, 'd': 4}

    value = div.to_interval_union(subst)

    int0_s = int0.to_interval(subst)
    int1_s = int1.to_interval(subst)

    tmp = int0_s / int1_s

    assert value.intervals[0] == tmp

def test_interval_union_add():
    """
    Testing the addition with interval unions
    """

    a = Symbol('a')
    b = Symbol('b')
    c = Symbol('c')
    d = Symbol('d')

    int0 = SymbolicInterval(c - d, a + b)
    int1 = SymbolicInterval(a*b, c*d)
    int2 = SymbolicInterval(a, c)
    int3 = SymbolicInterval(b, d)
    int4 = SymbolicInterval(c, d)

    intun0 = SymbolicIntervalUnion([int0, int1])
    intun1 = SymbolicIntervalUnion([int2, int3, int4])

    add = intun0 + intun1

    assert len(add.intervals) == len(intun0.intervals) * len(intun1.intervals)
    assert (intun0.intervals[0] + intun1.intervals[0]) in add.intervals

    subst = {'a': 1, 'b': 2, 'c': 3, 'd': 4}

    add_s = add.to_interval_union(subst)

    tmp = []
    for int_a in [int0, int1]:
        for int_b in [int2, int3, int4]:
            tmp.append(int_a.to_interval(subst) + int_b.to_interval(subst))

    assert add_s == IntervalUnion(tmp)

    add = intun0 + 2

    assert (intun0.intervals[0] + 2) in add.intervals

    add = 2 + intun0

    assert (intun0.intervals[0] + 2) in add.intervals

def test_interval_union_subtract():
    """
    Testing the subtraction with interval unions
    """

    a = Symbol('a')
    b = Symbol('b')
    c = Symbol('c')
    d = Symbol('d')

    int0 = SymbolicInterval(c - d, a + b)
    int1 = SymbolicInterval(a*b, c*d)
    int2 = SymbolicInterval(a, c)
    int3 = SymbolicInterval(b, d)
    int4 = SymbolicInterval(c, d)

    intun0 = SymbolicIntervalUnion([int0, int1])
    intun1 = SymbolicIntervalUnion([int2, int3, int4])

    sub = intun0 - intun1

    assert len(sub.intervals) == len(intun0.intervals) * len(intun1.intervals)
    assert (intun0.intervals[0] - intun1.intervals[0]) in sub.intervals

    subst = {'a': 1, 'b': 2, 'c': 3, 'd': 4}

    sub_s = sub.to_interval_union(subst)

    tmp = []
    for int_a in [int0, int1]:
        for int_b in [int2, int3, int4]:
            tmp.append(int_a.to_interval(subst) - int_b.to_interval(subst))

    assert sub_s == IntervalUnion(tmp)

    sub = intun0 - 2

    assert (intun0.intervals[0] - 2) in sub.intervals

    sub = 2 - intun0
    tmp = 2 - intun0.intervals[0]

    assert tmp in sub.intervals

def test_interval_union_multiply():
    """
    Testing the multiplication with interval unions
    """

    a = Symbol('a')
    b = Symbol('b')
    c = Symbol('c')
    d = Symbol('d')

    int0 = SymbolicInterval(c - d, a + b)
    int1 = SymbolicInterval(a*b, c*d)
    int2 = SymbolicInterval(a, c)
    int3 = SymbolicInterval(b, d)
    int4 = SymbolicInterval(c, d)

    intun0 = SymbolicIntervalUnion([int0, int1])
    intun1 = SymbolicIntervalUnion([int2, int3, int4])

    mult = intun0 * intun1

    assert len(mult.intervals) <= len(intun0.intervals) * len(intun1.intervals) * 12
    assert (intun0.intervals[0] * intun1.intervals[0]).intervals[0] in mult.intervals

    subst = {'a': 1, 'b': 2, 'c': 3, 'd': 4}

    mult_s = mult.to_interval_union(subst)

    tmp = []
    for int_a in [int0, int1]:
        for int_b in [int2, int3, int4]:
            tmp.append(int_a.to_interval(subst) * int_b.to_interval(subst))

    assert mult_s == IntervalUnion(tmp)

    mult = intun0 * 2
    tmp = intun0.intervals[0] * 2

    assert tmp.intervals[0] in mult.intervals

    mult = 2 * intun0
    tmp = 2 * intun0.intervals[0]

    assert tmp.intervals[0] in mult.intervals

def test_interval_union_division():
    """
    Testing the division with interval unions
    """

    a = Symbol('a')
    b = Symbol('b')
    c = Symbol('c')
    d = Symbol('d')

    int0 = SymbolicInterval(c - d, a + b)
    int1 = SymbolicInterval(a*b, c*d)
    int2 = SymbolicInterval(a, c)
    int3 = SymbolicInterval(b, d)
    int4 = SymbolicInterval(c, d)

    intun0 = SymbolicIntervalUnion([int0, int1])
    intun1 = SymbolicIntervalUnion([int2, int3, int4])

    div = intun0 / intun1

    assert len(div.intervals) <= len(intun0.intervals) * len(intun1.intervals) * 5 * 12
    assert (intun0.intervals[0] / intun1.intervals[0]).intervals[0] in div.intervals

    subst = {'a': 1, 'b': 2, 'c': 3, 'd': 4}

    div_s = div.to_interval_union(subst)

    tmp = []
    for int_a in [int0, int1]:
        for int_b in [int2, int3, int4]:
            tmp.append(int_a.to_interval(subst) / int_b.to_interval(subst))

    assert div_s == IntervalUnion(tmp)

    div = intun0 / 2
    tmp = intun0.intervals[0] / 2

    assert tmp.intervals[0] in div.intervals

    div = 2 / intun0
    tmp = 2 / intun0.intervals[0]

    assert tmp.intervals[0] in div.intervals

def test_interval_union_equality():
    """
    Testing the equality of interval unions
    """
    a = Symbol('a')
    b = Symbol('b')
    c = Symbol('c')
    d = Symbol('d')

    int0 = SymbolicInterval(c - d, a + b)
    int1 = SymbolicInterval(a*b, c*d)
    int2 = SymbolicInterval(a, c)
    int3 = SymbolicInterval(b, d)
    int4 = SymbolicInterval(c, d)

    intun0 = SymbolicIntervalUnion([int0, int1])
    intun1 = SymbolicIntervalUnion([int2, int3, int4])

    assert intun0 != intun1
    assert intun0 != 2
    assert intun0 == SymbolicIntervalUnion([int0, int1])

def test_cross_interval_intervalunion():
    """
    Testing cross operations between intervals and interval unions
    """

    a = Symbol('a')
    b = Symbol('b')
    c = Symbol('c')
    d = Symbol('d')

    int0 = SymbolicInterval(c - d, a + b)
    int1 = SymbolicInterval(a*b, c*d)
    int2 = SymbolicInterval(a, c)
    int3 = SymbolicInterval(b, d)
    int4 = SymbolicInterval(c, d)

    intun0 = SymbolicIntervalUnion([int0, int1])
    intun1 = SymbolicIntervalUnion([int2, int3, int4])

    res0 = intun0 + int0
    res1 = int0 + intun0

    assert res0 == res1

    res0 = int0 - intun0
    res1 = (-1)*(intun0 - int0)

    assert res0 == res1

    res0 = int0 * intun0
    res1 = intun0 * int0

    assert res0 == res1

    intun0 = SymbolicIntervalUnion([int2])

    res0 = int2 / intun0
    res1 = 1.0 / (intun0 / int2)

    subst = {'a': 1, 'b': 2, 'c': 3, 'd': 4}

    res0_s = res0.to_interval_union(subst)
    res1_s = res1.to_interval_union(subst)

    assert res0_s == res1_s

def test_interval_union_representation():
    """
    Testing the interval union representation
    """
    a = Symbol('a')
    b = Symbol('b')
    c = Symbol('c')
    d = Symbol('d')

    int0 = SymbolicInterval(c - d, a + b)
    int1 = SymbolicInterval(a*b, c*d)
    int2 = SymbolicInterval(a, c)
    int3 = SymbolicInterval(b, d)
    int4 = SymbolicInterval(c, d)

    intun0 = SymbolicIntervalUnion([int0, int1])
    intun1 = SymbolicIntervalUnion([int2, int3, int4])

    assert len(str(intun0)) > 0

    subst = {'a': 1, 'b': 2, 'c': 3, 'd': 4}

    assert int0.check_condition(subst)
    assert intun0.condition_mask(subst)[0]

    int5 = SymbolicInterval(c - d, a + b, condition=a < b)

    assert int5.check_condition(subst)

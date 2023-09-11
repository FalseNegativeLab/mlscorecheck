"""
Testing the complex interval
"""

from mlscorecheck.individual import Interval, IntervalUnion, ComplexInterval

def test_arithmetics():
    """
    Testing the arithmetics
    """

    int0 = Interval(1.1, 1.2)
    int1 = IntervalUnion([Interval(-0.3, -0.2), Interval(0.4, 0.5)])
    int2 = IntervalUnion([Interval(-0.2, -0.1), Interval(0.2, 0.3)])
    int3 = Interval(0.5, 0.6)

    ci0 = ComplexInterval(int0, int1)
    ci1 = ComplexInterval(int2, int3)

    assert (ci0 + ci1).real == int0 + int2
    assert (ci0 + ci1).imag == int1 + int3

    assert (ci0 - ci1).real == int0 - int2
    assert (ci0 - ci1).imag == int1 - int3

    assert (ci0 * ci1).real == int0 * int2 - int1 * int3
    assert (ci0 * ci1).imag == int0 * int3 + int1 * int2

    assert (ci0 / ci1).real == (int0 * int2 + int1 * int3) / (int2**2 + int3**2)
    assert (ci0 / ci1).imag == (int1 * int2 - int0 * int3) / (int2**2 + int3**2)

    assert (5 + ci0).real == 5 + int0
    assert (5 - ci0).real == 5 - int0
    assert (ci0 - 5).real == int0 - 5
    assert (5 * ci0).real == 5 * int0
    assert (5 / ci0).real == (ComplexInterval(Interval(5, 5), Interval(0, 0)) / ci0).real
    assert (ci0 / 5).real == ci0.real / 5

    assert ci0 != int0
    assert ci0 + ci1 == ci0 + ci1
    assert not ci0 + ci1 != ci0 + ci1 # pylint: disable = unneeded-not

    assert -ci0 == (-1) * ci0

    assert isinstance(str(ci0), str)

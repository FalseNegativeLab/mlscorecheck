from ._interval import Interval, IntervalUnion

def is_less_than_zero(value):
    if not isinstance(value, (Interval, IntervalUnion)):
        return value < 0
    if isinstance(value, Interval):
        return value.upper_bound < 0
    return all(interval.upper_bound < 0 for interval in value.intervals)

def is_zero(value, tolerance=1e-8):
    if not isinstance(value, (Interval, IntervalUnion)):
        return abs(value) < tolerance
    return value.contains(0.0)

def unify_results(value_list):
    if not isinstance(value_list[0], (Interval, IntervalUnion)):
        return value_list

    intervals = []
    for interval in value_list:
        if isinstance(interval, Interval):
            intervals.append(interval)
        else:
            intervals.extend(interval.intervals)

    return IntervalUnion(intervals)

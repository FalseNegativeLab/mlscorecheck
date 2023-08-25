import numpy as np

def compare_scores(scores0, scores1, subset=None, rounding_decimals=None, tolerance=1e-5):
    if subset is not None:
        scores0 = {key: scores0[key] for key in subset}
        scores1 = {key: scores1[key] for key in subset}

    if rounding_decimals is None:
        rounding_decimals = 5

    return all(abs(scores0[key] - scores1[key]) <= 10**(-rounding_decimals) + tolerance for key in scores0)

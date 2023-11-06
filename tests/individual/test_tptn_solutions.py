"""
This module tests the tptn solutions of the individual scores
"""

import pytest

import numpy as np

from mlscorecheck.core import safe_call
from mlscorecheck.individual import (
    tptn_solutions,
    is_applicable_tptn,
    generate_1_problem,
)
from mlscorecheck.scores import calculate_scores


@pytest.mark.parametrize("figure", ["tp", "tn"])
@pytest.mark.parametrize("score", list(tptn_solutions.keys()))
@pytest.mark.parametrize(
    "zeros",
    [
        [],
        ["tp"],
        ["tn"],
        ["fp"],
        ["fn"],
        ["tp", "fp"],
        ["tn", "fn"],
        ["tn", "tp"],
        ["fn", "fp"],
    ],
)
@pytest.mark.parametrize("random_state", [3, 5, 7, 11])
def test_tptn_solutions(figure: str, score: str, zeros: list, random_state: int):
    """
    Testing the tp-tn solutions

    Args:
        figure (str): the figure to test the solution for (``tp``/``tn``)
        score (str): the name of the score to test the solution for
        zeros (list(str)): the list of figures to set to zero
        random_state (int): the seed of the random state
    """
    evaluation, _ = generate_1_problem(
        zeros=zeros, add_complements=True, random_state=random_state
    )
    evaluation["beta_positive"] = 2
    evaluation["beta_negative"] = 2

    scores = calculate_scores(evaluation)

    if not (
        scores[score] is not None and is_applicable_tptn(score, scores[score], figure)
    ):
        return

    sol = tptn_solutions[score][figure]
    if sol is None:
        return

    val = safe_call(sol, scores | evaluation)

    if val is None or (
        isinstance(val, list) and (len(val) == 0 or all(value is None for value in val))
    ):
        return

    val = np.array(val)
    assert np.any(np.abs(evaluation[figure] - val) < 1e-8)


special_values = {
    "ppv": [1, 0],
    "npv": [1, 0],
    "fbp": [5, 0],
    "f1p": [2, 0],
    "fbn": [0, 5],
    "f1n": [2, 0],
}


@pytest.mark.parametrize("figure", ["tp", "tn"])
@pytest.mark.parametrize("score", list(tptn_solutions.keys()))
@pytest.mark.parametrize("random_state", [3, 5, 7])
def test_tptn_solutions_failure(figure: str, score: str, random_state: int):
    """
    Testing the tp-tn solutions with failure

    Args:
        figure (str): the figure to test the solution for (``tp``/``tn``)
        score (str): the name of the score to test the solution for
        random_state (int): the seed of the random state
    """
    evaluation, _ = generate_1_problem(
        zeros=[], add_complements=True, random_state=random_state
    )
    evaluation["beta_positive"] = 2
    evaluation["beta_negative"] = 2

    scores = calculate_scores(evaluation)

    sol = tptn_solutions[score][figure]
    if sol is None:
        return

    if score in special_values:
        for idx in range(len(special_values[score])):
            scores[score] = special_values[score][idx]

            val = safe_call(sol, scores | evaluation)

            if val is None or (
                isinstance(val, list)
                and (len(val) == 0 or all(value is None for value in val))
            ):
                assert True
                return

            val = np.array(val)
            assert np.all(np.abs(evaluation[figure] - val) > 1e-8)

    scores[score] = 0
    evaluation["n"] = 0
    evaluation["p"] = 0
    evaluation["tn"] = 0
    evaluation["tp"] = 0

    val = safe_call(sol, scores | evaluation)

    if val is None or (
        isinstance(val, list) and (len(val) == 0 or all(value is None for value in val))
    ):
        assert True
        return

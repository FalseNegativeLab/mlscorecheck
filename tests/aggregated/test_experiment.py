"""
Testing the experiment abstraction

The test cases with complicated bound structures are executed
with timeout to prevent hanging.

It is expected, depending on the solver, that some tests times out.

When failure with bounds is tested, the rounding precision is
not tested with 2 decimals, since accidentally with whatever
bounds it is likely to become feasible.
"""

import numpy as np
import pulp as pl
import pytest

from mlscorecheck.aggregated import (
    Experiment,
    compare_scores,
    generate_experiment,
    get_dataset_score_bounds,
    solve,
)

from ._evaluate_lp import evaluate_timeout

PREFERRED_SOLVER = "PULP_CBC_CMD"
solvers = pl.listSolvers(onlyAvailable=True)
PREFERRED_SOLVER = PREFERRED_SOLVER if PREFERRED_SOLVER in solvers else solvers[0]
solver = pl.getSolver(PREFERRED_SOLVER)
solver_timeout = pl.getSolver(PREFERRED_SOLVER, timeLimit=5)

two_combs = [
    ["acc", "sens"],
    ["acc", "spec"],
    ["acc", "bacc"],
    ["sens", "spec"],
    ["sens", "bacc"],
    ["spec", "bacc"],
]
three_combs = [
    ["acc", "sens", "spec"],
    ["acc", "sens", "bacc"],
    ["acc", "spec", "bacc"],
    ["sens", "spec", "bacc"],
]
four_combs = [["acc", "sens", "spec", "bacc"]]

random_seeds = list(range(5))


@pytest.mark.parametrize("random_seed", random_seeds)
def test_experiment_instantiation(random_seed: int):
    """
    Testing the creation of Experiment objects

    Args:
        random_seed (int): the random seed to use
    """

    experiment = generate_experiment(random_state=random_seed)
    experiment = Experiment(**experiment)

    assert experiment is not None

    experiment2 = Experiment(**experiment.to_dict())

    assert (
        experiment.figures["p"] == experiment2.figures["p"]
        and experiment.figures["n"] == experiment2.figures["n"]
    )


@pytest.mark.parametrize("random_seed", random_seeds)
def test_sampling_and_scores(random_seed: int):
    """
    Testing the score calculation in experiments

    Args:
        random_seed (int): the random seed to use
    """

    experiment = generate_experiment(random_state=random_seed)
    experiment = Experiment(**experiment)

    experiment.sample_figures()

    scores = experiment.calculate_scores()

    if experiment.aggregation == "som":
        value = float(experiment.figures["tp"] / experiment.figures["p"])
        assert abs(scores["sens"] - value) < 1e-10
    elif experiment.aggregation == "mos":
        assert (
            abs(
                np.mean(
                    [evaluation.scores["acc"] for evaluation in experiment.evaluations]
                )
                - scores["acc"]
            )
            < 1e-10
        )


@pytest.mark.parametrize("random_seed", random_seeds)
@pytest.mark.parametrize("aggregation", ["mos"])
@pytest.mark.parametrize("aggregation_folds", ["mos", "som"])
def test_get_dataset_score_bounds(
    random_seed: int, aggregation: str, aggregation_folds: str
):
    """
    Testing the score bounds determination

    Args:
        random_seed (int): the random seed to use
        aggregation (str): the aggregation on datasets ('mos'/'som')
        aggregation_folds (str): the aggregation on folds ('mos'/'som')
    """

    experiment = generate_experiment(
        random_state=random_seed,
        aggregation=aggregation,
        evaluation_params={"aggregation": aggregation_folds},
    )
    experiment = Experiment(**experiment)
    experiment.sample_figures().calculate_scores()

    score_bounds = get_dataset_score_bounds(experiment, feasible=True)

    for evaluation in experiment.evaluations:
        for key in score_bounds:
            assert (
                score_bounds[key][0] <= evaluation.scores[key] <= score_bounds[key][1]
            )

    score_bounds = get_dataset_score_bounds(experiment, feasible=False)

    for evaluation in experiment.evaluations:
        for key in score_bounds:
            if score_bounds[key][0] < 1.0:
                assert (
                    not score_bounds[key][0]
                    <= evaluation.scores[key]
                    <= score_bounds[key][1]
                )


@pytest.mark.parametrize("subset", two_combs + three_combs + four_combs)
@pytest.mark.parametrize("random_seed", random_seeds)
@pytest.mark.parametrize("aggregation", ["mos", "som"])
@pytest.mark.parametrize("rounding_decimals", [2, 3, 4])
def test_linear_programming_success(
    subset: list, random_seed: int, aggregation: str, rounding_decimals: int
):
    """
    Testing the linear programming functionalities in a successful scenario

    Args:
        subset (list): the score subset
        random_seed (int): the random seed to use
        aggregation (str): the aggregation to use ('mos'/'som')
        rounding_decimals (int): the number of decimals to round to
    """

    experiment = generate_experiment(random_state=random_seed, aggregation=aggregation)
    experiment = Experiment(**experiment)

    experiment.sample_figures(random_state=random_seed)

    scores = experiment.calculate_scores(rounding_decimals, subset)

    skeleton = Experiment(**experiment.to_dict())

    lp_program = solve(skeleton, scores, eps=10 ** (-rounding_decimals))

    assert lp_program.status == 1

    skeleton.populate(lp_program)

    assert compare_scores(
        scores,
        skeleton.calculate_scores(),
        eps=10 ** (-rounding_decimals),
        tolerance=1e-6,
    )

    assert skeleton.check_bounds()["bounds_flag"]


@pytest.mark.parametrize("subset", two_combs + three_combs + four_combs)
@pytest.mark.parametrize("random_seed", random_seeds)
@pytest.mark.parametrize("aggregation", ["mos"])
@pytest.mark.parametrize("rounding_decimals", [2, 3, 4])
def test_linear_programming_success_with_bounds(
    subset: list, random_seed: int, aggregation: str, rounding_decimals: int
):
    """
    Testing the linear programming functionalities with bounds

    Args:
        subset (list): the score subset
        random_seed (int): the random seed to use
        aggregation (str): the aggregation to use ('mos'/'som')
        rounding_decimals (int): the number of decimals to round to
    """

    experiment, scores = generate_experiment(
        random_state=random_seed,
        aggregation=aggregation,
        return_scores=True,
        feasible_dataset_score_bounds=True,
        score_subset=subset,
    )

    experiment = Experiment(**experiment)

    lp_program = solve(
        experiment, scores, eps=10 ** (-rounding_decimals), solver=solver_timeout
    )

    assert lp_program.status in (0, 1)

    evaluate_timeout(lp_program, experiment, scores, 10 ** (-rounding_decimals), subset)


@pytest.mark.parametrize("subset", two_combs + three_combs + four_combs)
@pytest.mark.parametrize("random_seed", random_seeds)
@pytest.mark.parametrize("aggregation", ["mos"])
@pytest.mark.parametrize("rounding_decimals", [3, 4])
def test_linear_programming_failure_with_bounds(
    subset: list, random_seed: int, aggregation: str, rounding_decimals: int
):
    """
    Testing the linear programming functionalities with bounds

    Args:
        subset (list): the score subset
        random_seed (int): the random seed to use
        aggregation (str): the aggregation to use ('mos'/'som')
        rounding_decimals (int): the number of decimals to round to
    """

    experiment, scores = generate_experiment(
        random_state=random_seed,
        aggregation=aggregation,
        return_scores=True,
        feasible_dataset_score_bounds=False,
        score_subset=subset,
    )

    experiment = Experiment(**experiment)

    lp_program = solve(
        experiment, scores, eps=10 ** (-rounding_decimals), solver=solver_timeout
    )

    assert lp_program.status in (0, -1)

    evaluate_timeout(lp_program, experiment, scores, 10 ** (-rounding_decimals), subset)


@pytest.mark.parametrize("subset", two_combs + three_combs + four_combs)
@pytest.mark.parametrize("random_seed", random_seeds)
@pytest.mark.parametrize("aggregation", ["mos"])
@pytest.mark.parametrize("rounding_decimals", [2, 3])
def test_linear_programming_success_both_bounds(
    subset: list, random_seed: int, aggregation: str, rounding_decimals: int
):
    """
    Testing the linear programming functionalities with both bounds

    Args:
        subset (list): the score subset
        random_seed (int): the random seed to use
        aggregation (str): the aggregation to use ('mos'/'som')
        rounding_decimals (int): the number of decimals to round to
    """

    experiment, scores = generate_experiment(
        random_state=random_seed,
        aggregation=aggregation,
        evaluation_params={"aggregation": "mos", "feasible_fold_score_bounds": True},
        return_scores=True,
        rounding_decimals=rounding_decimals,
        feasible_dataset_score_bounds=True,
        score_subset=subset,
    )

    experiment = Experiment(**experiment)

    lp_program = solve(
        experiment, scores, eps=10 ** (-rounding_decimals), solver=solver_timeout
    )

    assert lp_program.status in (0, 1)

    evaluate_timeout(lp_program, experiment, scores, 10 ** (-rounding_decimals), subset)


@pytest.mark.parametrize("subset", two_combs + three_combs + four_combs)
@pytest.mark.parametrize("random_seed", random_seeds)
@pytest.mark.parametrize("aggregation", ["mos"])
@pytest.mark.parametrize("rounding_decimals", [3, 4])
def test_linear_programming_failure_both_bounds(
    subset: list, random_seed: int, aggregation: str, rounding_decimals: int
):
    """
    Testing the linear programming functionalities with both bounds

    Args:
        subset (list): the score subset
        random_seed (int): the random seed to use
        aggregation (str): the aggregation to use ('mos'/'som')
        rounding_decimals (int): the number of decimals to round to
    """

    experiment, scores = generate_experiment(
        random_state=random_seed,
        aggregation=aggregation,
        evaluation_params={"aggregation": "mos", "feasible_fold_score_bounds": True},
        return_scores=True,
        rounding_decimals=rounding_decimals,
        feasible_dataset_score_bounds=False,
        score_subset=subset,
    )

    experiment = Experiment(**experiment)

    lp_program = solve(experiment, scores, eps=10 ** (-rounding_decimals))

    assert lp_program.status in (-1, 0)

    evaluate_timeout(lp_program, experiment, scores, 10 ** (-rounding_decimals), subset)


def test_others():
    """
    Testing other functionalities
    """

    experiment = generate_experiment(
        aggregation="som",
        feasible_dataset_score_bounds=True,
        random_state=5
    )
    with pytest.raises(ValueError):
        Experiment(**experiment)

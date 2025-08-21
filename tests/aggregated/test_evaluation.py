"""
This module tests the Evaluation abstraction
"""

import warnings

import pulp as pl
import pytest

from mlscorecheck.aggregated import (
    Evaluation,
    Experiment,
    compare_scores,
    generate_dataset,
    generate_evaluation,
    generate_folding,
    get_fold_score_bounds,
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


def test_evaluate_timeout() -> None:
    """
    Testing the evaluate_timeout function
    """

    class Mock:  # pylint: disable=too-few-public-methods
        """
        Mock lp_problem class
        """

        def __init__(self) -> None:
            """
            Constructor of the mock class
            """
            self.status = 0

    mock = Mock()

    # Create dummy objects for testing - need Experiment, not Evaluation
    dummy_evaluation_dict = generate_evaluation(random_state=42)
    dummy_experiment = Experiment(evaluations=[dummy_evaluation_dict], aggregation="som")
    dummy_scores: dict = {"acc": 0.5}
    dummy_subset: list[str] = ["acc"]

    with warnings.catch_warnings(record=True) as warn:
        evaluate_timeout(mock, dummy_experiment, dummy_scores, 0.1, dummy_subset)
        assert len(warn) == 1


@pytest.mark.parametrize("random_seed", random_seeds)
@pytest.mark.parametrize("aggregation", ["mos", "som"])
def test_instantiation(random_seed: int, aggregation: str) -> None:
    """
    Testing the instantiation of evaluations

    Args:
        random_seed (int): the random seed to use
        aggregation (str): the mode of aggregation
    """

    dataset = generate_dataset(random_state=random_seed)
    folding = generate_folding(dataset=dataset, random_state=random_seed)

    evaluation = Evaluation(dataset, folding, aggregation=aggregation)

    assert evaluation is not None

    evaluation2 = Evaluation(**evaluation.to_dict())

    assert (
        evaluation.figures["p"] == evaluation2.figures["p"]
        and evaluation.figures["n"] == evaluation2.figures["n"]
    )


@pytest.mark.parametrize("random_seed", random_seeds)
@pytest.mark.parametrize("aggregation", ["mos", "som"])
def test_sample_figures(random_seed: int, aggregation: str) -> None:
    """
    Testing the sampling of figures

    Args:
        random_seed (int): the random seed to use
        aggregation (str): the aggregation to use ('mos'/'som')
    """

    dataset = generate_dataset(random_state=random_seed)
    folding = generate_folding(dataset=dataset, random_state=random_seed)

    evaluation = Evaluation(dataset, folding, aggregation=aggregation)

    evaluation.sample_figures(random_state=random_seed).calculate_scores()

    assert evaluation.figures["tp"] >= 0 and evaluation.figures["tn"] >= 0


@pytest.mark.parametrize("subset", two_combs + three_combs + four_combs)
@pytest.mark.parametrize("random_seed", random_seeds)
@pytest.mark.parametrize("aggregation", ["mos", "som"])
@pytest.mark.parametrize("rounding_decimals", [2, 3, 4])
def test_linear_programming_success(
    subset: list[str], random_seed: int, aggregation: str, rounding_decimals: int
) -> None:
    """
    Testing the linear programming functionalities

    Args:
        subset (list): the score subset
        random_seed (int): the random seed to use
        aggregation (str): the aggregation to use ('mos'/'som')
        rounding_decimals (int): the number of decimals to round to
    """

    dataset = generate_dataset(random_state=random_seed)
    folding = generate_folding(dataset=dataset, random_state=random_seed)

    evaluation = Evaluation(dataset, folding, aggregation=aggregation)

    evaluation.sample_figures(random_state=random_seed)

    scores = evaluation.calculate_scores(rounding_decimals, subset)

    skeleton = Evaluation(dataset, folding, aggregation=aggregation)

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
@pytest.mark.parametrize("aggregation", ["mos", "som"])
@pytest.mark.parametrize("rounding_decimals", [2, 3, 4])
def test_linear_programming_evaluation_generation_success(
    subset: list[str], random_seed: int, aggregation: str, rounding_decimals: int
) -> None:
    """
    Testing the linear programming functionalities by generating the evaluation

    Args:
        subset (list): the score subset
        random_seed (int): the random seed to use
        aggregation (str): the aggregation to use ('mos'/'som')
        rounding_decimals (int): the number of decimals to round to
    """

    evaluation_dict = generate_evaluation(random_state=random_seed, aggregation=aggregation)
    assert isinstance(evaluation_dict, dict), "generate_evaluation should return dict when return_scores=False"

    evaluation = Evaluation(
        dataset=evaluation_dict["dataset"],
        folding=evaluation_dict["folding"],
        aggregation=evaluation_dict["aggregation"],
        fold_score_bounds=evaluation_dict.get("fold_score_bounds"),
    )

    evaluation.sample_figures(random_state=random_seed)

    scores = evaluation.calculate_scores(rounding_decimals, subset)

    skeleton = Evaluation(**evaluation.to_dict())

    lp_program = solve(skeleton, scores, eps=10 ** (-rounding_decimals))

    assert lp_program.status == 1

    skeleton.populate(lp_program)

    assert compare_scores(
        scores,
        skeleton.calculate_scores(),
        eps=10 ** (-rounding_decimals),
        tolerance=1e-6,
    )


@pytest.mark.parametrize("random_seed", random_seeds)
@pytest.mark.parametrize("aggregation", ["mos", "som"])
def test_linear_programming_evaluation_generation_failure(
    random_seed: int, aggregation: str
) -> None:
    """
    Testing the linear programming functionalities by generating the evaluation

    Args:
        random_seed (int): the random seed to use
        aggregation (str): the aggregation to use ('mos'/'som')
    """

    evaluation_dict = generate_evaluation(random_state=random_seed, aggregation=aggregation)
    assert isinstance(evaluation_dict, dict), "generate_evaluation should return dict when return_scores=False"

    evaluation = Evaluation(
        dataset=evaluation_dict["dataset"],
        folding=evaluation_dict["folding"],
        aggregation=evaluation_dict["aggregation"],
        fold_score_bounds=evaluation_dict.get("fold_score_bounds"),
    )

    evaluation.sample_figures(random_state=random_seed)

    scores = {"acc": 0.5, "sens": 0.6, "spec": 0.6}

    skeleton = Evaluation(**evaluation.to_dict())

    lp_program = solve(skeleton, scores, eps=1e-6)

    assert lp_program.status == -1


@pytest.mark.parametrize("random_seed", random_seeds)
@pytest.mark.parametrize("aggregation", ["mos", "som"])
def test_get_fold_score_bounds(random_seed: int, aggregation: str) -> None:
    """
    Testing the extraction of fold score bounds

    Args:
        random_seed (int): the random seed to use
        aggregation (str): the aggregation to use ('mos'/'som')
    """

    evaluation_dict = generate_evaluation(random_state=random_seed, aggregation=aggregation)
    assert isinstance(evaluation_dict, dict), "generate_evaluation should return dict when return_scores=False"

    evaluation = Evaluation(
        dataset=evaluation_dict["dataset"],
        folding=evaluation_dict["folding"],
        aggregation=evaluation_dict["aggregation"],
        fold_score_bounds=evaluation_dict.get("fold_score_bounds"),
    )
    evaluation.sample_figures().calculate_scores()

    score_bounds = get_fold_score_bounds(evaluation, feasible=True)

    for fold in evaluation.folds:
        for key in score_bounds:
            assert score_bounds[key][0] <= fold.scores[key] <= score_bounds[key][1]


@pytest.mark.parametrize("subset", two_combs + three_combs + four_combs)
@pytest.mark.parametrize("random_seed", random_seeds)
@pytest.mark.parametrize("aggregation", ["mos"])
@pytest.mark.parametrize("rounding_decimals", [3, 4])
def test_linear_programming_success_bounds(
    subset: list[str], random_seed: int, aggregation: str, rounding_decimals: int
) -> None:
    """
    Testing the linear programming functionalities by generating the evaluation
    with bounds

    Args:
        subset (list): the score subset
        random_seed (int): the random seed to use
        aggregation (str): the aggregation to use ('mos'/'som')
        rounding_decimals (int): the number of decimals to round to
    """

    evaluation, scores = generate_evaluation(
        random_state=random_seed,
        aggregation=aggregation,
        feasible_fold_score_bounds=True,
        rounding_decimals=rounding_decimals,
        return_scores=True,
        score_subset=subset,
    )

    evaluation = Evaluation(**evaluation)

    skeleton = Evaluation(**evaluation.to_dict())

    lp_program = solve(
        skeleton, scores, eps=10 ** (-rounding_decimals), solver=solver_timeout
    )

    assert lp_program.status in (0, 1)

    # Direct evaluation instead of evaluate_timeout since we have an Evaluation, not Experiment
    if lp_program.status == 1:
        populated = skeleton.populate(lp_program)
        assert compare_scores(
            scores, populated.calculate_scores(), 10 ** (-rounding_decimals), subset
        )
        assert populated.check_bounds()["bounds_flag"] is True


@pytest.mark.parametrize("subset", two_combs + three_combs + four_combs)
@pytest.mark.parametrize("random_seed", random_seeds)
@pytest.mark.parametrize("aggregation", ["mos"])
@pytest.mark.parametrize("rounding_decimals", [3, 4])
def test_linear_programming_failure_bounds(
    subset: list[str], random_seed: int, aggregation: str, rounding_decimals: int
) -> None:
    """
    Testing the linear programming functionalities by generating the evaluation
    with bounds

    Args:
        subset (list): the score subset
        random_seed (int): the random seed to use
        aggregation (str): the aggregation to use ('mos'/'som')
        rounding_decimals (int): the number of decimals to round to
    """

    evaluation, scores = generate_evaluation(
        random_state=random_seed,
        aggregation=aggregation,
        feasible_fold_score_bounds=False,
        rounding_decimals=rounding_decimals,
        return_scores=True,
        score_subset=subset,
    )

    evaluation = Evaluation(**evaluation)

    skeleton = Evaluation(**evaluation.to_dict())

    lp_program = solve(
        skeleton, scores, eps=10 ** (-rounding_decimals), solver=solver_timeout
    )

    assert lp_program.status in (-1, 0)

    # Direct evaluation instead of evaluate_timeout since we have an Evaluation, not Experiment  
    # For infeasible problems, just check the status


def test_others() -> None:
    """
    Testing other functionalities
    """

    evaluation_dict = generate_evaluation(aggregation="som",
                                        feasible_fold_score_bounds=True,
                                        random_state=5)
    assert isinstance(evaluation_dict, dict), "generate_evaluation should return dict when return_scores=False"
    with pytest.raises(ValueError):
        Evaluation(
            dataset=evaluation_dict["dataset"],
            folding=evaluation_dict["folding"],
            aggregation=evaluation_dict["aggregation"],
            fold_score_bounds=evaluation_dict.get("fold_score_bounds"),
        )

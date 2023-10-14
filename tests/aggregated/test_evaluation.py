"""
This module tests the Evaluation abstraction
"""

import warnings

import pytest

import pulp as pl

from mlscorecheck.aggregated import (Evaluation,
                                        generate_dataset,
                                        generate_folding,
                                        generate_evaluation,
                                        solve,
                                        compare_scores,
                                        get_fold_score_bounds)

from ._evaluate_lp import evaluate_timeout

PREFERRED_SOLVER = 'PULP_CBC_CMD'
solvers = pl.listSolvers(onlyAvailable=True)
PREFERRED_SOLVER = PREFERRED_SOLVER if PREFERRED_SOLVER in solvers else solvers[0]
solver = pl.getSolver(PREFERRED_SOLVER)
solver_timeout = pl.getSolver(PREFERRED_SOLVER, timeLimit=5)

two_combs = [['acc', 'sens'], ['acc', 'spec'], ['acc', 'bacc'],
            ['sens', 'spec'], ['sens', 'bacc'], ['spec', 'bacc']]
three_combs = [['acc', 'sens', 'spec'], ['acc', 'sens', 'bacc'],
                ['acc', 'spec', 'bacc'], ['sens', 'spec', 'bacc']]
four_combs = [['acc', 'sens', 'spec', 'bacc']]

random_seeds = list(range(5))

def test_evaluate_timeout():
    """
    Testing the evaluate_timeout function
    """

    class Mock: # pylint: disable=too-few-public-methods
        """
        Mock lp_problem class
        """
        def __init__(self):
            """
            Constructor of the mock class
            """
            self.status = 0

    mock = Mock()

    with warnings.catch_warnings(record=True) as warn:
        evaluate_timeout(mock, None, None, None, None)
        assert len(warn) == 1

@pytest.mark.parametrize('random_seed', random_seeds)
@pytest.mark.parametrize('aggregation', ['mor', 'rom'])
def test_instantiation(random_seed: int, aggregation: str):
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

    assert evaluation.figures['p'] == evaluation2.figures['p']\
            and evaluation.figures['n'] == evaluation2.figures['n']

@pytest.mark.parametrize('random_seed', random_seeds)
@pytest.mark.parametrize('aggregation', ['mor', 'rom'])
def test_sample_figures(random_seed: int, aggregation: str):
    """
    Testing the sampling of figures

    Args:
        random_seed (int): the random seed to use
        aggregation (str): the aggregation to use ('mor'/'rom')
    """

    dataset = generate_dataset(random_state=random_seed)
    folding = generate_folding(dataset=dataset, random_state=random_seed)

    evaluation = Evaluation(dataset, folding, aggregation=aggregation)

    evaluation.sample_figures(random_state=random_seed).calculate_scores()

    assert evaluation.figures['tp'] >= 0 and evaluation.figures['tn'] >= 0

@pytest.mark.parametrize('subset', two_combs + three_combs + four_combs)
@pytest.mark.parametrize('random_seed', random_seeds)
@pytest.mark.parametrize('aggregation', ['mor', 'rom'])
@pytest.mark.parametrize('rounding_decimals', [2, 3, 4])
def test_linear_programming_success(subset: list,
                                    random_seed: int,
                                    aggregation: str,
                                    rounding_decimals: int):
    """
    Testing the linear programming functionalities

    Args:
        subset (list): the score subset
        random_seed (int): the random seed to use
        aggregation (str): the aggregation to use ('mor'/'rom')
        rounding_decimals (int): the number of decimals to round to
    """

    dataset = generate_dataset(random_state=random_seed)
    folding = generate_folding(dataset=dataset, random_state=random_seed)

    evaluation = Evaluation(dataset, folding, aggregation=aggregation)

    evaluation.sample_figures(random_state=random_seed)

    scores = evaluation.calculate_scores(rounding_decimals, subset)

    skeleton = Evaluation(dataset, folding, aggregation=aggregation)

    lp_program = solve(skeleton, scores, eps=10**(-rounding_decimals))

    assert lp_program.status == 1

    skeleton.populate(lp_program)

    assert compare_scores(scores,
                            skeleton.calculate_scores(),
                            eps=10**(-rounding_decimals),
                            tolerance=1e-6)

    assert skeleton.check_bounds()['bounds_flag']

@pytest.mark.parametrize('subset', two_combs + three_combs + four_combs)
@pytest.mark.parametrize('random_seed', random_seeds)
@pytest.mark.parametrize('aggregation', ['mor', 'rom'])
@pytest.mark.parametrize('rounding_decimals', [2, 3, 4])
def test_linear_programming_evaluation_generation_success(subset: list,
                                                            random_seed: int,
                                                            aggregation: str,
                                                            rounding_decimals: int):
    """
    Testing the linear programming functionalities by generating the evaluation

    Args:
        subset (list): the score subset
        random_seed (int): the random seed to use
        aggregation (str): the aggregation to use ('mor'/'rom')
        rounding_decimals (int): the number of decimals to round to
    """

    evaluation = generate_evaluation(random_state=random_seed,
                                        aggregation=aggregation)

    evaluation = Evaluation(**evaluation)

    evaluation.sample_figures(random_state=random_seed)

    scores = evaluation.calculate_scores(rounding_decimals, subset)

    skeleton = Evaluation(**evaluation.to_dict())

    lp_program = solve(skeleton, scores, eps=10**(-rounding_decimals))

    assert lp_program.status == 1

    skeleton.populate(lp_program)

    assert compare_scores(scores,
                            skeleton.calculate_scores(),
                            eps=10**(-rounding_decimals),
                            tolerance=1e-6)

@pytest.mark.parametrize('random_seed', random_seeds)
@pytest.mark.parametrize('aggregation', ['mor', 'rom'])
def test_linear_programming_evaluation_generation_failure(random_seed: int,
                                                            aggregation: str):
    """
    Testing the linear programming functionalities by generating the evaluation

    Args:
        random_seed (int): the random seed to use
        aggregation (str): the aggregation to use ('mor'/'rom')
    """

    evaluation = generate_evaluation(random_state=random_seed,
                                        aggregation=aggregation)

    evaluation = Evaluation(**evaluation)

    evaluation.sample_figures(random_state=random_seed)

    scores = {'acc': 0.5, 'sens': 0.6, 'spec': 0.6}

    skeleton = Evaluation(**evaluation.to_dict())

    lp_program = solve(skeleton, scores, eps=1e-6)

    assert lp_program.status == -1

@pytest.mark.parametrize('random_seed', random_seeds)
@pytest.mark.parametrize('aggregation', ['mor', 'rom'])
def test_get_fold_score_bounds(random_seed: int, aggregation: str):
    """
    Testing the extraction of fold score bounds

    Args:
        random_seed (int): the random seed to use
        aggregation (str): the aggregation to use ('mor'/'rom')
    """

    evaluation = generate_evaluation(random_state=random_seed,
                                        aggregation=aggregation)

    evaluation = Evaluation(**evaluation)
    evaluation.sample_figures().calculate_scores()

    score_bounds = get_fold_score_bounds(evaluation, feasible=True)

    for fold in evaluation.folds:
        for key in score_bounds:
            assert score_bounds[key][0] <= fold.scores[key] <= score_bounds[key][1]

@pytest.mark.parametrize('subset', two_combs + three_combs + four_combs)
@pytest.mark.parametrize('random_seed', random_seeds)
@pytest.mark.parametrize('aggregation', ['mor'])
@pytest.mark.parametrize('rounding_decimals', [3, 4])
def test_linear_programming_success_bounds(subset: list,
                                            random_seed: int,
                                            aggregation: str,
                                            rounding_decimals: int):
    """
    Testing the linear programming functionalities by generating the evaluation
    with bounds

    Args:
        subset (list): the score subset
        random_seed (int): the random seed to use
        aggregation (str): the aggregation to use ('mor'/'rom')
        rounding_decimals (int): the number of decimals to round to
    """

    evaluation, scores = generate_evaluation(random_state=random_seed,
                                            aggregation=aggregation,
                                            feasible_fold_score_bounds=True,
                                            rounding_decimals=rounding_decimals,
                                            return_scores=True,
                                            score_subset=subset)

    evaluation = Evaluation(**evaluation)

    skeleton = Evaluation(**evaluation.to_dict())

    lp_program = solve(skeleton, scores, eps=10**(-rounding_decimals), solver=solver_timeout)

    assert lp_program.status in (0, 1)

    evaluate_timeout(lp_program, skeleton, scores, 10**(-rounding_decimals), subset)

@pytest.mark.parametrize('subset', two_combs + three_combs + four_combs)
@pytest.mark.parametrize('random_seed', random_seeds)
@pytest.mark.parametrize('aggregation', ['mor'])
@pytest.mark.parametrize('rounding_decimals', [3, 4])
def test_linear_programming_failure_bounds(subset: list,
                                            random_seed: int,
                                            aggregation: str,
                                            rounding_decimals: int):
    """
    Testing the linear programming functionalities by generating the evaluation
    with bounds

    Args:
        subset (list): the score subset
        random_seed (int): the random seed to use
        aggregation (str): the aggregation to use ('mor'/'rom')
        rounding_decimals (int): the number of decimals to round to
    """

    evaluation, scores = generate_evaluation(random_state=random_seed,
                                            aggregation=aggregation,
                                            feasible_fold_score_bounds=False,
                                            rounding_decimals=rounding_decimals,
                                            return_scores=True,
                                            score_subset=subset)

    evaluation = Evaluation(**evaluation)

    skeleton = Evaluation(**evaluation.to_dict())

    lp_program = solve(skeleton, scores, eps=10**(-rounding_decimals), solver=solver_timeout)

    assert lp_program.status in (-1, 0)

    evaluate_timeout(lp_program, skeleton, scores, 10**(-rounding_decimals), subset)

def test_others():
    """
    Testing other functionalities
    """

    evaluation = generate_evaluation(aggregation='rom',
                                        feasible_fold_score_bounds=True)
    with pytest.raises(ValueError):
        Evaluation(**evaluation)

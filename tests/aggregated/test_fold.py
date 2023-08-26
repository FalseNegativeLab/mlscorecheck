"""
This module tests the fold abstraction
"""

import pytest

import numpy as np
from mlscorecheck.aggregated import (Fold,
                                        generate_fold_specification,
                                        random_identifier,
                                        solve,
                                        compare_scores)

TOL = 1e-5

two_combs = [['acc', 'sens'], ['acc', 'spec'], ['acc', 'bacc'],
            ['sens', 'spec'], ['sens', 'bacc'], ['spec', 'bacc']]
three_combs = [['acc', 'sens', 'spec'], ['acc', 'sens', 'bacc'],
                ['acc', 'spec', 'bacc'], ['sens', 'spec', 'bacc']]
four_combs = [['acc', 'sens', 'spec', 'bacc']]

def test_random_identifier():
    """
    Testing the random identifier generation
    """

    assert random_identifier(16) != random_identifier(16)
    assert len(random_identifier(16)) == 16

def test_fold_creation():
    """
    Testing the creation of Fold objects
    """

    fold = Fold(p=5, n=10)
    assert fold.p == 5 and fold.n == 10

    fold2 = Fold(**fold.to_dict(problem_only=True))
    assert fold2.p == 5 and fold2.n == 10

    fold2 = Fold(**fold.to_dict(problem_only=False))
    assert fold2.p == 5 and fold2.n == 10

    fold1 = fold.sample(5)
    fold2 = fold.sample(6)
    assert fold1.figures != fold2.figures

    fold1 = fold.sample()
    fold2 = Fold(**fold1.to_dict(problem_only=False))
    assert fold1.figures == fold2.figures
    assert fold1.calculate_scores() == fold2.calculate_scores()
    assert fold1.calculate_scores(rounding_decimals=4) == fold2.calculate_scores(rounding_decimals=4)

    fold = Fold(p=5, n=10,
                score_bounds={key: (0, 1) for key in ['acc', 'sens', 'spec', 'bacc']})
    fold2 = Fold(**fold.to_dict(problem_only=False))
    assert fold.to_dict() == fold2.to_dict()
    assert fold.to_dict(problem_only=True) == fold2.to_dict(problem_only=True)

    fold = Fold(p=5, n=10,
                score_bounds={key: (0, 1) for key in ['acc', 'sens', 'spec', 'bacc']},
                figures={'tp': 0, 'tn': 0})
    fold2 = Fold(**fold.to_dict())
    assert fold.figures == fold2.figures

    fold = Fold(p=5, n=10, identifier='dummy')
    assert fold.to_dict()['identifier'] == 'dummy'

    fold2 = fold.sample()
    assert fold2.add_bounds(fold2.get_bounds(feasible=True)).check_bounds()['bounds_flag']
    assert not fold2.add_bounds(fold2.get_bounds(feasible=False)).check_bounds()['bounds_flag']

def test_fold_repr():
    """
    Testing the fold representation
    """

    assert isinstance(str(Fold(p=5, n=10)), str)

def test_fold_sampling():
    """
    Testing the fold sampling
    """

    fold = Fold(p=5, n=10)
    fold = fold.sample()

    scores = fold.calculate_scores()

    np.testing.assert_almost_equal(scores['acc'],
                                (fold.figures['tp'] + fold.figures['tn'])/(fold.p + fold.n))
    np.testing.assert_almost_equal(scores['sens'], fold.figures['tp']/fold.p)
    np.testing.assert_almost_equal(scores['spec'], fold.figures['tn']/fold.n)
    np.testing.assert_almost_equal(scores['bacc'], (scores['sens'] + scores['spec'])/2.0)

    assert fold.sample().calculate_scores() != fold.sample().calculate_scores()

def test_exceptions():
    """
    Testing the exceptions
    """
    fold = Fold(p=5, n=10)

    with pytest.raises(ValueError):
        fold.calculate_scores()

    with pytest.raises(ValueError):
        fold.check_bounds()

def test_fold_bounds():
    """
    Testing the use of bounds in folds
    """

    fold = Fold(p=5, n=10)
    fold = fold.sample()

    result = fold.check_bounds()
    assert result['score_bounds_flag'] is None
    assert result['bounds_flag']

    fold = fold.add_bounds(fold.get_bounds(feasible=True))

    result = fold.check_bounds()
    assert result['score_bounds_flag'] is True
    assert result['bounds_flag']

    fold = fold.add_bounds(fold.get_bounds(feasible=False))

    result = fold.check_bounds()
    assert result['score_bounds_flag'] is False
    assert result['bounds_flag'] is False

@pytest.mark.parametrize('score_subset', two_combs + three_combs + four_combs)
@pytest.mark.parametrize('rounding_decimals', [2, 3, 4])
@pytest.mark.parametrize('random_state', list(range(1, 20)))
def test_solving_success(score_subset, rounding_decimals, random_state):
    """
    Testing the successful solving capabilities
    """
    problem = Fold(**generate_fold_specification(random_state=random_state))

    sample = problem.sample(random_state)
    scores = sample.calculate_scores(score_subset, rounding_decimals)

    eps = 10**(-rounding_decimals)/2 + TOL

    result = solve(problem, scores, eps)

    assert result.status == 1

    populated = problem.populate(result)

    assert compare_scores(scores, populated.calculate_scores(), eps, score_subset)
    assert populated.check_bounds()['bounds_flag'] is True

@pytest.mark.parametrize('score_subset', two_combs + three_combs + four_combs)
@pytest.mark.parametrize('rounding_decimals', [2, 3, 4])
@pytest.mark.parametrize('random_state', list(range(1, 20)))
def test_solving_success_with_bounds(score_subset, rounding_decimals, random_state):
    """
    Testing the successful solving capabilities
    """
    problem = Fold(**generate_fold_specification(random_state=random_state))
    sample = problem.sample(random_state)
    scores = sample.calculate_scores(score_subset, rounding_decimals)
    problem = problem.add_bounds(sample.get_bounds(feasible=True))

    eps = 10**(-rounding_decimals)/2 + TOL

    result = solve(problem, scores, eps)

    assert result.status == 1

    populated = problem.populate(result)

    assert compare_scores(populated.calculate_scores(), scores, eps, score_subset)
    assert populated.check_bounds()['bounds_flag'] is True

@pytest.mark.parametrize('score_subset', three_combs + four_combs)
@pytest.mark.parametrize('rounding_decimals', [3, 4])
@pytest.mark.parametrize('random_state', list(range(1, 20)))
def test_solving_failure(score_subset, rounding_decimals, random_state):
    """
    Testing the solving capabilities with failure
    """
    random_state = np.random.RandomState(random_state)

    problem = Fold(**generate_fold_specification(random_state=random_state))
    sample = problem.sample()
    scores = sample.calculate_scores(score_subset, rounding_decimals)

    eps = 10**(-rounding_decimals)/2 + TOL

    # artificially distorting the scores
    scores = {key: random_state.random_sample() for key in scores}

    result = solve(problem, scores, eps)

    assert result.status != 1

@pytest.mark.parametrize('score_subset', two_combs + three_combs + four_combs)
@pytest.mark.parametrize('rounding_decimals', [2, 3, 4])
@pytest.mark.parametrize('random_state', list(range(1, 20)))
def test_solving_failure_with_bounds(score_subset, rounding_decimals, random_state):
    """
    Testing the failure of solving with bounds
    """
    problem = Fold(**generate_fold_specification(random_state=random_state))
    sample = problem.sample(random_state)
    scores = sample.calculate_scores(score_subset, rounding_decimals)
    problem = problem.add_bounds(sample.get_bounds(feasible=False))

    eps = 10**(-rounding_decimals)/2 + TOL

    result = solve(problem, scores, eps)

    assert result.status != 1

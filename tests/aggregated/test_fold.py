"""
This module tests the fold abstraction
"""

import pytest

import pulp as pl

import numpy as np
from mlscorecheck.aggregated import Fold

TOL = 1e-5

two_combs = [['acc', 'sens'], ['acc', 'spec'], ['acc', 'bacc'],
            ['sens', 'spec'], ['sens', 'bacc'], ['spec', 'bacc']]
three_combs = [['acc', 'sens', 'spec'], ['acc', 'sens', 'bacc'],
                ['acc', 'spec', 'bacc'], ['sens', 'spec', 'bacc']]
four_combs = [['acc', 'sens', 'spec', 'bacc']]

def test_fold_creation():
    """
    Testing the creation of Fold objects
    """

    fold = Fold(p=5, n=10, identifier='dummy')
    assert fold.to_dict() == {'p': 5, 'n': 10, 'identifier': 'dummy'}

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
    fold = fold.sample_figures(5)

    scores = fold.calculate_scores()

    np.testing.assert_almost_equal(scores['acc'],
                                (fold.tp + fold.tn)/(fold.p + fold.n))
    np.testing.assert_almost_equal(scores['sens'], fold.tp/fold.p)
    np.testing.assert_almost_equal(scores['spec'], fold.tn/fold.n)
    np.testing.assert_almost_equal(scores['bacc'], (scores['sens'] + scores['spec'])/2.0)

    assert fold.sample_figures(5).calculate_scores() != fold.sample_figures(6).calculate_scores()

def test_linear_programming():
    """
    Testing if the linear programming interfaces work
    """

    lp_problem = pl.LpProblem('dummy')

    fold = Fold(5, 10)

    fold.init_lp(scores={'acc': 0.5, 'bacc': 0.5, 'sens': 0.5, 'spec': 0.5})

    assert isinstance(fold.tp, pl.LpVariable)
    assert isinstance(fold.tn, pl.LpVariable)

    lp_problem += fold.tp + fold.tn

    lp_problem.solve()

    fold.populate(lp_problem)

    assert fold.tp == 0

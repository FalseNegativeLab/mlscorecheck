"""
Testing the problem generators for aggregated problems
"""

from mlscorecheck.aggregated import (generate_1_problem_with_evaluations,
                                        generate_problems_with_evaluations)

def test_generate_1_problem():
    """
    Testing the generation of 1 problem
    """
    _, problem = generate_1_problem_with_evaluations(folding='stratified_sklearn',
                                                            n_folds=2,
                                                            n_repeats=3,
                                                            random_state=5)
    assert 'n_repeats' in problem and 'n_folds' in problem

    _, problem = generate_1_problem_with_evaluations(folding='random',
                                                            random_state=5)
    assert 'folds' in problem

def test_generate_problems():
    """
    Testing the generation of multiple problems
    """

    _, problems = generate_problems_with_evaluations(n_problems=5,
                                                            random_state=5)

    assert len(problems) == 5

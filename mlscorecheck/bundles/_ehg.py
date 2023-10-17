"""
Test functions for the EHG problem
"""

from ..core import NUMERICAL_TOLERANCE
from ..experiments import load_ehg
from ..check import check_1_dataset_unknown_folds_mor_scores

__all__ = ['check_ehg']

def check_ehg(scores: dict,
                eps,
                n_folds: int,
                n_repeats: int,
                solver_name: str = None,
                timeout: int = None,
                verbosity: int = 1,
                numerical_tolerance: float = NUMERICAL_TOLERANCE) -> dict:
    """
    Checks the cross-validated EHG scores

    Args:
        scores (dict(str,float)): the dictionary of scores
        eps (float|dict(str,float)): the numerical uncertainties
        n_folds (int): the number of folds
        n_repeats (int): the number of repetitions
        solver_name (None|str): the solver to use
        timeout (None|int): the timeout for the linear programming solver in seconds
        verbosity (int): the verbosity level of the pulp linear programming solver
                            0: silent, non-zero: verbose.
        numerical_tolerance (float): in practice, beyond the numerical uncertainty of
                                    the scores, some further tolerance is applied. This is
                                    orders of magnitude smaller than the uncertainty of the
                                    scores. It does ensure that the specificity of the test
                                    is 1, it might slightly decrease the sensitivity.

    Returns:
        dict: the result of the consistency testing
    """
    evaluation = {'dataset': load_ehg(),
                    'folding': {'n_folds': n_folds, 'n_repeats': n_repeats}}

    return check_1_dataset_unknown_folds_mor_scores(scores=scores,
                                                    eps=eps,
                                                    dataset=evaluation['dataset'],
                                                    folding=evaluation['folding'],
                                                    solver_name=solver_name,
                                                    timeout=timeout,
                                                    verbosity=verbosity,
                                                    numerical_tolerance=numerical_tolerance)

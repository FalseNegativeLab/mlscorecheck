"""
Test functions for the EHG problem
"""

from ....core import NUMERICAL_TOLERANCE
from ....experiments import get_experiment
from ...binary import check_1_dataset_unknown_folds_mos

__all__ = ['check_tpehg']

def check_tpehg(scores: dict,
                eps,
                n_folds: int,
                n_repeats: int,
                *,
                score_bounds: dict = None,
                solver_name: str = None,
                timeout: int = None,
                verbosity: int = 1,
                numerical_tolerance: float = NUMERICAL_TOLERANCE) -> dict:
    """
    Checks the cross-validated TPEHG scores

    Args:
        scores (dict(str,float)): the dictionary of scores (supports only 'acc', 'sens', 'spec',
                                    'bacc')
        eps (float|dict(str,float)): the numerical uncertainties
        n_folds (int): the number of folds
        n_repeats (int): the number of repetitions
        score_bounds (dict(str,tuple(float,float))): the potential bounds on the scores
                                                            of the folds
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
        dict: A dictionary containing the results of the consistency check. The dictionary
        includes the following keys:

            - ``'inconsistency'``:
                A boolean flag indicating whether the set of feasible true
                positive (tp) and true negative (tn) pairs is empty. If True,
                it indicates that the provided scores are not consistent with the experiment.
            - ``'details'``:
                A list of dictionaries containing the details of the consistency tests. Each
                entry contains the specification of the folds being tested and the
                outcome of the ``check_1_dataset_known_folds_mos`` function.

    Examples:
        >>> from mlscorecheck.check.bundles.ehg import check_tpehg
        >>> # the 5-fold cross-validation scores reported in the paper
        >>> scores = {'acc': 0.9447, 'sens': 0.9139, 'spec': 0.9733}
        >>> eps = 0.0001
        >>> results = check_tpehg(scores=scores,
                                    eps=eps,
                                    n_folds=5,
                                    n_repeats=1)
        >>> results['inconsistency']
        # True
    """
    evaluation = {'dataset': get_experiment('ehg.tpehg'),
                    'folding': {'n_folds': n_folds, 'n_repeats': n_repeats}}

    return check_1_dataset_unknown_folds_mos(scores=scores,
                                                    eps=eps,
                                                    dataset=evaluation['dataset'],
                                                    folding=evaluation['folding'],
                                                    fold_score_bounds=score_bounds,
                                                    solver_name=solver_name,
                                                    timeout=timeout,
                                                    verbosity=verbosity,
                                                    numerical_tolerance=numerical_tolerance)

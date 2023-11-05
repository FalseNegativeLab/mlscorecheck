"""
This module implements the top level check function for
scores calculated by the mean-of-scores aggregation
in a k-fold scenario on one single dataset.
"""

from ...core import NUMERICAL_TOLERANCE
from ...aggregated import check_aggregated_scores, Experiment, Evaluation

__all__ = ["check_1_dataset_known_folds_mos"]


def check_1_dataset_known_folds_mos(
    dataset: dict,
    folding: dict,
    scores: dict,
    eps,
    fold_score_bounds: dict = None,
    *,
    solver_name: str = None,
    timeout: int = None,
    verbosity: int = 1,
    numerical_tolerance: float = NUMERICAL_TOLERANCE
) -> dict:
    """
    This function checks the consistency of scores calculated by applying k-fold cross validation
    to a single dataset and aggregating the figures over the folds in the mean of scores fashion.

    The test operates by constructing a linear program describing the experiment and checkings its
    feasibility.

    The test can only check the consistency of the 'acc', 'sens', 'spec' and 'bacc'
    scores. For a stronger test, one can add ``fold_score_bounds`` when, for example, the minimum
    and the maximum scores over the folds are also provided.

    Args:
        dataset (dict): The dataset specification.
        folding (dict): The folding specification.
        scores (dict(str,float)): The scores to check.
        eps (float|dict(str,float)): The numerical uncertainty(ies) of the scores.
        fold_score_bounds (None|dict(str,tuple(float,float))): Bounds on the scores in the folds.
        solver_name (None|str): The solver to use.
        timeout (None|int): The timeout for the linear programming solver in seconds.
        verbosity (int): The verbosity level of the pulp linear programming solver.
                         0: silent, non-zero: verbose.
        numerical_tolerance (float): In practice, beyond the numerical uncertainty of
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
            - ``'lp_status'``:
                The status of the lp solver.
            - ``'lp_configuration_scores_match'``:
                A flag indicating if the scores from the lp configuration match the scores
                provided.
            - ``'lp_configuration_bounds_match'``:
                Indicates if the specified bounds match the actual figures.
            - ``'lp_configuration'``:
                Contains the actual configuration of the linear programming solver.

    Raises:
        ValueError: if the problem is not specified properly

    Examples:
        >>> from mlscorecheck.check.binary import check_1_dataset_known_folds_mos
        >>> dataset = {'p': 126, 'n': 131}
        >>> folding = {'folds': [{'p': 52, 'n': 94}, {'p': 74, 'n': 37}]}
        >>> scores = {'acc': 0.573, 'sens': 0.768, 'bacc': 0.662}
        >>> result = check_1_dataset_known_folds_mos(dataset=dataset,
                                                    folding=folding,
                                                    scores=scores,
                                                    eps=1e-3)
        >>> result['inconsistency']
        # False

        >>> dataset = {'p': 398, 'n': 569}
        >>> folding = {'n_folds': 4, 'n_repeats': 2, 'strategy': 'stratified_sklearn'}
        >>> scores = {'acc': 0.9, 'spec': 0.9, 'sens': 0.6}
        >>> result = check_1_dataset_known_folds_mos(dataset=dataset,
                                                    folding=folding,
                                                    scores=scores,
                                                    eps=1e-2)
        >>> result['inconsistency']
        # True

        >>> dataset = {'dataset_name': 'common_datasets.glass_0_1_6_vs_2'}
        >>> folding = {'n_folds': 4, 'n_repeats': 2, 'strategy': 'stratified_sklearn'}
        >>> scores = {'acc': 0.9, 'spec': 0.9, 'sens': 0.6, 'bacc': 0.1, 'f1': 0.95}
        >>> result = check_1_dataset_known_folds_mos(dataset=dataset,
                                                    folding=folding,
                                                    fold_score_bounds={'acc': (0.8, 1.0)},
                                                    scores=scores,
                                                    eps=1e-2,
                                                    numerical_tolerance=1e-6)
        >>> result['inconsistency']
        # True
    """

    evaluation = Evaluation(
        dataset=dataset,
        folding=folding,
        fold_score_bounds=fold_score_bounds,
        aggregation="mos",
    )

    experiment = Experiment(evaluations=[evaluation.to_dict()], aggregation="mos")

    return check_aggregated_scores(
        experiment=experiment.to_dict(),
        scores=scores,
        eps=eps,
        solver_name=solver_name,
        timeout=timeout,
        verbosity=verbosity,
        numerical_tolerance=numerical_tolerance,
    )

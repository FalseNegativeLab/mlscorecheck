"""
This module implements consistency testing for scores calculated in a k-fold cross-validation
scenario with unknown fold configuration.
"""

from ...core import NUMERICAL_TOLERANCE
from ...aggregated import Dataset, repeated_kfolds_generator, kfolds_generator
from ._check_1_dataset_known_folds_mos import check_1_dataset_known_folds_mos

__all__ = ["check_1_dataset_unknown_folds_mos", "estimate_n_evaluations"]


def estimate_n_evaluations(
    dataset: dict, folding: dict, available_scores: list = None
) -> int:
    """
    Estimates the number of estimations with different fold combinations.

    Args:
        dataset (dict): the dataset specification
        folding (dict): the folding specification
        available_scores (list): the list of available scores

    Returns:
        int: the estimated number of different fold configurations.
    """
    dataset = Dataset(**dataset)
    n_repeats = folding.get("n_repeats", 1)

    available_scores = [] if available_scores is None else available_scores

    count = sum(
        1
        for _ in kfolds_generator(
            {"dataset": dataset.to_dict(), "folding": folding}, available_scores
        )
    )

    return count**n_repeats


def check_1_dataset_unknown_folds_mos(
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
    Checking the consistency of scores calculated in a k-fold cross validation on a single
    dataset, in a mean-of-scores fashion, without knowing the fold configuration.
    The function generates all possible fold configurations and tests the
    consistency of each. The scores are inconsistent if all the k-fold configurations
    lead to inconsistencies identified.

    The test operates by constructing a linear program describing the experiment and checkings its
    feasibility.

    The test can only check the consistency of the 'acc', 'sens', 'spec' and 'bacc'
    scores. For a stronger test, one can add fold_score_bounds when, for example, the minimum and
    the maximum scores over the folds are also provided.

    Note that depending on the size of the dataset (especially the number of minority instances)
    and the folding configuration, this test might lead to an untractable number of problems to
    be solved. Use the function ``estimate_n_evaluations`` to get an upper bound estimate
    on the number of fold combinations.

    The evaluation of possible fold configurations stops when a feasible configuration is found.

    Args:
        dataset (dict): the dataset specification
        folding (dict): the folding specification
        scores (dict(str,float)): the scores to check
        eps (float|dict(str,float)): the numerical uncertainty(ies) of the scores
        fold_score_bounds (None|dict(str,dict(str,str))): bounds on the scores in the folds
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

    Raises:
        ValueError: if the problem is not specified properly

    Examples:
        >>> from mlscorecheck.check.binary import check_1_dataset_unknown_folds_mos
        >>> dataset = {'p': 126, 'n': 131}
        >>> folding = {'n_folds': 2, 'n_repeats': 1}
        >>> scores = {'acc': 0.573, 'sens': 0.768, 'bacc': 0.662}
        >>> result = check_1_dataset_unknown_folds_mos(dataset=dataset,
                                                        folding=folding,
                                                        scores=scores,
                                                        eps=1e-3)
        >>> result['inconsistency']
        # False

        >>> dataset = {'p': 19, 'n': 97}
        >>> folding = {'n_folds': 3, 'n_repeats': 1}
        >>> scores = {'acc': 0.9, 'spec': 0.9, 'sens': 0.6}
        >>> result = check_1_dataset_unknown_folds_mos(dataset=dataset,
                                                        folding=folding,
                                                        scores=scores,
                                                        eps=1e-4)
        >>> result['inconsistency']
        # True
    """
    evaluation = {
        "dataset": dataset,
        "folding": folding,
        "fold_score_bounds": fold_score_bounds,
        "aggregation": "mos",
    }

    results = {"details": []}

    idx = 0
    for evaluation_0 in repeated_kfolds_generator(evaluation, list(scores.keys())):
        tmp = {
            "folds": evaluation_0["folding"]["folds"],
            "details": check_1_dataset_known_folds_mos(
                scores=scores,
                eps=eps,
                dataset=evaluation_0["dataset"],
                folding=evaluation_0["folding"],
                fold_score_bounds=evaluation_0.get("fold_score_bounds"),
                solver_name=solver_name,
                timeout=timeout,
                verbosity=verbosity,
                numerical_tolerance=numerical_tolerance,
            ),
            "configuration_id": idx,
        }
        results["details"].append(tmp)
        if not tmp["details"]["inconsistency"]:
            break
        idx += 1

    results["inconsistency"] = all(
        tmp["details"]["inconsistency"] for tmp in results["details"]
    )

    return results

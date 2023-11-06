"""
This module implements the top level check function for
scores calculated by the score-of-means aggregation
in a kfold scenario on one single dataset.
"""

from ...core import NUMERICAL_TOLERANCE
from ...individual import check_scores_tptn_pairs
from ...aggregated import Experiment

__all__ = ["check_1_dataset_kfold_som"]


def check_1_dataset_kfold_som(
    dataset: dict,
    folding: dict,
    scores: dict,
    eps,
    *,
    numerical_tolerance: float = NUMERICAL_TOLERANCE,
    prefilter_by_pairs: bool = True
) -> dict:
    """
    This function checks the consistency of scores calculated by applying k-fold cross validation
    to a single dataset and aggregating the figures over the folds in the score of means fashion.
    The test is performed by exhaustively testing all possible confusion matrices.

    Args:
        dataset (dict): The dataset specification.
        folding (dict): The folding specification.
        scores (dict(str,float)): The scores to check ('acc', 'sens', 'spec', 'bacc', 'npv', 'ppv',
                                'f1', 'fm', 'f1n', 'fbp', 'fbn', 'upm', 'gm', 'mk', 'lrp', 'lrn',
                                'mcc', 'bm', 'pt', 'dor', 'ji', 'kappa'). When using f-beta
                                positive or f-beta negative, also set 'beta_positive' and
                                'beta_negative'.
        eps (float|dict(str,float)): The numerical uncertainty(ies) of the scores.
        numerical_tolerance (float, optional): In practice, beyond the numerical uncertainty of
                                            the scores, some further tolerance is applied. This
                                            is orders of magnitude smaller than the uncertainty
                                            of the scores. It does ensure that the specificity
                                            of the test is 1, it might slightly decrease the
                                            sensitivity. Defaults to NUMERICAL_TOLERANCE.
        prefilter_by_pairs (bool): whether to do a prefiltering based on the score-pair tp-tn
                                    solutions (faster)

    Returns:
        dict: A dictionary containing the results of the consistency check. The dictionary
        includes the following keys:

            - ``'inconsistency'``:
                A boolean flag indicating whether the set of feasible true
                positive (tp) and true negative (tn) pairs is empty. If True,
                it indicates that the provided scores are not consistent with the dataset.
            - ``'details'``:
                A list providing further details from the analysis of the scores one
                after the other.
            - ``'n_valid_tptn_pairs'``:
                The number of tp and tn pairs that are compatible with all
                scores.
            - ``'prefiltering_details'``:
                The results of the prefiltering by using the solutions for
                the score pairs.
            - ``'evidence'``:
                The evidence for satisfying the consistency constraints.

    Raises:
        ValueError: if the problem is not specified properly

    Examples:
        >>> from mlscorecheck.check.binary import check_1_dataset_kfold_som
        >>> dataset = {'dataset_name': 'common_datasets.monk-2'}
        >>> folding = {'n_folds': 4, 'n_repeats': 3, 'strategy': 'stratified_sklearn'}
        >>> scores = {'spec': 0.668, 'npv': 0.744, 'ppv': 0.667,
                        'bacc': 0.706, 'f1p': 0.703, 'fm': 0.704}
        >>> result = check_1_dataset_kfold_som(dataset=dataset,
                                                folding=folding,
                                                scores=scores,
                                                eps=1e-3)
        >>> result['inconsistency']
        # False

        >>> dataset = {'p': 10, 'n': 20}
        >>> folding = {'n_folds': 5, 'n_repeats': 1}
        >>> scores = {'acc': 0.428, 'npv': 0.392, 'bacc': 0.442, 'f1p': 0.391}
        >>> result = check_1_dataset_kfold_som(dataset=dataset,
                                                folding=folding,
                                                scores=scores,
                                                eps=1e-3)
        >>> result['inconsistency']
        # True

    """
    if folding.get("folds") is None and folding.get("strategy") is None:
        # any folding strategy results the same
        folding = {**folding} | {"strategy": "stratified_sklearn"}

    # creating the experiment consisting of one single dataset, the
    # outer level aggregation can be arbitrary
    experiment = Experiment(
        evaluations=[{"dataset": dataset, "folding": folding, "aggregation": "som"}],
        aggregation="som",
    )

    # executing the individual tests
    return check_scores_tptn_pairs(
        scores=scores,
        eps=eps,
        p=experiment.figures["p"],
        n=experiment.figures["n"],
        numerical_tolerance=numerical_tolerance,
        prefilter_by_pairs=prefilter_by_pairs,
    )

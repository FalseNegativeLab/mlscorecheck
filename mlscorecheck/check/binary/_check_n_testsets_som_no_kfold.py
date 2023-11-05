"""
This module implements the top level check function for
scores calculated by the score-of-means aggregation
over multiple testsets (with no kfold).
"""

from ...core import NUMERICAL_TOLERANCE
from ...individual import check_scores_tptn_pairs
from ...aggregated import Experiment, Dataset

__all__ = ["check_n_testsets_som_no_kfold"]


def check_n_testsets_som_no_kfold(
    testsets: list,
    scores: dict,
    eps,
    *,
    numerical_tolerance: float = NUMERICAL_TOLERANCE,
    prefilter_by_pairs: bool = True,
):
    """
    Checking the consistency of scores calculated by aggregating the figures
    over testsets in the score of means fashion, without k-folding.

    The test is performed by exhaustively testing all possible confusion matrices.

    Args:
        datasets (list(dict)): the specification of the evaluations
        scores (dict(str,float)): the scores to check ('acc', 'sens', 'spec',
                                    'bacc', 'npv', 'ppv', 'f1', 'fm', 'f1n',
                                    'fbp', 'fbn', 'upm', 'gm', 'mk', 'lrp', 'lrn', 'mcc',
                                    'bm', 'pt', 'dor', 'ji', 'kappa'), when using
                                    f-beta positive or f-beta negative, also set
                                    'beta_positive' and 'beta_negative'.
        eps (float|dict(str,float)): the numerical uncertainty(ies) of the scores
        numerical_tolerance (float): in practice, beyond the numerical uncertainty of
                                    the scores, some further tolerance is applied. This is
                                    orders of magnitude smaller than the uncertainty of the
                                    scores. It does ensure that the specificity of the test
                                    is 1, it might slightly decrease the sensitivity.
        prefilter_by_pairs (bool): whether to prefilter the solution space by pair
                                    solutions when possible to speed up the process

    Returns:
        dict: A dictionary containing the results of the consistency check. The dictionary
        includes the following keys:

            - ``'inconsistency'``:
                A boolean flag indicating whether the set of feasible true
                positive (tp) and true negative (tn) pairs is empty. If True,
                it indicates that the provided scores are not consistent with the experiment.
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
        >>> from mlscorecheck.check.binary import check_n_datasets_som_no_kfold
        >>> testsets = [{'p': 405, 'n': 223}, {'p': 3, 'n': 422}, {'p': 109, 'n': 404}]
        >>> scores = {'acc': 0.4719, 'npv': 0.6253, 'f1p': 0.3091}
        >>> result = check_n_datasets_som_no_kfold(testsets=testsets,
                                                    scores=scores,
                                                    eps=1e-3)
        >>> result['inconsistency']
        # False

        >>> scores['npv'] = 0.6263
        >>> result = check_n_datasets_som_no_kfold(testsets=testsets,
                                                    scores=scores,
                                                    eps=1e-3)
        >>> result['inconsistency']
        # True
    """

    datasets = [Dataset(**dataset) for dataset in testsets]

    evaluations = [
        {
            "dataset": dataset.to_dict(),
            "folding": {
                "folds": [
                    {
                        "p": dataset.p,
                        "n": dataset.n,
                        "identifier": f"{dataset.identifier}_{idx}",
                    }
                ]
            },
            "aggregation": "mos",
        }
        for idx, dataset in enumerate(datasets)
    ]

    # creating the experiment consisting of one single dataset, the
    # outer level aggregation can be arbitrary
    experiment = Experiment(evaluations=evaluations, aggregation="som")

    # executing the individual tests
    return check_scores_tptn_pairs(
        scores=scores,
        eps=eps,
        p=experiment.figures["p"],
        n=experiment.figures["n"],
        numerical_tolerance=numerical_tolerance,
        prefilter_by_pairs=prefilter_by_pairs,
    )

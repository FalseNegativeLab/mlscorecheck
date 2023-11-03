"""
This module implements tests for the DRISHTI_GS dataset
"""

from ....experiments import get_experiment
from ....core import NUMERICAL_TOLERANCE
from ...binary import check_1_testset_no_kfold
from ...binary import check_n_testsets_mos_no_kfold
from ...binary import check_n_testsets_som_no_kfold

__all__ = ['_prepare_testsets_drishti_gs',
            'check_drishti_gs_segmentation_image',
            'check_drishti_gs_segmentation_aggregated_mos',
            'check_drishti_gs_segmentation_aggregated_som',
            'check_drishti_gs_segmentation_aggregated']

def _prepare_testsets_drishti_gs(subset,
                                    target: str,
                                    confidence: float):
    """
    Preparing the testsets for the DRISHTI_GS dataset

    Args:
        subset (str|list): the subset of images to be used
        target (str): the target anatomical part ('OD'/'OC')
        confidence (float): the confidence level for thresholding (from [0,1]),
                            will be used to threshold the images at threshold*255

    Returns:
        list(dict): the list of testset specifications
    """
    data = get_experiment('retina.drishti_gs')

    if subset in ['train', 'test']:
        entries = data[subset]
    else:
        subset = [subset] if isinstance(subset, str) else subset
        entries = {}
        for identifier in subset:
            entries[identifier] = data['train'].get(identifier, data['test'].get(identifier))

    threshold = 255 * confidence
    testsets = []

    for entry in entries:
        tmp = entries[entry][target]
        total_p = 0
        total_n = 0
        for count, value in zip(tmp['counts'], tmp['values']):
            if value >= threshold:
                total_p += count
            else:
                total_n += count
        testsets.append({'p': total_p, 'n': total_n, 'identifier': entry})

    return testsets

def check_drishti_gs_segmentation_image(image_identifier: str,
                                            confidence: float,
                                            target: str,
                                            scores: dict,
                                            eps: float,
                                            *,
                                            numerical_tolerance: float = NUMERICAL_TOLERANCE):
    """
    Testing the segmentation results on one image.

    Args:
        image_identifier (str): the image identifier (e.g. '053')
        confidence (float): the confidence level (in [0,1]), used for thresholding
                            the soft segmentation ground truth image at threshold*255
        target (str): the target anatomical part ('OD'/'OC')
        scores (dict): the scores to check ('acc', 'sens', 'spec',
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

    Examples:
        >>> from mlscorecheck.check.bundles.retina import check_drishti_gs_segmentation_image
        >>> scores = {'acc': 0.5966, 'sens': 0.3, 'spec': 0.6067, 'f1p': 0.0468}
        >>> results = check_drishti_gs_segmentation_image(image_identifier='053',
                                    confidence=0.75,
                                    target='OD',
                                    scores=scores,
                                    eps=1e-4)
        >>> results['inconsistency']
        # False
    """
    testset = _prepare_testsets_drishti_gs(subset=[image_identifier],
                                            target=target,
                                            confidence=confidence)[0]

    return check_1_testset_no_kfold(testset=testset,
                                            scores=scores,
                                            eps=eps,
                                            numerical_tolerance=numerical_tolerance)

def check_drishti_gs_segmentation_aggregated_mos(subset,
                                            confidence: float,
                                            target: str,
                                            scores: dict,
                                            eps: float,
                                            *,
                                            score_bounds: dict = None,
                                            solver_name: str = None,
                                            timeout: int = None,
                                            verbosity: int = 1,
                                            numerical_tolerance: float = NUMERICAL_TOLERANCE):
    """
    Testing the scores shared for a set of images with the MoS aggregation.

    Args:
        subset (str|list): the subset ('test'/'train') or the list of identifiers,
                            e.g. ['053', '086']
        confidence (float): the confidence level (in [0,1]), used for thresholding
                            the soft segmentation ground truth image at threshold*255
        target (str): the target anatomical part ('OD'/'OC')
        scores (dict(str,float)): the scores to be tested (supports only 'acc', 'sens', 'spec',
                                    'bacc')
        eps (float|dict(str,float)): the numerical uncertainty(ies) of the scores
        score_bounds (dict(str,tuple(float,float))): the potential bounds on the scores
                                                            of the images
        solver_name (None|str): the solver to use
        timeout (None|int): the timeout for the linear programming solver in seconds
        verbosity (int): the verbosity of the linear programming solver,
                            0: silent, 1: verbose.
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
            - ``'lp_status'``:
                The status of the lp solver.
            - ``'lp_configuration_scores_match'``:
                A flag indicating if the scores from the lp configuration match the scores
                provided.
            - ``'lp_configuration_bounds_match'``:
                Indicates if the specified bounds match the actual figures.
            - ``'lp_configuration'``:
                Contains the actual configuration of the linear programming solver.
    """
    testsets = _prepare_testsets_drishti_gs(subset=subset,
                                            target=target,
                                            confidence=confidence)

    return check_n_testsets_mos_no_kfold(testsets=testsets,
                                                scores=scores,
                                                eps=eps,
                                                testset_score_bounds=score_bounds,
                                                solver_name=solver_name,
                                                timeout=timeout,
                                                verbosity=verbosity,
                                                numerical_tolerance=numerical_tolerance)

def check_drishti_gs_segmentation_aggregated_som(subset: str,
                                            confidence: float,
                                            target: str,
                                            scores: dict,
                                            eps: float,
                                            *,
                                            numerical_tolerance: float = NUMERICAL_TOLERANCE):
    """
    Testing the scores shared for a set of images with the SoM aggregation.

    Args:
        subset (str|list): the subset ('test'/'train') or the list of identifiers,
                            e.g. ['053', '086']
        confidence (float): the confidence level (in [0,1]), used for thresholding
                            the soft segmentation ground truth image at threshold*255
        target (str): the target anatomical part ('OD'/'OC')
        scores (dict(str,float)): the scores to be tested ('acc', 'sens', 'spec',
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
    """
    testsets = _prepare_testsets_drishti_gs(subset=subset,
                                            target=target,
                                            confidence=confidence)

    return check_n_testsets_som_no_kfold(testsets=testsets,
                                                scores=scores,
                                                eps=eps,
                                                numerical_tolerance=numerical_tolerance,
                                                prefilter_by_pairs=True)

def check_drishti_gs_segmentation_aggregated(subset: str,
                                            confidence: float,
                                            target: str,
                                            scores: dict,
                                            eps: float,
                                            *,
                                            score_bounds: dict = None,
                                            solver_name: str = None,
                                            timeout: int = None,
                                            verbosity: int = 1,
                                            numerical_tolerance: float = NUMERICAL_TOLERANCE):
    """
    Testing the scores shared for a set of images with both the MoS and SoM aggregations.

    Args:
        subset (str|list): the subset ('test'/'train') or the list of identifiers,
                            e.g. ['053', '086']
        confidence (float): the confidence level (in [0,1]), used for thresholding
                            the soft segmentation ground truth image at threshold*255
        target (str): the target anatomical part ('OD'/'OC')
        scores (dict(str,float)): the scores to be tested
        eps (float|dict(str,float)): the numerical uncertainty(ies) of the scores
        score_bounds (dict(str,tuple(float,float))): the potential bounds on the scores
                                                            of the images
        solver_name (None|str): the solver to use
        timeout (None|int): the timeout for the linear programming solver in seconds
        verbosity (int): the verbosity of the linear programming solver,
                            0: silent, 1: verbose.
        numerical_tolerance (float): in practice, beyond the numerical uncertainty of
                                    the scores, some further tolerance is applied. This is
                                    orders of magnitude smaller than the uncertainty of the
                                    scores. It does ensure that the specificity of the test
                                    is 1, it might slightly decrease the sensitivity.

    Returns:
        dict: The summary of the results, with the following entries:

            - ``'inconsistency'``:
                All findings.
            - ``details*``:
                The details of the analysis for the two assumptions.

    Examples:
        >>> from mlscorecheck.check.bundles.retina import check_drishti_gs_segmentation_aggregated
        >>> scores = {'acc': 0.4767, 'sens': 0.4845, 'spec': 0.4765, 'f1p': 0.0512}
        >>> results = check_drishti_gs_segmentation_aggregated(subset='test',
                                    confidence=0.75,
                                    target='OD',
                                    scores=scores,
                                    eps=1e-4)
        >>> results['inconsistency']
        # {'inconsistency_som': False, 'inconsistency_mos': False}
    """
    results = {}

    results['details_mos'] = check_drishti_gs_segmentation_aggregated_mos(subset=subset,
                                                    confidence=confidence,
                                                    target=target,
                                                    scores=scores,
                                                    eps=eps,
                                                    score_bounds=score_bounds,
                                                    solver_name=solver_name,
                                                    timeout=timeout,
                                                    verbosity=verbosity,
                                                    numerical_tolerance=numerical_tolerance)

    results['details_som'] = check_drishti_gs_segmentation_aggregated_som(subset=subset,
                                                    confidence=confidence,
                                                    target=target,
                                                    scores=scores,
                                                    eps=eps,
                                                    numerical_tolerance=numerical_tolerance)

    results['inconsistency'] = {'inconsistency_som': results['details_som']['inconsistency'],
                                'inconsistency_mos': results['details_mos']['inconsistency']}

    return results

"""
This module implements tests for the DIARETDB1 dataset
"""

from ....experiments import get_experiment
from ...binary import check_1_testset_no_kfold
from ...binary import check_n_testsets_mos_no_kfold
from ...binary import check_n_testsets_som_no_kfold
from ....core import NUMERICAL_TOLERANCE, logger

__all__ = ['_prepare_configuration_diaretdb1',
            'check_diaretdb1_class',
            'check_diaretdb1_segmentation_image_assumption',
            'check_diaretdb1_segmentation_image',
            'check_diaretdb1_segmentation_aggregated_assumption',
            'check_diaretdb1_segmentation_aggregated']

def _prepare_testsets_diaretdb1(subset_indices,
                                    data,
                                    key,
                                    assumption,
                                    threshold):
    """
    Helper function for generating the diaretdb1 evaluation configuration

    Args:
        subset_indices (list): the list of image indices to consider
        data (dict): the entire dataset
        key (str): the key on the use of the class combinations
        assumption (str): the assumption on using all pixels or the FoV ('fov'/'all')
        threshold (float): the threshold on the confidence

    Returns:
        list, dict: the image level testsets and the overall testset
    """
    testsets = []
    testset = {'p': 0, 'n': 0}

    for img_idx in subset_indices:
        values = data[img_idx][key][assumption]['values']
        counts = data[img_idx][key][assumption]['counts']

        total_p = 0
        total_n = 0
        for count, value in zip(counts, values):
            if value >= threshold:
                total_p += count
            else:
                total_n += count


        testsets.append({'identifier': img_idx, 'p': total_p, 'n': total_n})

        testset['p'] += (total_p > 0)
        testset['n'] += (total_p == 0)

    return testsets, testset

def _prepare_configuration_diaretdb1(*,
                                        subset,
                                        class_name,
                                        pixel_level: bool,
                                        assumption: str,
                                        confidence: float,
                                        only_valid=False) -> list:
    """
    Prepares the experiment confuguration based on the description

    Args:
        subset (str|list): the subset of images to be used ('train'/'test'/ a
                            list of identifiers ['001', '002', ...])
        class_name (str|list): the name or list of the class(es) to be used as the positive
                            samples ('hardexudates', 'softexudates', 'hemorrhages', 'redsmalldots')
        pixel_level (bool): whether a segmentation (True) or image labeling (False) is to be
                            prepared
        assumption (str): the assumption on the region of evaluation (only effective if
                            pixel_level=True)
        confidence (float): the float value of confidence threshold (in [0,1]) used to create the
                            final labeling
        only_valid (bool): effective only if pixel_level = True, keeps only those image statistics
                            where the foreground object is present

    Returns:
        dict|list: the dictionary of the testset if pixel_level = False, otherwise the list of
                    testset specifications
    """

    class_name = [class_name] if isinstance(class_name, str) else class_name
    data = get_experiment('retina.diaretdb1')

    subset_indices = data[subset] if isinstance(subset, str) else subset
    data = data['distributions']

    mapping = {'hardexudates': 'he',
                'softexudates': 'se',
                'hemorrhages': 'hr',
                'redsmalldots': 'rsd'}
    key = '-'.join(sorted([mapping[class_] for class_ in class_name]))

    testsets, testset = _prepare_testsets_diaretdb1(subset_indices,
                                                    data,
                                                    key,
                                                    assumption,
                                                    confidence*255)

    if only_valid:
        testsets = [tset for tset in testsets if tset['p'] > 0 and tset['n'] > 0]

    return testsets if pixel_level else testset

def check_diaretdb1_class(*,
                            subset: str,
                            class_name,
                            confidence: float,
                            scores: dict,
                            eps,
                            numerical_tolerance: float = NUMERICAL_TOLERANCE) -> dict:
    """
    Tests the scores describing the labeling of images in DIARETDB1. The problem is
    a multi-labeling problem, this test function supports binary the testing of
    binary subproblems (for example, the 'hardexudates' class being treated as
    the positive label).

    Args:
        subset (str): the subset to be used ('train'/'test'), typically 'test'
        class_name (str|list): the name or list of names of classes used as "positive"
        confidence (float): the confidence threshold, typically 0.75
        scores (dict(str,float)): the scores to be tested ('acc', 'sens', 'spec',
                                    'bacc', 'npv', 'ppv', 'f1', 'fm', 'f1n',
                                    'fbp', 'fbn', 'upm', 'gm', 'mk', 'lrp', 'lrn', 'mcc',
                                    'bm', 'pt', 'dor', 'ji', 'kappa'), when using
                                    f-beta positive or f-beta negative, also set
                                    'beta_positive' and 'beta_negative'.
        eps (float): the numerical uncertainty
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
        >>> from mlscorecheck.check.bundles.retina import check_diaretdb1_class
        >>> scores = {'acc': 0.3115, 'sens': 1.0, 'spec': 0.0455, 'f1p': 0.4474}
        >>> results = check_diaretdb1_class(subset='test',
                                class_name=['hardexudates', 'softexudates'],
                                confidence=0.75,
                                scores=scores,
                                eps=1e-4)
        >>> results['inconsistency']
        # False
    """
    testset = _prepare_configuration_diaretdb1(subset=subset,
                                class_name=class_name,
                                pixel_level=False,
                                assumption='all',
                                confidence=confidence)

    return check_1_testset_no_kfold(testset=testset,
                                            scores=scores,
                                            eps=eps,
                                            numerical_tolerance=numerical_tolerance)

def check_diaretdb1_segmentation_image_assumption(*,
                                    image_identifier: str,
                                    class_name,
                                    assumption: str,
                                    confidence: float,
                                    scores: dict,
                                    eps,
                                    numerical_tolerance: float = NUMERICAL_TOLERANCE) -> dict:
    """
    Tests the scores describing the segmentation of images in DIARETDB1. This test function
    supports binary the testing of binary subproblems (for example, the pixels of the
    'hardexudates' class being segmented in an image).

    Args:
        image_identifier (str): the identifier of the image to be tested (e.g. '001')
        class_name (str|list): the name or list of names of classes used as "positive"
        assumption (str): the assumption on the region of evaluation ('fov'/'all')
        confidence (float): the confidence threshold, typically 0.75
        scores (dict(str,float)): the scores to be tested ('acc', 'sens', 'spec',
                                    'bacc', 'npv', 'ppv', 'f1', 'fm', 'f1n',
                                    'fbp', 'fbn', 'upm', 'gm', 'mk', 'lrp', 'lrn', 'mcc',
                                    'bm', 'pt', 'dor', 'ji', 'kappa'), when using
                                    f-beta positive or f-beta negative, also set
                                    'beta_positive' and 'beta_negative'.
        eps (float): the numerical uncertainty
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
    testset_test = _prepare_configuration_diaretdb1(subset='test',
                                class_name=class_name,
                                pixel_level=True,
                                assumption=assumption,
                                confidence=confidence)
    testset_train = _prepare_configuration_diaretdb1(subset='train',
                                class_name=class_name,
                                pixel_level=True,
                                assumption=assumption,
                                confidence=confidence)
    testset = [tset for tset in testset_test + testset_train
                    if tset['identifier'] == image_identifier][0]

    return check_1_testset_no_kfold(testset=testset,
                                            scores=scores,
                                            eps=eps,
                                            numerical_tolerance=numerical_tolerance)

def check_diaretdb1_segmentation_image(*,
                                    image_identifier: str,
                                    class_name,
                                    confidence: float,
                                    scores: dict,
                                    eps,
                                    numerical_tolerance: float = NUMERICAL_TOLERANCE) -> dict:
    """
    Tests the scores describing the segmentation of images in DIARETDB1. This test function
    supports binary the testing of binary subproblems (for example, the pixels of the
    'hardexudates' class being segmented in an image). The test evaluates both assumptions
    of using the FoV or all pixels for evaluation.

    Args:
        image_identifier (str): the identifier of the image to be tested (e.g. '001')
        class_name (str|list): the name or list of names of classes used as "positive"
        confidence (float): the confidence threshold, typically 0.75
        scores (dict(str,float)): the scores to be tested ('acc', 'sens', 'spec',
                                    'bacc', 'npv', 'ppv', 'f1', 'fm', 'f1n',
                                    'fbp', 'fbn', 'upm', 'gm', 'mk', 'lrp', 'lrn', 'mcc',
                                    'bm', 'pt', 'dor', 'ji', 'kappa'), when using
                                    f-beta positive or f-beta negative, also set
                                    'beta_positive' and 'beta_negative'.
        eps (float): the numerical uncertainty
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
        >>> from mlscorecheck.check.bundles.retina import check_diaretdb1_segmentation_image
        >>> scores = {'acc': 0.5753, 'sens': 0.0503, 'spec': 0.6187, 'f1p': 0.0178}
        >>> results = check_diaretdb1_segmentation_image(image_identifier='005',
                                class_name=['hardexudates', 'softexudates'],
                                confidence=0.75,
                                scores=scores,
                                eps=1e-4)
        >>> results['inconsistency']
        # {'inconsistency_fov': True, 'inconsistency_all': False}
    """
    results = {}
    results['details_fov'] = check_diaretdb1_segmentation_image_assumption(
                                    image_identifier=image_identifier,
                                    class_name=class_name,
                                    confidence=confidence,
                                    assumption='fov',
                                    scores=scores,
                                    eps=eps,
                                    numerical_tolerance=numerical_tolerance)
    results['details_all'] = check_diaretdb1_segmentation_image_assumption(
                                    image_identifier=image_identifier,
                                    class_name=class_name,
                                    confidence=confidence,
                                    assumption='all',
                                    scores=scores,
                                    eps=eps,
                                    numerical_tolerance=numerical_tolerance)
    results['inconsistency'] = {'inconsistency_fov': results['details_fov']['inconsistency'],
                                'inconsistency_all': results['details_all']['inconsistency']}

    return results

def check_diaretdb1_segmentation_aggregated_assumption(*,
                                    subset: str,
                                    class_name,
                                    assumption: str,
                                    confidence: float,
                                    only_valid: bool,
                                    scores: dict,
                                    eps,
                                    score_bounds: dict = None,
                                    solver_name: str = None,
                                    timeout: int = None,
                                    verbosity: int = 1,
                                    numerical_tolerance: float = NUMERICAL_TOLERANCE) -> dict:
    """
    Tests the scores describing the segmentation of multiple images of DIARETDB1 in an aggregated
    way. This test function supports binary the testing of binary subproblems (for example, the
    pixels of the 'hardexudates' class being segmented in an image).

    Args:
        subset (str|list): the subset of images to be used ('train'/'test') or the list of
                            image identifiers to be tested (e.g. '001')
        class_name (str|list): the name or list of names of classes used as "positive"
        assumption (str): the assumption on the region of evaluation ('fov'/'all')
        confidence (float): the confidence threshold, typically 0.75
        only_valid (bool): if True, works with that subset of the images, where both positives and
                            negatives are present (e.g. images where the class
                            class_name='hardexudates' is not present with confidence=0.75 level
                            are discarded). If False, sensitivity is specified in ``scores`` and
                            one of the images has 0 positives, the MoS test cannot be executed
        scores (dict(str,float)): the scores to be tested
        eps (float): the numerical uncertainty
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
    """

    testsets = _prepare_configuration_diaretdb1(subset=subset,
                                class_name=class_name,
                                pixel_level=True,
                                assumption=assumption,
                                confidence=confidence,
                                only_valid=only_valid)

    results = {}
    results['details_som'] = check_n_testsets_som_no_kfold(testsets=testsets,
                                                    scores=scores,
                                                    eps=eps,
                                                    numerical_tolerance=numerical_tolerance)
    results['inconsistency'] = {'inconsistency_som': results['details_som']['inconsistency']}

    if not (any(testset['p'] == 0 for testset in testsets) and 'sens' in scores):
        results['details_mos'] = check_n_testsets_mos_no_kfold(
                                                        testsets=testsets,
                                                        scores=scores,
                                                        eps=eps,
                                                        testset_score_bounds=score_bounds,
                                                        solver_name=solver_name,
                                                        timeout=timeout,
                                                        verbosity=verbosity,
                                                        numerical_tolerance=numerical_tolerance)
        results['inconsistency']['inconsistency_mos'] = results['details_mos']['inconsistency']
    else:
        logger.info('some testsets have 0 positives and the sens score is specified, thus, MoS'
                    'aggregation is undefined')

    return results

def check_diaretdb1_segmentation_aggregated(*,
                                    subset: str,
                                    class_name,
                                    confidence: float,
                                    only_valid: bool,
                                    scores: dict,
                                    eps,
                                    score_bounds: dict = None,
                                    solver_name: str = None,
                                    timeout: int = None,
                                    verbosity: int = 1,
                                    numerical_tolerance: float = NUMERICAL_TOLERANCE) -> dict:
    """
    Tests the scores describing the segmentation of multiple images of DIARETDB1 in an aggregated
    way. This test function supports binary the testing of binary subproblems (for example, the
    pixels of the 'hardexudates' class being segmented in an image). The test evaluates
    both assumption on the region of evaluation.

    Args:
        subset (str|list): the subset of images to be used ('train'/'test') or the list of
                            image identifiers to be tested (e.g. '001')
        class_name (str|list): the name or list of names of classes used as "positive"
        confidence (float): the confidence threshold, typically 0.75
        only_valid (bool): if True, works with that subset of the images, where both positives and
                            negatives are present (e.g. images where the class
                            class_name='hardexudates' is not present with confidence=0.75 level
                            are discarded). If False, sensitivity is specified in ``scores`` and
                            one of the images has 0 positives, the MoS test cannot be executed
        scores (dict(str,float)): the scores to be tested
        eps (float): the numerical uncertainty
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
        >>> from mlscorecheck.check.bundles.retina import check_diaretdb1_segmentation_aggregated
        >>> scores = {'acc': 0.7143, 'sens': 0.3775, 'spec': 0.7244}
        >>> results = check_diaretdb1_segmentation_aggregated(subset='test',
                                class_name='hardexudates',
                                confidence=0.5,
                                only_valid=True,
                                scores=scores,
                                eps=1e-4)
        >>> results['inconsistency']
        # {'inconsistency_fov_som': True,
        # 'inconsistency_all_som': True,
        # 'inconsistency_fov_mos': False,
        # 'inconsistency_all_mos': False}
    """

    results = {}

    results['details_fov'] = check_diaretdb1_segmentation_aggregated_assumption(
                                    subset=subset,
                                    class_name=class_name,
                                    assumption='fov',
                                    confidence=confidence,
                                    only_valid=only_valid,
                                    scores=scores,
                                    eps=eps,
                                    score_bounds=score_bounds,
                                    solver_name=solver_name,
                                    timeout=timeout,
                                    verbosity=verbosity,
                                    numerical_tolerance=numerical_tolerance)

    results['details_all'] = check_diaretdb1_segmentation_aggregated_assumption(
                                    subset=subset,
                                    class_name=class_name,
                                    assumption='all',
                                    confidence=confidence,
                                    only_valid=only_valid,
                                    scores=scores,
                                    eps=eps,
                                    score_bounds=score_bounds,
                                    solver_name=solver_name,
                                    timeout=timeout,
                                    verbosity=verbosity,
                                    numerical_tolerance=numerical_tolerance)

    results['inconsistency'] = {'inconsistency_fov_som':
                                    results['details_fov']['details_som']['inconsistency'],
                                'inconsistency_all_som':
                                    results['details_all']['details_som']['inconsistency']}

    if 'details_mos' in results['details_fov']:
        results['inconsistency']['inconsistency_fov_mos'] = \
            results['details_fov']['details_mos']['inconsistency']
    if 'details_mos' in results['details_all']:
        results['inconsistency']['inconsistency_all_mos'] = \
            results['details_all']['details_mos']['inconsistency']

    return results

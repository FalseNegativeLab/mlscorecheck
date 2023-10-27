"""
This module implements the test suit for the retina vessel
segmentation HRF dataset
"""

from ...core import NUMERICAL_TOLERANCE
from ...experiments import get_experiment
from ...check import (check_1_testset_no_kfold_scores,
                        check_n_testsets_mos_no_kfold_scores,
                        check_n_testsets_som_no_kfold_scores)

__all__ = ['check_hrf_vessel_aggregated_mos_assumption',
            'check_hrf_vessel_aggregated_som_assumption',
            'check_hrf_vessel_aggregated',
            'check_hrf_vessel_image_assumption',
            'check_hrf_vessel_image',
            '_filter_hrf']

def _filter_hrf(data, imageset, assumption):
    """
    Filters the HRF dataset

    Args:
        data (dict): all data
        imageset (str|list): 'all' or the list of identifiers
        assumption (str): the assumption to test ('fov'/'all')

    Returns:
        list: the image subset specification
    """

    if imageset == 'all':
        return data[assumption]['images']

    testsets = []
    subset = data[assumption]['images']
    testsets = [entry for entry in subset if entry['identifier'] in imageset]

    return testsets


def check_hrf_vessel_aggregated_mos_assumption(imageset,
                            assumption: str,
                            scores: dict,
                            eps,
                            *,
                            score_bounds: dict = None,
                            solver_name: str = None,
                            timeout: int = None,
                            verbosity: int = 1,
                            numerical_tolerance: float = NUMERICAL_TOLERANCE) -> dict:
    """
    Checking the consistency of scores with calculated for some images of
    the HRF dataset with the mean-of-scores aggregation.

    Args:
        imageset (str|list): 'all' or the list of identifiers of images (e.g. ['13_h', '01_g'])
        assumption (str): the assumption on the region of evaluation to test ('fov'/'all')
        scores (dict): the scores to check
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
        dict: the dictionary of the results of the analysis, the
        ``inconsistency`` entry indicates if inconsistencies have
        been found. The aggregated_results entry is empty if
        the execution of the linear programming based check was
        unnecessary. The result has four more keys. Under ``lp_status``
        one finds the status of the lp solver, under ``lp_configuration_scores_match``
        one finds a flag indicating if the scores from the lp configuration
        match the scores provided, ``lp_configuration_bounds_match`` indicates
        if the specified bounds match the actual figures and finally
        ``lp_configuration`` contains the actual configuration of the
        linear programming solver.

    Raises:
        ValueError: if the problem is not specified properly
    """
    data = get_experiment('retina.hrf')

    testsets = _filter_hrf(data, imageset, assumption)

    return check_n_testsets_mos_no_kfold_scores(testsets=testsets,
                                                scores=scores,
                                                eps=eps,
                                                testset_score_bounds=score_bounds,
                                                solver_name=solver_name,
                                                timeout=timeout,
                                                verbosity=verbosity,
                                                numerical_tolerance=numerical_tolerance)


def check_hrf_vessel_aggregated_som_assumption(imageset,
                            assumption: str,
                            scores: dict,
                            eps,
                            numerical_tolerance=NUMERICAL_TOLERANCE):
    """
    Tests the consistency of scores calculated on the HRF dataset using
    the score-of-means aggregation.

    Args:
        imageset (str|list): 'all' or the list of identifiers of images (e.g. ['13_h', '01_g'])
        assumption (str): the assumption on the region of evaluation to test ('fov'/'all')
        scores (dict): the scores to check
        eps (float|dict(str,float)): the numerical uncertainty(ies) of the scores
        numerical_tolerance (float): in practice, beyond the numerical uncertainty of
                                    the scores, some further tolerance is applied. This is
                                    orders of magnitude smaller than the uncertainty of the
                                    scores. It does ensure that the specificity of the test
                                    is 1, it might slightly decrease the sensitivity.

    Returns:
        dict: a summary of the results. When the ``inconsistency`` flag is True, it indicates
        that the set of feasible ``tp``, ``tn`` pairs is empty. The list under the key
        ``details`` provides further details from the analysis of the scores one after the other.
        Under the key ``n_valid_tptn_pairs`` one finds the number of tp and tn pairs compatible with
        all scores. Under the key ``prefiltering_details`` one finds the results of the prefiltering
        by using the solutions for the score pairs.

    Raises:
        ValueError: if the problem is not specified properly
    """
    data = get_experiment('retina.hrf')

    testsets = _filter_hrf(data, imageset, assumption)

    return check_n_testsets_som_no_kfold_scores(testsets=testsets,
                                                scores=scores,
                                                eps=eps,
                                                numerical_tolerance=numerical_tolerance,
                                                prefilter_by_pairs=True)


def check_hrf_vessel_image_assumption(image_identifier: str,
                            assumption: str,
                            scores: dict,
                            eps,
                            *,
                            numerical_tolerance: float = NUMERICAL_TOLERANCE):
    """
    Testing the scores calculated for one image of the HRF dataset

    Args:
        image_identifier (str): the identifier of the image (like "01_g")
        assumption (str): the assumption on the region of evaluation to test ('fov'/'all')
        scores (dict(str,float)): the scores to be tested
        eps (float): the numerical uncertainty
        numerical_tolerance (float): the additional numerical tolerance

    Returns:
        dict: a summary of the results. When the ``inconsistency`` flag is True, it indicates
        that the set of feasible ``tp``, ``tn`` pairs is empty. The list under the key
        ``details`` provides further details from the analysis of the scores one after the other.
        Under the key ``n_valid_tptn_pairs`` one finds the number of tp and tn pairs compatible
        with all scores. Under the key ``prefiltering_details`` one finds the results of the
        prefiltering by using the solutions for the score pairs.
    """
    images = get_experiment('retina.hrf')
    testset = [image for image in images[assumption]['images']
                if image['identifier'] == image_identifier]

    testset = testset[0]

    return check_1_testset_no_kfold_scores(testset=testset,
                                            scores=scores,
                                            eps=eps,
                                            numerical_tolerance=numerical_tolerance,
                                            prefilter_by_pairs=True)


def check_hrf_vessel_aggregated(imageset,
                                scores: dict,
                                eps,
                                *,
                                score_bounds: dict = None,
                                solver_name: str = None,
                                timeout: int = None,
                                verbosity: int = 1,
                                numerical_tolerance: float = NUMERICAL_TOLERANCE) -> dict:
    """
    Testing the scores calculated for the HRF dataset with all assumptions

    Args:
        imageset (str|list): 'all' or the list of identifiers of images (e.g. ['13_h', '01_g'])
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
        dict: a summary of the results. Under the ``inconsistency`` key one finds all
        findings, under the keys ``details*`` the details of the analysis can
        be found.

    Examples:
        >>> from mlscorecheck.bundles.retina import check_hrf_vessel_aggregated
        >>> scores = {'acc': 0.4841, 'sens': 0.5665, 'spec': 0.475}
        >>> k = 4
        >>> results = check_hrf_vessel_aggregated(scores=scores,
                                                    eps=10**(-k),
                                                    imageset='all',
                                                    verbosity=0)
        >>> results['inconsistency']
        # {'inconsistency_fov_mos': False,
        # 'inconsistency_fov_som': True,
        # 'inconsistency_all_mos': False,
        # 'inconsistency_all_som': True}
    """
    results = {}

    for assumption in ['fov', 'all']:
        results[f'details_{assumption}_mos'] = check_hrf_vessel_aggregated_mos_assumption(
                                                            imageset=imageset,
                                                            assumption=assumption,
                                                            scores=scores,
                                                            eps=eps,
                                                            score_bounds=score_bounds,
                                                            solver_name=solver_name,
                                                            timeout=timeout,
                                                            verbosity=verbosity,
                                                            numerical_tolerance=numerical_tolerance)
        results[f'details_{assumption}_som'] = check_hrf_vessel_aggregated_som_assumption(
                                                            imageset=imageset,
                                                            assumption=assumption,
                                                            scores=scores,
                                                            eps=eps,
                                                            numerical_tolerance=numerical_tolerance)

    results['inconsistency'] = {f'inconsistency_{tmp}': results[f'details_{tmp}']['inconsistency']
                                    for tmp in ['fov_mos', 'fov_som', 'all_mos', 'all_som']}

    return results


def check_hrf_vessel_image(image_identifier: str,
                            scores: dict,
                            eps,
                            *,
                            numerical_tolerance: float = NUMERICAL_TOLERANCE):
    """
    Testing the scores calculated for one image of the HRF dataset with
    both assumptions on the region of evaluation ('fov'/'all')

    Args:
        image_identifier (str): the identifier of the image (like "01_g")
        scores (dict(str,float)): the scores to be tested
        eps (float): the numerical uncertainty
        numerical_tolerance (float): in practice, beyond the numerical uncertainty of
                                    the scores, some further tolerance is applied. This is
                                    orders of magnitude smaller than the uncertainty of the
                                    scores. It does ensure that the specificity of the test
                                    is 1, it might slightly decrease the sensitivity.

    Returns:
        dict: a summary of the results. Under the ``inconsistency`` key one finds all
        findings, under the keys ``details*`` the details of the analysis can
        be found.

    Examples:
        >>> from mlscorecheck.bundles.retina import check_hrf_vessel_image
        >>> scores = {'acc': 0.5562, 'sens': 0.5049, 'spec': 0.5621}
        >>> identifier = '13_h'
        >>> k = 4
        >>> results = check_hrf_vessel_image(scores=scores,
                                                eps=10**(-k),
                                                image_identifier=identifier)
        >>> results['inconsistency']
        # {'inconsistency_fov': False, 'inconsistency_all': True}
    """
    results = {}

    for assumption in ['fov', 'all']:
        results[f'details_{assumption}'] = check_hrf_vessel_image_assumption(
            image_identifier=image_identifier,
            assumption=assumption,
            scores=scores,
            eps=eps,
            numerical_tolerance=numerical_tolerance
        )

    results['inconsistency'] = {'inconsistency_fov': results['details_fov']['inconsistency'],
                                'inconsistency_all': results['details_all']['inconsistency']}

    return results

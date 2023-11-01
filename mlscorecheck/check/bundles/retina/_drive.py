"""
This module implements the test suit for the retina vessel
segmentation drive dataset
"""

from ....core import NUMERICAL_TOLERANCE
from ....experiments import get_experiment
from ...binary import (check_1_testset_no_kfold,
                        check_n_testsets_mos_no_kfold,
                        check_n_testsets_som_no_kfold)

__all__ = ['check_drive_vessel_aggregated_mos_assumption',
            'check_drive_vessel_aggregated_som_assumption',
            'check_drive_vessel_aggregated',
            'check_drive_vessel_image_assumption',
            'check_drive_vessel_image',
            '_filter_drive']

def _filter_drive(data, imageset, annotator, assumption):
    """
    Filters the DRIVE dataset

    Args:
        data (dict): all data
        imageset (str|list): the subset specification
        annotator (int): the annotation to use (1/2)
        assumption (str): the assumption to test ('fov'/'all')

    Returns:
        list: the image subset specification
    """
    if isinstance(imageset, str) and imageset in {'train', 'test'}:
        return data[(annotator, assumption)][imageset]['images']

    testsets = []
    subset_train = data[(annotator, assumption)]['train']['images']
    subset_test = data[(annotator, assumption)]['test']['images']
    testsets = [entry for entry in subset_train if entry['identifier'] in imageset]\
                + [entry for entry in subset_test if entry['identifier'] in imageset]

    return testsets


def check_drive_vessel_aggregated_mos_assumption(imageset,
                            assumption: str,
                            annotator: int,
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
    the DRIVE dataset with the mean-of-scores aggregation.

    Args:
        imageset (str|list): 'train'/'test' for all images in the train or test set, or a list of
                            identifiers of images (e.g. ['21', '22'])
        assumption (str): the assumption on the region of evaluation to test ('fov'/'all')
        annotator (int): the annotation to be used (1/2) (typically annotator 1 is used in papers)
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
    data = get_experiment('retina.drive')

    testsets = _filter_drive(data, imageset, annotator, assumption)

    return check_n_testsets_mos_no_kfold(testsets=testsets,
                                                scores=scores,
                                                eps=eps,
                                                testset_score_bounds=score_bounds,
                                                solver_name=solver_name,
                                                timeout=timeout,
                                                verbosity=verbosity,
                                                numerical_tolerance=numerical_tolerance)


def check_drive_vessel_aggregated_som_assumption(imageset,
                            assumption: str,
                            annotator: int,
                            scores: dict,
                            eps,
                            *,
                            numerical_tolerance=NUMERICAL_TOLERANCE):
    """
    Tests the consistency of scores calculated on the DRIVE dataset using
    the score-of-means aggregation.

    Args:
        imageset (str|list): 'train'/'test' for all images in the train or test set, or a list of
                            identifiers of images (e.g. ['21', '22'])
        assumption (str): the assumption on the region of evaluation to test ('fov'/'all')
        annotator (int): the annotation to be used (1/2) (typically annotator 1 is used in papers)
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
    data = get_experiment('retina.drive')

    testsets = _filter_drive(data, imageset, annotator, assumption)

    return check_n_testsets_som_no_kfold(testsets=testsets,
                                                scores=scores,
                                                eps=eps,
                                                numerical_tolerance=numerical_tolerance,
                                                prefilter_by_pairs=True)


def check_drive_vessel_image_assumption(image_identifier: str,
                            assumption: str,
                            annotator: str,
                            scores: dict,
                            eps,
                            *,
                            numerical_tolerance: float = NUMERICAL_TOLERANCE):
    """
    Testing the scores calculated for one image of the DRIVE dataset

    Args:
        image_identifier (str): the identifier of the image (like "21")
        assumption (str): the assumption on the region of evaluation to test ('fov'/'all')
        annotator (int): the annotation to use (1, 2) (typically annotator 1 is used in papers)
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
    images = get_experiment('retina.drive')
    testset = [image for image in images[(annotator, assumption)]['train']['images']
                if image['identifier'] == image_identifier]
    testset = testset + [image for image in images[(annotator, assumption)]['test']['images']
                if image['identifier'] == image_identifier]
    testset = testset[0]

    return check_1_testset_no_kfold(testset=testset,
                                            scores=scores,
                                            eps=eps,
                                            numerical_tolerance=numerical_tolerance,
                                            prefilter_by_pairs=True)


def check_drive_vessel_aggregated(imageset,
                                annotator: int,
                                scores: dict,
                                eps,
                                *,
                                score_bounds: dict = None,
                                solver_name: str = None,
                                timeout: int = None,
                                verbosity: int = 1,
                                numerical_tolerance: float = NUMERICAL_TOLERANCE) -> dict:
    """
    Testing the scores calculated for the DRIVE dataset with all assumptions

    Args:
        imageset (str|list): 'train'/'test' for all images in the train or test set, or a list of
                            identifiers of images (e.g. ['21', '22'])
        annotator (int): the annotation to use (1, 2) (typically annotator 1 is used in papers)
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
        >>> from mlscorecheck.check.bundles.retina import check_drive_vessel_aggregated
        >>> scores = {'acc': 0.9494, 'sens': 0.7450, 'spec': 0.9793}
        >>> k = 4
        >>> results = check_drive_vessel_aggregated(scores=scores,
                                                    eps=10**(-k),
                                                    imageset='test',
                                                    annotator=1,
                                                    verbosity=0)
        >>> results['inconsistency']
        # {'inconsistency_fov_mos': False,
        #  'inconsistency_fov_som': False,
        #  'inconsistency_all_mos': True,
        #  'inconsistency_all_som': True}
    """
    results = {}

    for assumption in ['fov', 'all']:
        results[f'details_{assumption}_mos'] = check_drive_vessel_aggregated_mos_assumption(
                                                            imageset=imageset,
                                                            annotator=annotator,
                                                            assumption=assumption,
                                                            scores=scores,
                                                            eps=eps,
                                                            score_bounds=score_bounds,
                                                            solver_name=solver_name,
                                                            timeout=timeout,
                                                            verbosity=verbosity,
                                                            numerical_tolerance=numerical_tolerance)
        results[f'details_{assumption}_som'] = check_drive_vessel_aggregated_som_assumption(
                                                            imageset=imageset,
                                                            annotator=annotator,
                                                            assumption=assumption,
                                                            scores=scores,
                                                            eps=eps,
                                                            numerical_tolerance=numerical_tolerance)

    results['inconsistency'] = {f'inconsistency_{tmp}': results[f'details_{tmp}']['inconsistency']
                                    for tmp in ['fov_mos', 'fov_som', 'all_mos', 'all_som']}

    return results


def check_drive_vessel_image(image_identifier: str,
                            annotator: str,
                            scores: dict,
                            eps,
                            *,
                            numerical_tolerance: float = NUMERICAL_TOLERANCE):
    """
    Testing the scores calculated for one image of the DRIVE dataset with
    both assumptions on the region of evaluation ('fov'/'all')

    Args:
        image_identifier (str): the identifier of the image (like "21")
        annotator (int): the annotation to use (1, 2) (typically annotator 1 is used in papers)
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
        >>> from mlscorecheck.check.bundles.retina import check_drive_vessel_image
        >>> scores = {'acc': 0.9633, 'sens': 0.7406, 'spec': 0.9849}
        >>> identifier = '01'
        >>> k = 4
        >>> results = check_drive_vessel_image(scores=scores,
                                                eps=10**(-k),
                                                image_identifier=identifier,
                                                annotator=1)
        >>> results['inconsistency']
        # {'inconsistency_fov': True, 'inconsistency_all': False}
    """
    results = {}

    for assumption in ['fov', 'all']:
        results[f'details_{assumption}'] = check_drive_vessel_image_assumption(
            image_identifier=image_identifier,
            assumption=assumption,
            annotator=annotator,
            scores=scores,
            eps=eps,
            numerical_tolerance=numerical_tolerance
        )

    results['inconsistency'] = {'inconsistency_fov': results['details_fov']['inconsistency'],
                                'inconsistency_all': results['details_all']['inconsistency']}

    return results

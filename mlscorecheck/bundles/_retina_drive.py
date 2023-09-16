"""
This module implements the test suit for the retina vessel
segmentation drive dataset
"""

from ..core import NUMERICAL_TOLERANCE, logger
from ..experiments import load_drive
from ..check import (check_1_testset_no_kfold_scores,
                        check_n_datasets_mor_kfold_rom_scores,
                        check_n_datasets_rom_kfold_rom_scores)

__all__ = ['drive_aggregated_fov_pixels',
            'drive_aggregated_all_pixels',
            'drive_image_fov_pixels',
            'drive_image_all_pixels',
            'drive_aggregated',
            'drive_image',
            'filter_drive']

def _drive_aggregated_test_scores(data: list,
                                    scores: dict,
                                    eps,
                                    aggregation: str,
                                    *,
                                    solver_name: str = None,
                                    timeout: int = None,
                                    verbosity: int = 1,
                                    numerical_tolerance: float = NUMERICAL_TOLERANCE) -> dict:
    """
    Checking the consistency of the a dataset.

    Args:
        data (list(dict)): the datasets
        scores (dict(str,float)): the scores to check
        eps (float|dict(str,float)): the numerical uncertainty
        aggregation (str): the mode of aggregation ('mor'/'rom')

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
        dict: the results of the analysis, the attribute ``inconsistency``
                shows if inconsistency has been found
    """
    experiment = {'evaluations': [{'dataset': dataset,
                                    'folding': {'n_folds': 1, 'n_repeats': 1},
                                    'aggregation': 'rom'} for dataset in data],
                    'aggregation': aggregation}
    if aggregation == 'mor':
        return check_n_datasets_mor_kfold_rom_scores(scores=scores,
                                                        eps=eps,
                                                        experiment=experiment,
                                                        solver_name=solver_name,
                                                        timeout=timeout,
                                                        verbosity=verbosity,
                                                        numerical_tolerance=numerical_tolerance)
    return check_n_datasets_rom_kfold_rom_scores(scores=scores,
                                                eps=eps,
                                                evaluations=experiment['evaluations'])

def filter_drive(data: list, subset: list = None) -> list:
    """
    Filter a dataset

    Args:
        data (list(dict)): a list of datasets
        subset (list|None): the list of identifiers

    Returns:
        list(dict): the filtered dataset

    Raises:
        ValueError: if the filtering results an empty set
    """
    if subset is None:
        return data
    result = [dataset for dataset in data if dataset['identifier'] in subset]
    if len(result) == 0:
        raise ValueError('There is no images remaining. Please check if the image_set '\
                        '("train"/"test") and the image identifiers are specified properly')
    return result

def drive_aggregated(scores: dict,
                        eps,
                        image_set: str,
                        subset: list = None,
                        *,
                        solver_name: str = None,
                        timeout: int = None,
                        verbosity: int = 1,
                        numerical_tolerance: float = NUMERICAL_TOLERANCE) -> dict:
    """
    Testing the consistency of DRIVE scores with multiple
    aggregation techniques and with the field of view (FoV)
    and without the field of view (no FoV). The aggregated
    check can test the 'acc', 'sens', 'spec' and 'bacc' scores.

    Args:
        scores (dict(str,float)): the scores to check
        eps (float|dict(str,float)): the numerical uncertainty
        image_set (str): the image image_set to test ('train'/'test')
        subset (list|None): the list of identifiers to involve, e.g. ['01', '02']
                            note that the identifiers need to be in accordance
                            with the image_set
        solver_name (None|str): the solver to use
        timeout (None|int): the timeout for the linear programming solver in seconds
        verbosity (int): the verbosity of the pulp linear programming solver,
                            0: silent, non-zero: verbose
        numerical_tolerance (float): in practice, beyond the numerical uncertainty of
                                    the scores, some further tolerance is applied. This is
                                    orders of magnitude smaller than the uncertainty of the
                                    scores. It does ensure that the specificity of the test
                                    is 1, it might slightly decrease the sensitivity.

    Returns:
        dict: the result of the analysis

    Raises:
        ValueError: if the filtering results an empty set

    Examples:
        >>> drive_aggregated(scores={'acc': 0.9478, 'sens': 0.8532, 'spec': 0.9801},
                            eps=1e-4,
                            image_set='test')
        # {'mor_fov_pixels_inconsistency': True,
            'mor_all_pixels_inconsistency': True,
            'rom_fov_pixels_inconsistency': True,
            'rom_all_pixels_inconsistency': True}
    """
    logger.info('testing MoR FoV pixels')
    results_fov_mor = drive_aggregated_fov_pixels(scores=scores,
                                                eps=eps,
                                                aggregation='mor',
                                                image_set=image_set,
                                                subset=subset,
                                                solver_name=solver_name,
                                                timeout=timeout,
                                                verbosity=verbosity,
                                                numerical_tolerance=numerical_tolerance)
    logger.info('testing MoR all pixels')
    results_no_fov_mor = drive_aggregated_all_pixels(scores=scores,
                                                    eps=eps,
                                                    aggregation='mor',
                                                    image_set=image_set,
                                                    subset=subset,
                                                    solver_name=solver_name,
                                                    timeout=timeout,
                                                    verbosity=verbosity,
                                                    numerical_tolerance=numerical_tolerance)
    logger.info('testing RoM FoV pixels')
    results_fov_rom = drive_aggregated_fov_pixels(scores=scores,
                                                eps=eps,
                                                aggregation='rom',
                                                image_set=image_set,
                                                subset=subset,
                                                solver_name=solver_name,
                                                timeout=timeout,
                                                verbosity=verbosity,
                                                numerical_tolerance=numerical_tolerance)
    logger.info('testing RoM all pixels')
    results_no_fov_rom = drive_aggregated_all_pixels(scores=scores,
                                                    eps=eps,
                                                    aggregation='rom',
                                                    image_set=image_set,
                                                    subset=subset,
                                                    solver_name=solver_name,
                                                    timeout=timeout,
                                                    verbosity=verbosity,
                                                    numerical_tolerance=numerical_tolerance)

    return {'mor_fov_pixels_inconsistency': results_fov_mor['inconsistency'],
            'mor_all_pixels_inconsistency': results_no_fov_mor['inconsistency'],
            'rom_fov_pixels_inconsistency': results_fov_rom['inconsistency'],
            'rom_all_pixels_inconsistency': results_no_fov_rom['inconsistency']}

def drive_image(scores: dict,
                eps,
                image_set: str,
                identifier: str) -> dict:
    """
    Testing the consistency of DRIVE a drive image scores with multiple
    with the field of view (FoV) and without the field of view (no FoV).

    Args:
        scores (dict(str,float)): the scores to check
        eps (float|dict(str,float)): the numerical uncertainty
        image_set (str): the image set to test ('train'/'test')
        identifier (str): the identifier of the image (like '01' or '22')

    Returns:
        dict: the result of the analysis

    Raises:
        ValueError: if the specified image cannot be found

    Examples:
        >>> drive_image(scores={'acc': 0.9478, 'npv': 0.8532,
                                'f1p': 0.9801, 'ppv': 0.8543},
                        eps=1e-4,
                        image_set='test',
                        identifier='01')
        # {'fov_inconsistency': True, 'no_fov_inconsistency': True}
    """
    results_fov = drive_image_fov_pixels(scores, eps, image_set, identifier)
    results_no_fov = drive_image_all_pixels(scores, eps, image_set, identifier)
    return {'fov_pixels_inconsistency': results_fov['inconsistency'],
            'all_pixels_inconsistency': results_no_fov['inconsistency']}

def drive_aggregated_fov_pixels(scores: dict,
                                eps,
                                aggregation: str,
                                image_set: str,
                                subset: list = None,
                                *,
                                solver_name: str = None,
                                timeout: int = None,
                                verbosity: int = 1,
                                numerical_tolerance: float = NUMERICAL_TOLERANCE) -> dict:
    """
    Testing the consistency of DRIVE scores with the field of view (FoV) pixels only.
    The aggregated check can test the 'acc', 'sens', 'spec' and 'bacc' scores.

    Args:
        scores (dict(str,float)): the scores to check
        eps (float|dict(str,float)): the numerical uncertainty
        aggregation (str): the aggregation technique ('mor'/'rom')
        image_set (str): the image set to test ('train'/'test')
        subset (list|None): the list of identifiers to involve, e.g. ['01', '02']
                            note that the identifiers need to be in accordance
                            with the image_set
        solver_name (None|str): the solver to use
        timeout (None|int): the timeout for the linear programming solver in seconds
        verbosity (int): the verbosity of the pulp linear programming solver,
                            0: silent, non-zero: verbose
        numerical_tolerance (float): in practice, beyond the numerical uncertainty of
                                    the scores, some further tolerance is applied. This is
                                    orders of magnitude smaller than the uncertainty of the
                                    scores. It does ensure that the specificity of the test
                                    is 1, it might slightly decrease the sensitivity.

    Returns:
        dict: the result of the analysis, the 'inconsistency' flag
        shows if inconsistency has been found, under the remaining
        keys the details of the analysis can be found

    Raises:
        ValueError: if the filtering results an empty set

    Examples:
        >>> result = drive_aggregated_fov_pixels(scores={'acc': 0.9478,
                                                            'sens': 0.8532,
                                                            'spec': 0.9801},
                                            eps=1e-4,
                                            image_set='test')
        >>> result['inconsistency']
        # True
    """
    assert image_set in ('train', 'test')
    assert aggregation in ('mor', 'rom')

    data = load_drive()[f'{image_set}_fov']['images']
    data = filter_drive(data, subset)

    return _drive_aggregated_test_scores(data,
                                            scores,
                                            eps,
                                            aggregation,
                                            solver_name=solver_name,
                                            timeout=timeout,
                                            verbosity=verbosity,
                                            numerical_tolerance=numerical_tolerance)

def drive_aggregated_all_pixels(scores: dict,
                                eps,
                                aggregation: str,
                                image_set: str,
                                subset: list = None,
                                *,
                                solver_name: str = None,
                                timeout: int = None,
                                verbosity: int = 1,
                                numerical_tolerance: float = NUMERICAL_TOLERANCE) -> dict:
    """
    Testing the consistency of DRIVE scores with all pixels.
    The aggregated check can test the 'acc', 'sens', 'spec' and 'bacc' scores.

    Args:
        scores (dict(str,float)): the scores to check
        eps (float|dict(str,float)): the numerical uncertainty
        aggregation (str): the aggregation technique ('mor'/'rom')
        image_set (str): the image set to test ('train'/'test')
        subset (list|None): the list of identifiers to involve
        solver_name (None|str): the solver to use
        timeout (None|int): the timeout for the linear programming solver in seconds
        verbosity (int): the verbosity of the pulp linear programming solver,
                            0: silent, non-zero: verbose
        numerical_tolerance (float): in practice, beyond the numerical uncertainty of
                                    the scores, some further tolerance is applied. This is
                                    orders of magnitude smaller than the uncertainty of the
                                    scores. It does ensure that the specificity of the test
                                    is 1, it might slightly decrease the sensitivity.

    Returns:
        dict: the result of the analysis, the ``inconsistency`` flag
        shows if inconsistency has been found, under the remaining
        keys the details of the analysis can be found

    Raises:
        ValueError: if the filtering results an empty set

    Examples:
        >>> result = drive_aggregated_all_pixels(scores={'acc': 0.9478,
                                                            'sens': 0.8532,
                                                            'spec': 0.9801},
                                            eps=1e-4,
                                            image_set='test')
        >>> result['inconsistency']
        # True
    """
    assert image_set in ('train', 'test')
    assert aggregation in ('mor', 'rom')

    data = load_drive()[f'{image_set}_no_fov']['images']
    data = filter_drive(data, subset)

    return _drive_aggregated_test_scores(data,
                                            scores,
                                            eps,
                                            aggregation,
                                            solver_name=solver_name,
                                            timeout=timeout,
                                            verbosity=verbosity,
                                            numerical_tolerance=numerical_tolerance)

def drive_image_fov_pixels(scores: dict,
                            eps,
                            image_set: list,
                            identifier: str) -> dict:
    """
    Testing the consistency of DRIVE a drive image scores with the field of
    view (FoV) pixels only.

    Args:
        scores (dict(str,float)): the scores to check
        eps (float|dict(str,float)): the numerical uncertainty
        image_set (str): the image set to test ('train'/'test')
        identifier (str): the identifier of the image (like '01' or '22')

    Returns:
        dict: the result of the analysis, the ``inconsistency`` flag
        shows if inconsistency has been found, the rest of the
        keys contain the the details of the analysis, for the
        interpretation see the documentation of the
        check_1_dataset_no_kfold_scores function.

    Raises:
        ValueError: if the specified image cannot be found

    Examples:
        >>> result = drive_image_fov_pixels(scores={'acc': 0.9478, 'npv': 0.8532,
                                            'f1p': 0.9801, 'ppv': 0.8543},
                                    eps=1e-4,
                                    image_set='test',
                                    identifier='01')
        >>> result['inconsistency']
        # True
    """
    assert image_set in ('train', 'test')

    data = load_drive()[f'{image_set}_fov']['images']
    image = filter_drive(data, [identifier])[0]

    return check_1_testset_no_kfold_scores(scores=scores,
                                            eps=eps,
                                            testset=image,
                                            prefilter_by_pairs=True)

def drive_image_all_pixels(scores: dict,
                            eps,
                            image_set: list,
                            identifier: str) -> dict:
    """
    Testing the consistency of DRIVE a drive image scores
    without all pixels.

    Args:
        scores (dict(str,float)): the scores to check
        eps (float|dict(str,float)): the numerical uncertainty
        image_set (str): the image set to test ('train'/'test')
        identifier (str): the identifier of the image (like '01' or '22')

    Returns:
        dict: the result of the analysis, the 'inconsistency' flag
        shows if inconsistency has been found, the rest of the
        keys contain the the details of the analysis, for the
        interpretation see the documentation of the
        check_1_dataset_no_kfold_scores function.

    Raises:
        ValueError: if the specified image cannot be found

    Examples:
        >>> result = drive_image_all_pixels(scores={'acc': 0.9478, 'npv': 0.8532,
                                            'f1p': 0.9801, 'ppv': 0.8543},
                                    eps=1e-4,
                                    image_set='test',
                                    identifier='01')
        >>> result['inconsistency']
        # True
    """
    assert image_set in ('train', 'test')

    data = load_drive()[f'{image_set}_no_fov']['images']
    image = filter_drive(data, [identifier])[0]

    return check_1_testset_no_kfold_scores(scores=scores,
                                            eps=eps,
                                            testset=image,
                                            prefilter_by_pairs=True)

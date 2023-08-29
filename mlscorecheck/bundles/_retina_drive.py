"""
This module implements the test suit for the retina vessel
segmentation drive dataset
"""

from ..experiments import load_drive
from ..check import (check_1_testset_no_kfold_scores,
                        check_n_datasets_mor_kfold_rom_scores,
                        check_n_datasets_rom_kfold_rom_scores)

__all__ = ['drive_aggregated_fov',
            'drive_aggregated_no_fov',
            'drive_image_fov',
            'drive_image_no_fov',
            'drive_aggregated',
            'drive_image',
            'filter_drive']

def _drive_aggregated_test_scores(data, scores, eps, aggregation):
    """
    Checking the consistency of the a dataset.

    Args:
        data (list(dict)): the datasets
        scores (dict(str,float)): the scores to check
        eps (float/dict(str,float)): the numerical uncertainty
        aggregation (str): the mode of aggregation ('mor'/'rom')

    Returns:
        dict: the results of the analysis, the attribute 'inconsistency'
                shows if inconsistency has been found
    """
    if aggregation == 'mor':
        return check_n_datasets_mor_kfold_rom_scores(scores=scores,
                                                        eps=eps,
                                                        datasets=data)
    return check_n_datasets_rom_kfold_rom_scores(scores=scores,
                                                eps=eps,
                                                datasets=data)

def filter_drive(data, subset=None):
    """
    Filter a dataset

    Args:
        data (list(dict)): a list of datasets
        subset (list/None): the list of identifiers

    Returns:
        list(dict): the filtered dataset
    """
    if subset is None:
        return data
    result = [dataset for dataset in data if dataset['identifier'] in subset]
    if len(result) == 0:
        raise ValueError('There is no images remaining. Please check if the bundle '\
                        '("train"/"test") and the image identifiers are specified properly')
    return result

def drive_aggregated(scores, eps, bundle, subset=None):
    """
    Testing the consistency of DRIVE scores with multiple
    aggregation techniques and with the field of view (FoV)
    and without the field of view (no FoV). The aggregated
    check can test the 'acc', 'sens', 'spec' and 'bacc' scores.

    Args:
        scores (dict(str,float)): the scores to check
        eps (float/dict(str,float)): the numerical uncertainty
        bundle (str): the image bundle to test ('train'/'test')
        subset (list/None): the list of identifiers to involve, e.g. ['01', '02']
                            note that the identifiers need to be in accordance
                            with the bundle

    Returns:
        dict: the result of the analysis

    Examples:
        drive_aggregated(scores={'acc': 0.9478, 'sens': 0.8532, 'spec': 0.9801},
                            eps=1e-4,
                            bundle='test')
        >> {'mor_fov_inconsistency': True,
            'mor_no_fov_inconsistency': True,
            'rom_fov_inconsistency': True,
            'rom_no_fov_inconsistency': True}
    """
    results_fov_mor = drive_aggregated_fov(scores=scores,
                                        eps=eps,
                                        aggregation='mor',
                                        bundle=bundle,
                                        subset=subset)
    results_no_fov_mor = drive_aggregated_no_fov(scores=scores,
                                            eps=eps,
                                            aggregation='mor',
                                            bundle=bundle,
                                            subset=subset)
    results_fov_rom = drive_aggregated_fov(scores=scores,
                                        eps=eps,
                                        aggregation='rom',
                                        bundle=bundle,
                                        subset=subset)
    results_no_fov_rom = drive_aggregated_no_fov(scores=scores,
                                            eps=eps,
                                            aggregation='rom',
                                            bundle=bundle,
                                            subset=subset)
    return {'mor_fov_inconsistency': results_fov_mor['inconsistency'],
            'mor_no_fov_inconsistency': results_no_fov_mor['inconsistency'],
            'rom_fov_inconsistency': results_fov_rom['inconsistency'],
            'rom_no_fov_inconsistency': results_no_fov_rom['inconsistency']}

def drive_image(scores, eps, bundle, identifier):
    """
    Testing the consistency of DRIVE a drive image scores with multiple
    with the field of view (FoV) and without the field of view (no FoV).

    Args:
        scores (dict(str,float)): the scores to check
        eps (float/dict(str,float)): the numerical uncertainty
        bundle (str): the image bundle to test ('train'/'test')
        subset (list/None): the list of identifiers to involve, e.g. ['01', '02']
                            note that the identifiers need to be in accordance
                            with the bundle

    Returns:
        dict: the result of the analysis

    Examples:
        drive_image(scores={'acc': 0.9478, 'npv': 0.8532,
                            'f1p': 0.9801, 'ppv': 0.8543},
                    eps=1e-4,
                    bundle='test',
                    identifier='01')
        >> {'fov_inconsistency': True, 'no_fov_inconsistency': True}
    """
    results_fov = drive_image_fov(scores, eps, bundle, identifier)
    results_no_fov = drive_image_no_fov(scores, eps, bundle, identifier)
    return {'fov_inconsistency': results_fov['inconsistency'],
            'no_fov_inconsistency': results_no_fov['inconsistency']}

def drive_aggregated_fov(scores, eps, aggregation, bundle, subset=None):
    """
    Testing the consistency of DRIVE scores with the field of view (FoV).
    The aggregated check can test the 'acc', 'sens', 'spec' and 'bacc' scores.

    Args:
        scores (dict(str,float)): the scores to check
        eps (float/dict(str,float)): the numerical uncertainty
        aggregation (str): the aggregation technique ('mor'/'rom')
        bundle (str): the image bundle to test ('train'/'test')
        subset (list/None): the list of identifiers to involve, e.g. ['01', '02']
                            note that the identifiers need to be in accordance
                            with the bundle

    Returns:
        dict: the result of the analysis, the 'inconsistency' flag
                shows if inconsistency has been found, under the remaining
                keys the details of the analysis can be found

    Examples:
        result = drive_aggregated_fov(scores={'acc': 0.9478, 'sens': 0.8532, 'spec': 0.9801},
                                        eps=1e-4,
                                        bundle='test')
        result['inconsistency']
        >> True
    """
    assert bundle in ('train', 'test')
    assert aggregation in ('mor', 'rom')

    data = load_drive()[f'{bundle}_fov']['datasets']
    data = filter_drive(data, subset)

    return _drive_aggregated_test_scores(data, scores, eps, aggregation)

def drive_aggregated_no_fov(scores, eps, aggregation, bundle, subset=None):
    """
    Testing the consistency of DRIVE scores with no field of view (no FoV).
    The aggregated check can test the 'acc', 'sens', 'spec' and 'bacc' scores.

    Args:
        scores (dict(str,float)): the scores to check
        eps (float/dict(str,float)): the numerical uncertainty
        aggregation (str): the aggregation technique ('mor'/'rom')
        bundle (str): the image bundle to test ('train'/'test')
        subset (list/None): the list of identifiers to involve

    Returns:
        dict: the result of the analysis, the 'inconsistency' flag
                shows if inconsistency has been found, under the remaining
                keys the details of the analysis can be found

    Examples:
        result = drive_aggregated_no_fov(scores={'acc': 0.9478, 'sens': 0.8532, 'spec': 0.9801},
                                        eps=1e-4,
                                        bundle='test')
        result['inconsistency']
        >> True
    """
    assert bundle in ('train', 'test')
    assert aggregation in ('mor', 'rom')

    data = load_drive()[f'{bundle}_no_fov']['datasets']
    data = filter_drive(data, subset)

    return _drive_aggregated_test_scores(data, scores, eps, aggregation)

def drive_image_fov(scores, eps, bundle, identifier):
    """
    Testing the consistency of DRIVE a drive image scores with the field of
    view (FoV).

    Args:
        scores (dict(str,float)): the scores to check
        eps (float/dict(str,float)): the numerical uncertainty
        bundle (str): the image bundle to test ('train'/'test')
        identifier (str): the identifier of the image (like '01' or '22')

    Returns:
        dict: the result of the analysis, the 'inconsistency' flag
                shows if inconsistency has been found, the rest of the
                keys contain the the details of the analysis, for the
                interpretation see the documentation of the
                check_1_dataset_no_kfold_scores function.

    Examples:
        result = drive_image_fov(scores={'acc': 0.9478, 'npv': 0.8532,
                                        'f1p': 0.9801, 'ppv': 0.8543},
                                eps=1e-4,
                                bundle='test',
                                identifier='01')
        result['inconsistency']
        >> True
    """
    assert bundle in ('train', 'test')

    data = load_drive()[f'{bundle}_fov']['datasets']
    image = filter_drive(data, [identifier])[0]['folds'][0]

    return check_1_testset_no_kfold_scores(scores=scores,
                                            eps=eps,
                                            testset=image)

def drive_image_no_fov(scores, eps, bundle, identifier):
    """
    Testing the consistency of DRIVE a drive image scores
    without the field of view (no FoV).

    Args:
        scores (dict(str,float)): the scores to check
        eps (float/dict(str,float)): the numerical uncertainty
        bundle (str): the image bundle to test ('train'/'test')
        identifier (str): the identifier of the image (like '01' or '22')

    Returns:
        dict: the result of the analysis, the 'inconsistency' flag
                shows if inconsistency has been found, the rest of the
                keys contain the the details of the analysis, for the
                interpretation see the documentation of the
                check_1_dataset_no_kfold_scores function.

    Examples:
        result = drive_image_fov(scores={'acc': 0.9478, 'npv': 0.8532,
                                        'f1p': 0.9801, 'ppv': 0.8543},
                                eps=1e-4,
                                bundle='test',
                                identifier='01')
        result['inconsistency']
        >> True
    """
    assert bundle in ('train', 'test')

    data = load_drive()[f'{bundle}_no_fov']['datasets']
    image = filter_drive(data, [identifier])[0]['folds'][0]

    return check_1_testset_no_kfold_scores(scores=scores,
                                            eps=eps,
                                            testset=image)

"""
This module implements the tests for the STARE retina vessel segmentation dataset
"""

from ....experiments import get_experiment
from ...binary import check_n_testsets_som_no_kfold
from ...binary import check_n_testsets_mos_no_kfold
from ...binary import check_1_testset_no_kfold
from ....core import NUMERICAL_TOLERANCE

__all__ = ['check_stare_vessel_image',
            'check_stare_vessel_aggregated',
            'check_stare_vessel_aggregated_mos',
            'check_stare_vessel_aggregated_som',
            '_filter_stare']

def _filter_stare(data, imageset, annotator):
    """
    Filters the STARE dataset

    Args:
        data (dict): all data
        imageset (str|list): the subset specification
        annotator (str): the annotation to use ('ah'/'vk')

    Returns:
        list: the image subset specification
    """
    if imageset == 'all':
        return data[annotator]['images']

    return [dataset for dataset in data[annotator]['images'] if dataset['identifier'] in imageset]

def check_stare_vessel_aggregated_mos(imageset,
                            annotator: str,
                            scores: dict,
                            eps,
                            *,
                            score_bounds: dict = None,
                            solver_name: str = None,
                            timeout: int = None,
                            verbosity: int = 1,
                            numerical_tolerance: float = NUMERICAL_TOLERANCE) -> dict:
    """
    Checking the consistency of scores calculated for some images of
    the STARE dataset with the mean of scores aggregation.

    Args:
        imageset (str|list): 'all' if all images are used, or a list of identifiers of
                            images (e.g. ['im0082', 'im0235'])
        annotator (str): the annotation to be used ('ah'/'vk')
        scores (dict): the scores to check (supports only 'acc', 'sens', 'spec', 'bacc').
                                Full names in camel case, like
                                'positive_predictive_value', synonyms, like 'true_positive_rate'
                                or 'tpr' instead of 'sens' and complements, like
                                'false_positive_rate' for (1 - 'spec') can also be used.
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
    data = get_experiment('retina.stare')

    testsets = _filter_stare(data, imageset, annotator)

    return check_n_testsets_mos_no_kfold(testsets=testsets,
                                                scores=scores,
                                                eps=eps,
                                                testset_score_bounds=score_bounds,
                                                solver_name=solver_name,
                                                timeout=timeout,
                                                verbosity=verbosity,
                                                numerical_tolerance=numerical_tolerance)


def check_stare_vessel_aggregated_som(imageset,
                            annotator,
                            scores,
                            eps,
                            numerical_tolerance=NUMERICAL_TOLERANCE):
    """
    Tests the consistency of scores calculated on the STARE dataset using
    the score-of-means aggregation.

    Args:
        imageset (str|list): 'all' if all images are used, or a list of identifiers of
                            images (e.g. ['im0082', 'im0235'])
        annotator (str): the annotation to be used ('ah'/'vk')
        scores (dict): the scores to check ('acc', 'sens', 'spec',
                                    'bacc', 'npv', 'ppv', 'f1', 'fm', 'f1n',
                                    'fbp', 'fbn', 'upm', 'gm', 'mk', 'lrp', 'lrn', 'mcc',
                                    'bm', 'pt', 'dor', 'ji', 'kappa'). When using f-beta
                                positive or f-beta negative, also set 'beta_positive' and
                                'beta_negative'. Full names in camel case, like
                                'positive_predictive_value', synonyms, like 'true_positive_rate'
                                or 'tpr' instead of 'sens' and complements, like
                                'false_positive_rate' for (1 - 'spec') can also be used.
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
    data = get_experiment('retina.stare')

    testsets = _filter_stare(data, imageset, annotator)

    return check_n_testsets_som_no_kfold(testsets=testsets,
                                                scores=scores,
                                                eps=eps,
                                                numerical_tolerance=numerical_tolerance,
                                                prefilter_by_pairs=True)

def check_stare_vessel_aggregated(imageset,
                                annotator: str,
                                scores: dict,
                                eps,
                                *,
                                score_bounds: dict = None,
                                solver_name: str = None,
                                timeout: int = None,
                                verbosity: int = 1,
                                numerical_tolerance: float = NUMERICAL_TOLERANCE) -> dict:
    """
    Testing the scores calculated for the STARE dataset

    Args:
        imageset (str|list): 'all' if all images are used, or a list of identifiers of
                            images (e.g. ['im0082', 'im0235'])
        annotator (str): the annotation to be used ('ah'/'vk')
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
        >>> from mlscorecheck.check.bundles.retina import check_stare_vessel_aggregated
        >>> scores = {'acc': 0.4964, 'sens': 0.5793, 'spec': 0.4871, 'bacc': 0.5332}
        >>> results = check_stare_vessel_aggregated(imageset='all',
                                                    annotator='ah',
                                                    scores=scores,
                                                    eps=1e-4,
                                                    verbosity=0)
        >>> results['inconsistency']
        # {'inconsistency_mos': False, 'inconsistency_som': True}
    """
    results = {}

    results['details_mos'] = check_stare_vessel_aggregated_mos(
                                                        imageset=imageset,
                                                        annotator=annotator,
                                                        scores=scores,
                                                        eps=eps,
                                                        score_bounds=score_bounds,
                                                        solver_name=solver_name,
                                                        timeout=timeout,
                                                        verbosity=verbosity,
                                                        numerical_tolerance=numerical_tolerance)
    results['details_som'] = check_stare_vessel_aggregated_som(
                                                        imageset=imageset,
                                                        annotator=annotator,
                                                        scores=scores,
                                                        eps=eps,
                                                        numerical_tolerance=numerical_tolerance)

    results['inconsistency'] = {f'inconsistency_{tmp}': results[f'details_{tmp}']['inconsistency']
                                    for tmp in ['mos', 'som']}
    return results


def check_stare_vessel_image(image_identifier: str,
                            annotator: str,
                            scores: dict,
                            eps,
                            *,
                            numerical_tolerance: float = NUMERICAL_TOLERANCE):
    """
    Testing the scores calculated for one image of the STARE dataset

    Args:
        image_identifier (str): the identifier of the image (like "im0235")
        annotator (str): the annotation to use ('ah'/'vk')
        scores (dict(str,float)): the scores to be tested ('acc', 'sens', 'spec',
                                    'bacc', 'npv', 'ppv', 'f1', 'fm', 'f1n',
                                    'fbp', 'fbn', 'upm', 'gm', 'mk', 'lrp', 'lrn', 'mcc',
                                    'bm', 'pt', 'dor', 'ji', 'kappa'). When using f-beta
                                positive or f-beta negative, also set 'beta_positive' and
                                'beta_negative'. Full names in camel case, like
                                'positive_predictive_value', synonyms, like 'true_positive_rate'
                                or 'tpr' instead of 'sens' and complements, like
                                'false_positive_rate' for (1 - 'spec') can also be used.
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
        >>> from mlscorecheck.check.bundles.retina import check_stare_vessel_image
        >>> img_identifier = 'im0235'
        >>> scores = {'acc': 0.4699, 'npv': 0.8993, 'f1p': 0.134}
        >>> results = check_stare_vessel_image(image_identifier=img_identifier,
                                                annotator='ah',
                                                scores=scores,
                                                eps=1e-4)
        >>> results['inconsistency']
        # False
    """
    images = get_experiment('retina.stare')
    image = [image for image in images[annotator]['images']
                if image['identifier'] == image_identifier][0]

    return check_1_testset_no_kfold(testset=image,
                                            scores=scores,
                                            eps=eps,
                                            numerical_tolerance=numerical_tolerance,
                                            prefilter_by_pairs=True)

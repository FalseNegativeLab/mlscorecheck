"""
Tests for the DIARETDB0 dataset
"""

from mlscorecheck.core import NUMERICAL_TOLERANCE
from mlscorecheck.experiments import get_experiment
from mlscorecheck.check import (check_n_testsets_mos_no_kfold_scores,
                                check_n_testsets_som_no_kfold_scores)

__all__ = ['_prepare_configuration_diaretdb0',
            'check_diaretdb0_class',
            'check_diaretdb0_class_som',
            'check_diaretdb0_class_mos']

def _prepare_configuration_diaretdb0(subset: str,
                                    batch,
                                    class_name: str) -> list:
    """
    Prepare the testset specifications for a "one vs rest" setup

    Args:
        subset (str): 'train' or 'test'
        batch (str|list): 'all' or the list of batches (['1', '2', ... , '9'])
        class_name (str|list): the name of the class being evaluated ('neovascularisation'|
                        'hardexudates'|'softexudates'|'hemorrhages'|'redsmalldots'), a list
                        if multiple classes are treated as positive

    Returns:
        list: the list of testset specification
    """
    data = get_experiment('retina.diaretdb0')
    testsets = []
    class_name = [class_name] if isinstance(class_name, str) else class_name

    classes = data['classes']
    data = data[subset + 'sets']
    batch = list(data.keys()) if batch == 'all' else batch


    for bdx in batch:
        tmp = {"identifier": bdx}
        for img_iden in data[bdx]:
            if any(class_ in classes[img_iden] for class_ in class_name):
                tmp['p'] = tmp.get('p', 0) + 1
            else:
                tmp['n'] = tmp.get('n', 0) + 1
        testsets.append(tmp)

    return testsets

def check_diaretdb0_class_som(subset: str,
                                    batch,
                                    class_name,
                                    scores: dict,
                                    eps,
                                    *,
                                    numerical_tolerance: float = NUMERICAL_TOLERANCE) -> dict:
    """
    Testing the scores calculated for the DIARETDB0 dataset. The dataset is an image
    labeling dataset, where various images can be labeled by the lesion recognized on the
    images. There are 5 different lesion labels, referred to as ``class_name`` in the arguments.
    The test considers the labeling of a certain lesion (class) as a binary classification
    problem as the images with the label treated as positive and the images without the
    label treated as negative samples. Furthermore, there are multiple batches of train and
    test images (9), the list of batches used for the evaluation can be passed with the
    ``batch`` argument. The actual subset from the batches being evaluated is passed through
    the ``subset`` argument. The test assumes that the scores are aggregated across
    the batches with the SoM aggregation.

    Args:
        subset (str): 'train'/'test'
        batch (str|list): the list of batches used, 'all' for all batches, or a subset of
                        ['1', '2', ..., '9']
        class_name (str|list): the name of the class being evaluated ('neovascularisation'|
                        'hardexudates'|'softexudates'|'hemorrhages'|'redsmalldots'), a list if
                        a list of classes is treated as positive
        scores (dict(str,float)): the scores to be tested
        eps (float): the numerical uncertainty
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
    """
    testsets = _prepare_configuration_diaretdb0(subset,
                                                batch,
                                                class_name)

    return check_n_testsets_som_no_kfold_scores(testsets=testsets,
                                                    scores=scores,
                                                    eps=eps,
                                                    numerical_tolerance=numerical_tolerance)


def check_diaretdb0_class_mos(subset: str,
                                    batch,
                                    class_name,
                                    scores: dict,
                                    eps,
                                    *,
                                    solver_name: str = None,
                                    timeout: int = None,
                                    verbosity: int = 1,
                                    numerical_tolerance: float = NUMERICAL_TOLERANCE) -> dict:
    """
    Testing the scores calculated for the DIARETDB0 dataset. The dataset is an image
    labeling dataset, where various images can be labeled by the lesion recognized on the
    images. There are 5 different lesion labels, referred to as ``class_name`` in the arguments.
    The test considers the labeling of a certain lesion (class) as a binary classification
    problem as the images with the label treated as positive and the images without the
    label treated as negative samples. Furthermore, there are multiple batches of train and
    test images (9), the list of batches used for the evaluation can be passed with the
    ``batch`` argument. The actual subset from the batches being evaluated is passed through
    the ``subset`` argument. The test assumes that the scores are aggregated across
    the batches with the MoS aggregation.

    Args:
        subset (str): 'train'/'test'
        batch (str|list): the list of batches used, 'all' for all batches, or a subset of
                        ['1', '2', ..., '9']
        class_name (str|list): the name of the class being evaluated ('neovascularisation'|
                        'hardexudates'|'softexudates'|'hemorrhages'|'redsmalldots'), a list if
                        a list of classes is treated as positive
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
    """
    testsets = _prepare_configuration_diaretdb0(subset,
                                                batch,
                                                class_name)

    return check_n_testsets_mos_no_kfold_scores(testsets=testsets,
                                                        scores=scores,
                                                        eps=eps,
                                                        numerical_tolerance=numerical_tolerance,
                                                        solver_name=solver_name,
                                                        timeout=timeout,
                                                        verbosity=verbosity)


def check_diaretdb0_class(subset: str,
                            batch,
                            class_name,
                            scores: dict,
                            eps,
                            *,
                            solver_name: str = None,
                            timeout: int = None,
                            verbosity: int = 1,
                            numerical_tolerance: float = NUMERICAL_TOLERANCE) -> dict:
    """
    Testing the scores calculated for the DIARETDB0 dataset. The dataset is an image
    labeling dataset, where various images can be labeled by the lesion recognized on the
    images. There are 5 different lesion labels, referred to as ``class_name`` in the arguments.
    The test considers the labeling of a certain lesion (class) as a binary classification
    problem as the images with the label treated as positive and the images without the
    label treated as negative samples. Furthermore, there are multiple batches of train and
    test images (9), the list of batches used for the evaluation can be passed with the
    ``batch`` argument. The actual subset from the batches being evaluated is passed through
    the ``subset`` argument. The test assumes that the scores are aggregated across
    the batches, thus, executes the tests with both the SoM and MoS aggregation assumptions.

    Args:
        subset (str): 'train'/'test'
        batch (str|list): the list of batches used, 'all' for all batches, or a subset of
                        ['1', '2', ..., '9']
        class_name (str|list): the name of the class being evaluated ('neovascularisation'|
                        'hardexudates'|'softexudates'|'hemorrhages'|'redsmalldots'), a list if
                        a list of classes is treated as positive
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
        >>> from mlscorecheck.bundles.retina import check_diaretdb0_class
        >>> scores = {'acc': 0.4271, 'sens': 0.406, 'spec': 0.4765}
        >>> k = 4
        >>> results = check_diaretdb0_class(subset='test',
                                            batch='all',
                                            class_name='hardexudates',
                                            scores=scores,
                                            eps=1e-4)
        >>> results['inconsistency']
        # {'inconsistency_som': True, 'inconsistency_mos': False}
    """
    results = {}
    results['details_som'] = check_diaretdb0_class_som(subset=subset,
                                                        batch=batch,
                                                        class_name=class_name,
                                                        scores=scores,
                                                        eps=eps,
                                                        numerical_tolerance=numerical_tolerance)

    results['details_mos'] = check_diaretdb0_class_mos(subset=subset,
                                                        batch=batch,
                                                        class_name=class_name,
                                                        scores=scores,
                                                        eps=eps,
                                                        numerical_tolerance=numerical_tolerance,
                                                        solver_name=solver_name,
                                                        timeout=timeout,
                                                        verbosity=verbosity)

    results['inconsistency'] = {'inconsistency_som': results['details_som']['inconsistency'],
                                'inconsistency_mos': results['details_mos']['inconsistency']}

    return results

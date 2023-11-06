"""
This module implements some functionalities to generate random datasets, foldings,
evaluations and experiments.
"""

import numpy as np

from ..core import init_random_state, dict_minmax, dict_mean, round_scores
from ..experiments import dataset_statistics
from ..scores import calculate_scores, calculate_scores_for_lp

from ..individual import sample_multiclass_dataset, generate_multiclass_dataset
from ._folding_utils import multiclass_stratified_folds, create_folds_multiclass
from ..scores import calculate_multiclass_scores


from ._dataset import Dataset
from ._folding import Folding
from ._evaluation import Evaluation
from ._experiment import Experiment

__all__ = [
    "generate_dataset",
    "generate_folding",
    "generate_evaluation",
    "generate_experiment",
    "get_fold_score_bounds",
    "get_dataset_score_bounds",
    "generate_scores_for_testsets",
    "generate_dataset_folding_multiclass",
]


def generate_dataset(
    max_p: int = 500, max_n: int = 500, random_state=None, no_name: bool = False
) -> dict:
    """
    Generate a random dataset specification

    Args:
        max_p (int): the maximum number of positives
        max_n (int): the maximum number of negatives
        random_state (None|int|np.random.RandomState): the random seed/state to be used
        no_name (bool): if True, doesn't generate dataset specifications
                        with ``dataset_name``

    Returns:
        dict: the dataset specification
    """
    random_state = init_random_state(random_state)

    if random_state.randint(2) == 0 or no_name:
        p = random_state.randint(1, max_p + 1)
        n = random_state.randint(1, max_n + 1)
        return {"p": p, "n": n}

    return {"dataset_name": random_state.choice(list(dataset_statistics.keys()))}


def generate_folding(
    *,
    dataset: dict,
    max_folds: int = 10,
    max_repeats: int = 5,
    strategies: list = None,
    random_state=None,
    no_folds: bool = False,
) -> dict:
    """
    Generate a random folding specification for a dataset

    Args:
        dataset (dict): the dataset specification
        max_folds (int): the maximum number of folds
        max_repeats (int): the maximum number of repeats
        strategies (None|list): the list of potential folding strategies ('stratified_sklearn')
        random_state (None|int|np.random.RandomState): the random seed/state to be used
        no_folds (bool): if True, doesn't generate folding with exact folds specifications

    Returns:
        dict: the folding specification
    """
    random_state = init_random_state(random_state)

    strategies = ["stratified_sklearn"] if strategies is None else strategies

    dataset = Dataset(**dataset)
    p, n = dataset.p, dataset.n
    max_folds = min(p, n, max_folds)

    n_folds = random_state.randint(1, max_folds + 1)
    n_repeats = random_state.randint(1, max_repeats + 1)
    strategy = random_state.choice(strategies)

    if random_state.randint(2) == 0 or no_folds:
        return {"n_folds": n_folds, "n_repeats": n_repeats, "strategy": strategy}

    folding = Folding(n_folds=n_folds, n_repeats=n_repeats, strategy=strategy)

    return {
        "folds": [fold.to_dict() for fold in folding.generate_folds(dataset, "mos")]
    }


def generate_evaluation(  # pylint: disable=too-many-locals
    *,
    max_p: int = 500,
    max_n: int = 500,
    max_folds: int = 10,
    max_repeats: int = 5,
    strategies: list = None,
    feasible_fold_score_bounds: bool = None,
    aggregation: str = None,
    random_state=None,
    return_scores: bool = False,
    rounding_decimals: int = None,
    no_name: bool = False,
    no_folds: bool = False,
    score_subset: list = None,
) -> dict:
    """
    Generate a random evaluation specification

    Args:
        max_p (int): the maximum number of positives
        max_n (int): the maximum number of negatives
        max_folds (int): the maximum number of folds
        max_repeats (int): the maximum number of repeats
        strategies (None|list): the list of potential folding strategies ('stratified_sklearn')
        feasible_fold_score_bounds (None|bool): If None, no fold_score_bounds are added, if True
                                                feasible bounds are added, otherwise infeasible
                                                ones
        aggregation (None|str): if None a random aggregation is chosen, otherwise the specified
                                aggregation is used ('som'/'mos')
        random_state (None|int|np.random.RandomState): the random seed/state to be used
        return_scores (bool): whether to return the scores (corresponding to the bounds) too
        rounding_decimals (None|int): the number of decimals to round to
        no_name (bool): if True, doesn't generate evaluations with ``dataset_name``
        no_folds (bool): if True, doesn't generate foldings with a list of folds

    Returns:
        dict[,dict]: the evaluation specification (and the scores if the ``return_scores``
        parameter is set)
    """
    random_state = init_random_state(random_state)

    result = {
        "dataset": generate_dataset(
            max_p=max_p, max_n=max_n, random_state=random_state, no_name=no_name
        )
    }
    result["folding"] = generate_folding(
        dataset=result["dataset"],
        max_folds=max_folds,
        max_repeats=max_repeats,
        strategies=strategies,
        random_state=random_state,
        no_folds=no_folds,
    )

    aggregation = (
        aggregation if aggregation is not None else random_state.choice(["som", "mos"])
    )

    evaluation = Evaluation(
        dataset=result["dataset"], folding=result["folding"], aggregation=aggregation
    ).sample_figures(random_state)

    if aggregation == "som":
        scores = calculate_scores(
            problem=evaluation.figures | {"beta_positive": 2, "beta_negative": 2},
            rounding_decimals=rounding_decimals,
        )
        scores["beta_positive"] = 2
        scores["beta_negative"] = 2
    else:
        scores = evaluation.calculate_scores(
            rounding_decimals, score_subset=score_subset
        )

    if feasible_fold_score_bounds is None:
        result["fold_score_bounds"] = None
    else:
        result["fold_score_bounds"] = get_fold_score_bounds(
            evaluation, feasible_fold_score_bounds
        )

    result["aggregation"] = aggregation

    return (result, scores) if return_scores else result


def get_fold_score_bounds(
    evaluation: Evaluation, feasible: bool = True, numerical_tolerance: float = 1 * 1e-1
) -> dict:
    """
    Extract fold score bounds from an evaluation (sampled and scores computed)

    Args:
        evaluation (Evaluation): an evaluation object
        feasible (bool): whether the bounds should lead to feasible solutions
        numerical_tolerance (float): the numerical tolerance

    Returns:
        dict(str,tuple(float,float)): the score bounds
    """
    score_bounds = dict_minmax([fold.scores for fold in evaluation.folds])

    for key, value in score_bounds.items():
        score_bounds[key] = (
            max(0.0, value[0] - numerical_tolerance),
            min(1.0, value[1] + numerical_tolerance),
        )
    if feasible:
        return score_bounds

    for key, value in score_bounds.items():
        score_bounds[key] = (value[1], 1.0)

    return score_bounds


def generate_experiment(
    *,
    max_evaluations: int = 5,
    evaluation_params: dict = None,
    feasible_dataset_score_bounds: bool = None,
    aggregation: str = None,
    random_state=None,
    return_scores: bool = False,
    rounding_decimals: int = None,
    score_subset: list = None,
) -> dict:
    """
    Generate a random experiment specification

    Args:
        max_evaluations (int): the maximum number of evaluations
        evaluation_params (dict|None): the parameters of ``generate_evaluations``
        feasible_dataset_score_bounds (None|bool): whether to add feasible (``True``) or
                                                    infeasible (``False``) score bounds. No
                                                    bounds are added if None.
        aggregation (None|str): if None a random aggregation is chosen, otherwise the specified
                                aggregation is used ('som'/'mos')
        random_state (None|int|np.random.RandomState): the random seed/state to be used
        return_scores (bool): whether to return the scores (corresponding to the bounds) too
        rounding_decimals (int|None): the number of decimals to round to

    Returns:
        dict[,dict]: the experiment specification (and the scores if the ``return_scores``
        parameter is set)
    """
    if evaluation_params is None:
        evaluation_params = {}

    random_state = init_random_state(random_state)

    n_evaluations = random_state.randint(1, max_evaluations + 1)

    aggregation = (
        aggregation if aggregation is not None else random_state.choice(["som", "mos"])
    )

    evaluations = [
        generate_evaluation(**evaluation_params, random_state=random_state)
        for _ in range(n_evaluations)
    ]

    experiment = Experiment(
        evaluations=evaluations, aggregation=aggregation
    ).sample_figures(random_state)
    if aggregation == "som":
        scores = calculate_scores(
            problem=experiment.figures | {"beta_positive": 2, "beta_negative": 2},
            rounding_decimals=rounding_decimals,
        )
        scores["beta_positive"] = 2
        scores["beta_negative"] = 2
    else:
        scores = experiment.calculate_scores(
            rounding_decimals, score_subset=score_subset
        )

    if evaluation_params.get("feasible_fold_score_bounds") is not None:
        for idx, evaluation in enumerate(experiment.evaluations):
            evaluations[idx]["fold_score_bounds"] = get_fold_score_bounds(
                evaluation, evaluation_params["feasible_fold_score_bounds"]
            )

    if feasible_dataset_score_bounds is None:
        score_bounds = None
    else:
        score_bounds = get_dataset_score_bounds(
            experiment, feasible_dataset_score_bounds
        )

    experiment = {
        "evaluations": evaluations,
        "aggregation": aggregation,
        "dataset_score_bounds": score_bounds,
    }

    return (experiment, scores) if return_scores else experiment


def get_dataset_score_bounds(
    experiment: Experiment, feasible: bool = True, numerical_tolerance: float = 2 * 1e-2
) -> dict:
    """
    Extract fold score bounds from an experiment (sampled and scores computed)

    Args:
        experiment (Experiment): an experiment object
        feasible (bool): whether the bounds should lead to feasible solutions
        numerical_tolerance (float): the numerical tolerance

    Returns:
        dict(str,tuple(float,float)): the score bounds
    """
    score_bounds = dict_minmax(
        [evaluation.scores for evaluation in experiment.evaluations]
    )
    for key, value in score_bounds.items():
        score_bounds[key] = (
            max(0.0, value[0] - numerical_tolerance),
            min(1.0, value[1] + numerical_tolerance),
        )
    if feasible:
        return score_bounds

    for key, value in score_bounds.items():
        score_bounds[key] = (value[1], 1.0)

    return score_bounds


def generate_scores_for_testsets(
    testsets, rounding_decimals=None, subset=None, random_state=None, aggregation="mos"
):
    """
    Sample scores for testsets

    Args:
        testsets (list(dict)): the list of testsets
        rounding_decimals (None|int): the number of decimals to round to
        subset (None|list): the subset of scores
        aggregation (str): the mode of aggregation ('mos'/'som')
        random_state (None|int|np.random.RandomState): the random state/seed to use

    Returns:
        dict: the scores
    """
    if not isinstance(random_state, np.random.RandomState):
        random_state = np.random.RandomState(random_state)

    subset = subset if subset is not None else ["acc", "sens", "spec"]

    testsets = [{"p": testset["p"], "n": testset["n"]} for testset in testsets]
    for testset in testsets:
        testset["tp"] = random_state.randint(testset["p"] + 1)
        testset["tn"] = random_state.randint(testset["n"] + 1)

    if aggregation == "mos":
        scores = [calculate_scores_for_lp(testset) for testset in testsets]
        scores = round_scores(dict_mean(scores), rounding_decimals=rounding_decimals)
        return {key: value for key, value in scores.items() if key in subset}

    mean_figures = dict_mean(testsets)

    scores = calculate_scores(
        mean_figures | {"beta_positive": 2, "beta_negative": 2},
        rounding_decimals=rounding_decimals,
        subset=subset,
    )

    return scores | {"beta_positive": 2, "beta_negative": 2}


def generate_dataset_folding_multiclass(
    *,
    random_state=None,
    max_n_classes=5,
    min_n_classes=3,
    max_class_size=200,
    min_class_size=10,
    max_n_folds=5,
    max_n_repeats=3,
    average=None,
    aggregation=None,
    rounding_decimals=None,
    subset=None,
):
    """
    Generates a multiclass dataset and folding with scores

    Args:
        random_state (None|int|np.random.RandomState): the random state/seed to use
        max_n_classes (int): the maximum number of classes
        min_n_classes (int): the minimum number of classes
        max_class_size (int): the maximum class size
        min_class_size (int): the minimum class size
        max_n_folds (int): the maximum number of folds
        max_n_repeats (int): the maximum number of repeats
        average (str): the type of averaging to use
        aggregation (str): the type of aggregation to use
        rounding_decimals (None|int): the number of decimals to round to
        subset (None|list): the subset of scores

    Returns:
        tuple(dict,dict,dict): the dataset, folding and scores
    """
    if not isinstance(random_state, np.random.RandomState):
        random_state = np.random.RandomState(random_state)

    if not aggregation in {"mos", "som"}:
        raise ValueError(f"Unknown aggregation: {aggregation}")

    dataset = generate_multiclass_dataset(
        random_state=random_state,
        max_n_classes=max_n_classes,
        min_n_classes=min_n_classes,
        max_class_size=max_class_size,
        min_class_size=min_class_size,
    )

    folding = {
        "n_folds": min(
            [min(list(dataset.values())), random_state.randint(2, max_n_folds)]
        )
    }

    if random_state.randint(2) == 0:
        folding = folding | {
            "n_repeats": random_state.randint(1, max_n_repeats),
            "strategy": "stratified_sklearn",
        }
    elif random_state.randint(2) == 0:
        folding = {
            "folds": multiclass_stratified_folds(
                dataset=dataset, n_folds=folding["n_folds"]
            )
        }
    else:
        folding = {"n_folds": 1}

    samples = [
        sample_multiclass_dataset(dataset=fold, random_state=random_state)
        for fold in create_folds_multiclass(dataset, folding)
    ]

    if aggregation == "mos":
        scores = [
            calculate_multiclass_scores(
                sample,
                average=average,
                additional_symbols={"beta_positive": 2, "beta_negative": 2},
                subset=subset,
            )
            for sample in samples
        ]
        scores = round_scores(dict_mean(scores), rounding_decimals=rounding_decimals)
        return dataset, folding, scores

    # if aggregation == 'som':
    scores = calculate_multiclass_scores(
        np.sum(np.array(samples), axis=0),
        average=average,
        additional_symbols={"beta_positive": 2, "beta_negative": 2},
        rounding_decimals=rounding_decimals,
        subset=subset,
    )
    return dataset, folding, scores | {"beta_positive": 2, "beta_negative": 2}

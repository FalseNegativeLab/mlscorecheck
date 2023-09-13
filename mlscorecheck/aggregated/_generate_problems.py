
from ..core import (init_random_state, dict_mean, dict_minmax, NUMERICAL_TOLERANCE)
from ..experiments import dataset_statistics

from ._fold import Fold
from ._dataset import Dataset
from ._folding import Folding
from ._evaluation import Evaluation
from ._experiment import Experiment

from ._folding_utils import _create_folds

__all__ = ['generate_dataset',
            'generate_folding',
            'generate_evaluation',
            'generate_experiment',
            'get_fold_score_bounds',
            'get_dataset_score_bounds']

def generate_dataset(max_p: int=500,
                        max_n: int=500,
                        random_state=None):
    """
    Generate a random dataset specification

    Args:
        max_p (int): the maximum number of positives
        max_n (int): the maximum number of negatives
        random_state (None|int|np.random.RandomState): the random seed/state to be used

    Returns:
        dict: the dataset specification
    """
    random_state = init_random_state(random_state)

    if random_state.randint(2) == 0:
        p = random_state.randint(1, max_p+1)
        n = random_state.randint(1, max_n+1)
        return {'p': p,
                'n': n}

    return {'dataset_name': random_state.choice(list(dataset_statistics.keys()))}

def generate_folding(dataset: dict,
                        max_folds: int=10,
                        max_repeats: int=5,
                        strategies: list=None,
                        random_state=None):
    """
    Generate a random folding specification for a dataset

    Args:
        dataset (dict): the dataset specification
        max_folds (int): the maximum number of folds
        max_repeats (int): the maximum number of repeats
        strategies (None|list): the list of potential folding strategies ('stratified_sklearn')
        random_state (None|int|np.random.RandomState): the random seed/state to be used

    Returns:
        dict: the folding specification
    """
    random_state = init_random_state(random_state)

    strategies = ['stratified_sklearn'] if strategies is None else strategies

    ds = Dataset(**dataset)
    p, n = ds.p, ds.n
    max_folds = min(p, n, max_folds)

    n_folds = random_state.randint(1, max_folds+1)
    n_repeats = random_state.randint(1, max_repeats+1)
    strategy = random_state.choice(strategies)

    if random_state.randint(2) == 0:
        return {'n_folds': n_folds,
                'n_repeats': n_repeats,
                'strategy': strategy}

    folding = Folding(n_folds=n_folds,
                        n_repeats=n_repeats,
                        strategy=strategy)

    return {'folds': [fold.to_dict() for fold in folding.generate_folds(ds, 'mor')]}

def generate_evaluation(max_p: int=500,
                            max_n: int=500,
                            max_folds: int=10,
                            max_repeats: int=5,
                            strategies: list=None,
                            feasible_fold_score_bounds: bool=None,
                            aggregation: str=None,
                            random_state=None,
                            return_scores: bool=False,
                            rounding_decimals: int=None):
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
                                aggregation is used ('rom'/'mor')
        random_state (None|int|np.random.RandomState): the random seed/state to be used
        return_scores (bool): whether to return the scores (corresponding to the bounds) too

    Returns:
        dict[,dict]: the evaluation specification (and the scores if the ``return_scores``
        parameter is set)
    """
    random_state = init_random_state(random_state)
    dataset_spec = generate_dataset(max_p=max_p,
                                        max_n=max_n,
                                        random_state=random_state)
    folding_spec = generate_folding(dataset=dataset_spec,
                                        max_folds=max_folds,
                                        max_repeats=max_repeats,
                                        strategies=strategies,
                                        random_state=random_state)
    aggregation = aggregation if aggregation is not None else random_state.choice(['rom', 'mor'])

    evaluation = Evaluation(dataset=dataset_spec,
                            folding=folding_spec,
                            aggregation=aggregation).sample_figures()

    scores = evaluation.calculate_scores(rounding_decimals)

    for fold in evaluation.folds:
        print(fold.tp, fold.tn, fold.p, fold.n, fold.scores)
    print(scores)

    if feasible_fold_score_bounds is None:
        fold_score_bounds = None
    else:
        fold_score_bounds = get_fold_score_bounds(evaluation, feasible_fold_score_bounds)
    print(fold_score_bounds)

    evaluation = {'dataset': dataset_spec,
                    'folding': folding_spec,
                    'aggregation': aggregation,
                    'fold_score_bounds': fold_score_bounds}

    return (evaluation, scores) if return_scores else evaluation

def get_fold_score_bounds(evaluation: Evaluation,
                            feasible: bool=True,
                            numerical_tolerance: float=1*1e-1):
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
        score_bounds[key] = (max(0.0, value[0] - numerical_tolerance),
                                min(1.0, value[1] + numerical_tolerance))
    if feasible:
        return score_bounds

    for key, value in score_bounds.items():
        score_bounds[key] = (value[1], 1.0)

    return score_bounds

def generate_experiment(max_evaluations: int=5,
                        max_p: int=500,
                        max_n: int=500,
                        max_folds: int=10,
                        max_repeats: int=5,
                        strategies=None,
                        feasible_fold_score_bounds: bool=None,
                        feasible_dataset_score_bounds: bool=None,
                        aggregation_folds: str=None,
                        aggregation: str=None,
                        random_state=None,
                        return_scores: bool=False,
                        rounding_decimals: int=None):
    """
    Generate a random experiment specification

    Args:
        max_p (int): the maximum number of positives
        max_n (int): the maximum number of negatives
        max_folds (int): the maximum number of folds
        max_repeats (int): the maximum number of repeats
        strategies (None|list): the list of potential folding strategies ('stratified_sklearn')
        feasible_fold_score_bounds (None|bool): If None, no fold_score_bounds are added, if True
                                                feasible bounds are added, otherwise infeasible
                                                ones
        feasible_dataset_score_bounds (None|bool): If None, no dataset_score_bounds are added,
                                                if True feasible bounds are added, otherwise
                                                infeasible ones
        aggregation_folds (None|str): if None a random aggregation is chosen for the folds,
                                otherwise the specified aggregation is used ('rom'/'mor')
        aggregation (None|str): if None a random aggregation is chosen, otherwise the specified
                                aggregation is used ('rom'/'mor')
        random_state (None|int|np.random.RandomState): the random seed/state to be used
        return_scores (bool): whether to return the scores (corresponding to the bounds) too

    Returns:
        dict[,dict]: the experiment specification (and the scores if the ``return_scores``
        parameter is set)
    """
    random_state = init_random_state(random_state)

    n_evaluations = random_state.randint(1, max_evaluations+1)

    evaluations = [generate_evaluation(max_p=max_p,
                                        max_n=max_n,
                                        max_folds=max_folds,
                                        max_repeats=max_repeats,
                                        strategies=strategies,
                                        feasible_fold_score_bounds=feasible_fold_score_bounds,
                                        aggregation=aggregation_folds,
                                        random_state=random_state)
                                                for _ in range(n_evaluations)]

    aggregation = (aggregation if aggregation is not None
                            else random_state.choice(['rom', 'mor']))

    experiment = Experiment(evaluations=evaluations,
                                aggregation=aggregation).sample_figures()
    scores = experiment.calculate_scores(rounding_decimals)

    if feasible_fold_score_bounds is not None:
        for idx, evaluation in enumerate(experiment.evaluations):
            evaluations[idx]['fold_score_bounds'] = get_fold_score_bounds(evaluation,
                                                                        feasible_fold_score_bounds)

    if feasible_dataset_score_bounds is None:
        score_bounds = None
    else:
        score_bounds = get_dataset_score_bounds(experiment, feasible_dataset_score_bounds)

    experiment = {'evaluations': evaluations,
                    'aggregation': aggregation,
                    'dataset_score_bounds': score_bounds}

    return (experiment, scores) if return_scores else experiment

def get_dataset_score_bounds(experiment: Experiment,
                                feasible: bool=True,
                                numerical_tolerance: float=2*1e-2):
    """
    Extract fold score bounds from an experiment (sampled and scores computed)

    Args:
        experiment (Experiment): an experiment object
        feasible (bool): whether the bounds should lead to feasible solutions
        numerical_tolerance (float): the numerical tolerance

    Returns:
        dict(str,tuple(float,float)): the score bounds
    """
    score_bounds = dict_minmax([evaluation.scores for evaluation in experiment.evaluations])
    for key, value in score_bounds.items():
        score_bounds[key] = (max(0.0, value[0] - numerical_tolerance),
                                min(1.0, value[1] + numerical_tolerance))
    if feasible:
        return score_bounds

    for key, value in score_bounds.items():
        score_bounds[key] = (value[1], 1.0)

    return score_bounds

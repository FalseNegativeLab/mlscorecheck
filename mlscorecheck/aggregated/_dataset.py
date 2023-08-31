"""
This module implements an abstraction for a dataset
"""
# disabling pylint false positives
# pylint: disable=no-member

import pulp as pl

from ..individual import calculate_scores_for_lp, calculate_scores
from ..core import (logger, init_random_state, dict_mean, round_scores, dict_minmax,
                    NUMERICAL_TOLERANCE)
from ..experiments import lookup_dataset

from ._fold import (Fold, random_identifier, generate_fold_specification)
from ._folding import _create_folds
from ._linear_programming import add_bounds
from ._utils import check_bounds, aggregated_scores, create_bounds

from ..experiments import dataset_statistics

__all__ = ['Dataset',
            'generate_dataset_specification',
            'create_folds_for_dataset',
            'generate_dataset_and_scores']

def generate_dataset_pn(max_p,
                        max_n,
                        max_n_folds,
                        max_n_repeats,
                        random_state):
    """
    Generates a random dataset specification

    Args:
        max_p (int): the maximum number of positives
        max_n (int): the maximum number of negatives
        max_n_folds (int): the maximum number of n_folds
        max_n_repeats (int): the maximum number of n_repeats
        random_state (int|np.random.RandomState|None): the random state/seed to use

    Returns:
        dict: a random dataset specification
    """
    p = random_state.randint(1, max_p+1)
    n = random_state.randint(1, max_n+1)
    n_folds = random_state.randint(1, min(p+1, n+1, max_n_folds))
    n_repeats = random_state.randint(1, max_n_repeats+1)

    dataset = {'p': p,
                'n': n,
                'n_folds': n_folds,
                'n_repeats': n_repeats}
    if n_folds > 1:
        dataset['folding'] = 'stratified_sklearn'

    return dataset

def generate_dataset_name(max_n_folds,
                            max_n_repeats,
                            random_state):
    """
    Generates a random dataset specification

    Args:
        max_n_folds (int): the maximum number of n_folds
        max_n_repeats (int): the maximum number of n_repeats
        random_state (int|np.random.RandomState|None): the random state/seed to use

    Returns:
        dict: a random dataset specification
    """
    name = random_state.choice(list(dataset_statistics.keys()))
    details = dataset_statistics[name]
    p = details['p']
    n = details['n']
    n_folds = random_state.randint(1, min(p+1, n+1, max_n_folds))
    n_repeats = random_state.randint(1, max_n_repeats+1)

    dataset = {'name': name,
                'n_folds': n_folds,
                'n_repeats': n_repeats}
    if n_folds > 1:
        dataset['folding'] = 'stratified_sklearn'

    return dataset

def generate_dataset_folds(max_n_folds,
                            max_n_repeats,
                            random_state):
    """
    Generates a random dataset specification

    Args:
        max_n_folds (int): the maximum number of n_folds
        max_n_repeats (int): the maximum number of n_repeats
        random_state (int|np.random.RandomState|None): the random state/seed to use

    Returns:
        dict: a random dataset specification
    """
    n_folds = random_state.randint(1, max_n_folds+1) * random_state.randint(1, max_n_repeats+1)
    folds = [generate_fold_specification(random_state=random_state) for _ in range(n_folds)]

    return {'folds': folds}

def generate_dataset_specification(*,
                                    max_p=1000,
                                    max_n=1000,
                                    max_n_folds=10,
                                    max_n_repeats=5,
                                    random_state=None,
                                    aggregation=None):
    """
    Generates a random dataset specification

    Args:
        max_p (int): the maximum number of positives
        max_n (int): the maximum number of negatives
        max_n_folds (int): the maximum number of n_folds
        max_n_repeats (int): the maximum number of n_repeats
        random_state (int|np.random.RandomState|None): the random state/seed to use
        aggregation (None|str): 'mor'/'rom' - the aggregation to use if specified

    Returns:
        dict: a random dataset specification
    """
    random_state = init_random_state(random_state)

    spec_type = random_state.randint(3)
    if spec_type == 0:
        # the dataset specification type with p, n, n_folds, n_repeats
        dataset = generate_dataset_pn(max_p=max_p,
                                        max_n=max_n,
                                        max_n_folds=max_n_folds,
                                        max_n_repeats=max_n_repeats,
                                        random_state=random_state)
    elif spec_type == 1:
        # the dataset specification type with a predefined dataset name
        dataset = generate_dataset_name(max_n_folds=max_n_folds,
                                        max_n_repeats=max_n_repeats,
                                        random_state=random_state)
    elif spec_type == 2:
        # a dataset specification with random folds
        dataset = generate_dataset_folds(max_n_folds=max_n_folds,
                                            max_n_repeats=max_n_repeats,
                                            random_state=random_state)

    dataset['aggregation'] = (random_state.choice(['mor', 'rom'])
                                if aggregation is None else aggregation)
    dataset['identifier'] = random_state.choice([None, random_identifier(8)])

    return dataset

def generate_dataset_and_scores(*,
                                score_subset=None,
                                rounding_decimals=None,
                                fold_score_bounds=False,
                                feasible_score_bounds=True,
                                max_p=1000,
                                max_n=1000,
                                max_n_folds=10,
                                max_n_repeats=5,
                                random_state=None,
                                aggregation=None):
    """
    Generate a dataset and scores from a sample

    Args:
        score_subset (list|None): the list of scores to calculate
        rounding_decimals (ident|None): the number of decimal places to round to
        fold_score_bounds (bool): whether to add fold bounds
        feasible_score_bounds (bool): whether the fold bounds should be feasible
        max_p (int): the maximum number of positives
        max_n (int): the maximum number of negatives
        max_n_folds (int): the maximum number of n_folds
        max_n_repeats (int): the maximum number of n_repeats
        random_state (int|np.random.RandomState|None): the random state/seed to use
        aggregation (None|str): 'mor'/'rom' - the aggregation to use if specified

    Returns:
        dict, dict: the dataset specification and the scores
    """
    dataset_spec = generate_dataset_specification(max_p=max_p,
                                                max_n=max_n,
                                                max_n_folds=max_n_folds,
                                                max_n_repeats=max_n_repeats,
                                                random_state=random_state,
                                                aggregation=aggregation)
    dataset = Dataset(**dataset_spec) # pylint: disable=missing-kwoa
    sample = dataset.sample(random_state)
    if dataset.aggregation == 'rom':
        scores = calculate_scores(sample.calculate_figures(), rounding_decimals=rounding_decimals)
    else:
        scores = sample.calculate_scores(rounding_decimals=rounding_decimals)
    scores = scores if score_subset is None else {key: value for key, value in scores.items()
                                                if key in score_subset}
    if fold_score_bounds:
        dataset_spec['fold_score_bounds'] = sample.get_fold_bounds(feasible=feasible_score_bounds)

    del dataset_spec['identifier']
    del dataset_spec['aggregation']

    return dataset_spec, scores

def create_folds_for_dataset(*,
                            p,
                            n,
                            n_folds,
                            n_repeats,
                            folds,
                            folding,
                            fold_score_bounds,
                            aggregation,
                            name,
                            identifier):
    """
    Checks if the parameterization of the dataset is correct and creates
    the folds accordingly.

    Args:
        p (None|int): the number of positives
        n (None|int): the number of negatives
        n_folds (None|int): the number of folds
        n_repeats (None|int): the number of repetitions
        folds (None|list): the list of fold specifications
        folding (str): the folding strategy
        fold_score_bounds (None|dict(str,tuple)): the bound specification for scores in
                                                    the folds
        aggregation (str): 'rom'/'mor - the aggregation strategy
        name (None|str): the name of the dataset to look-up
        identifier (None|str): the identifier

    Returns:
        list(dict): the list of fold specifications

    Raises:
        ValueError: if the dataset specification is inconsistent
    """

    if (p is None and n is not None) or (p is not None and n is None):
        raise ValueError('either specify both p and n or neither of them')

    if p is not None and name is not None:
        raise ValueError('either specify p and n or the name of the dataset')

    if p is not None and folds is not None:
        raise ValueError('either specify (p and n or the name of the dataset) '\
                        'or the list of fold specifications')

    if (n_folds is not None or n_repeats is not None) and folds is not None:
        raise ValueError('n_folds, n_repeats and folding cannot be specified '\
                        'together with the list of fold specifications')

    if name is not None and folds is not None:
        raise ValueError('either specify the name of the dataset or the list '\
                        'of fold specifications')

    if p is None and name is None and folds is None:
        raise ValueError('at least the p,n or the name of the dataset or the '\
                        'list of fold specifications needs to be specified')

    if ((p is not None or name is not None)
        and ((n_folds is None or n_folds == 1) and aggregation is None)):
        # if the dataset is specified by p, n or the dataset name and
        # the number of folds is not specified or it is one, then the
        # aggregation can be anything
        aggregation = 'rom'

    if aggregation not in ('mor', 'rom'):
        raise ValueError(f'aggregation {aggregation} is not supported')

    if folds is None:
        n_folds = n_folds if n_folds is not None else 1
        n_repeats = n_repeats if n_repeats is not None else 1

        if p is None:
            dataset = lookup_dataset(name)
            p = dataset['p']
            n = dataset['n']

        if n_folds > 1 and folding is None and aggregation == 'mor':
            raise ValueError('for mean of ratios aggregation the folding needs '\
                                'to be specified if n_folds > 1')

        if n_folds > 1 and folding is None and fold_score_bounds is not None:
            raise ValueError('for fold score bounds to be set folding needs to '\
                                'be specified if n_folds > 1')

        return _create_folds(p=p,
                                n=n,
                                n_folds=n_folds,
                                n_repeats=n_repeats,
                                folding=folding,
                                score_bounds=fold_score_bounds,
                                identifier=identifier)

    return ([fold | {'score_bounds': {**fold_score_bounds}} for fold in folds]
            if fold_score_bounds is not None else folds)

class Dataset:
    """
    An abstraction for a dataset
    """

    def __init__(self,
                    *,
                    aggregation=None,
                    identifier=None,
                    name=None,
                    p=None,
                    n=None,
                    folds=None,
                    n_folds=None,
                    n_repeats=None,
                    folding=None,
                    score_bounds=None,
                    fold_score_bounds=None):
        """
        Constructor of the dataset. Specify either p and n or a name
        to look up the p and n statistics. The name must be in the supported
        format like 'common_datasets.ADA'. Alternatively, one can pass
        a list of fold specifications.

        Args:
            aggregation (str): 'rom'/'mor - the aggregation strategy
            identifier (None|str): the identifier
            name (None|str): the name of the dataset to look-up
            p (None|int): the number of positives
            n (None|int): the number of negatives
            folds (None|list): the list of fold specifications
            n_folds (None|int): the number of folds
            n_repeats (None|int): the number of repetitions
            folding (str): the folding strategy
            score_bounds (None|dict(str,tuple)): the bound specification for scores
            fold_score_bounds (None|dict(str,tuple)): the bound specification for scores in
                                                        the folds

        Examples:
            ds0 = Dataset(p=5, n=10, aggregation='rom')
            ds1 = Dataset(name='common_datasets.ADA', aggregation='mor')
            ds2 = Dataset(folds=[{p=5, n=10}, {p=2, n=8}], aggregation='rom')

        Raises:
            ValueError: if the dataset specification/parametrization is inconsistent
        """

        # the id of the dataset is set to the name or a random id is generated
        if identifier is None and name is not None:
            self.identifier = name.replace('-', '_')
        elif identifier is not None:
            self.identifier = identifier
        else:
            logger.info('generating a random id for the dataset')
            self.identifier = random_identifier(16)

        self.folds = create_folds_for_dataset(p=p,
                                                n=n,
                                                n_folds=n_folds,
                                                n_repeats=n_repeats,
                                                folds=folds,
                                                folding=folding,
                                                fold_score_bounds=fold_score_bounds,
                                                aggregation=aggregation,
                                                name=name,
                                                identifier=self.identifier)

        self.aggregation = aggregation

        self.score_bounds = score_bounds

        self.linear_programming = None

        self.initialize_folds()

    def to_dict(self, problem_only=False):
        """
        Returns a dict representation

        Args:
            problem_only (bool): whether to return the problem only (True) or add the
                                figures and scores (False)

        Returns:
            dict: the dict representation
        """

        return {'identifier': self.identifier,
                'score_bounds': self.score_bounds,
                'folds': [fold.to_dict(problem_only) for fold in self.folds],
                'aggregation': self.aggregation}

    def __repr__(self):
        """
        Returning a string representation

        Returns:
            str: the string representation
        """
        return str(self.to_dict())

    def has_downstream_bounds(self):
        """
        Checks if the dataset has score bounds specified

        Returns:
            bool: a flag indicating if downstream score bounds are specified
        """
        return any(fold.has_bounds() for fold in self.folds)

    def initialize_folds(self):
        """
        Initialize all folds by turning the specifications into fold objects
        """
        if not isinstance(self.folds[0], Fold):
            self.folds = [Fold(**fold) for fold in self.folds]

    def sample(self, random_state=None):
        """
        Samples the problem, that is, generates random (but physical) tp and tn values

        Args:
            random_state (None|int|np.random.RandomState): the random state to use

        Returns:
            Dataset: the sampled dataset
        """
        random_state = init_random_state(random_state)

        return Dataset(identifier=self.identifier,
                        aggregation=self.aggregation,
                        score_bounds=self.score_bounds,
                        folds=[fold.sample(random_state=random_state).to_dict(problem_only=False)
                                for fold in self.folds])

    def calculate_figures(self):
        """
        Calculate the aggregated raw figures for the dataset

        Returns:
            dict(str,int): the tp, tn, p and n scores
        """
        figures = {'p': 0, 'n': 0}

        for fold in self.folds:
            figures['p'] += fold.p
            figures['n'] += fold.n

        if self.folds[0].figures is not None:
            figures['tp'] = 0
            figures['tn'] = 0
            for fold in self.folds:
                figures['tp'] += fold.figures['tp']
                figures['tn'] += fold.figures['tn']

        return figures

    def calculate_scores(self, score_subset=None, rounding_decimals=None):
        """
        Calculates all scores for the fold

        Args:
            score_subset (None|list): the list of scores to calculate (subset of
                                        'acc', 'sens', 'spec', 'bacc')
            rounding_decimals (None|float): how many digits to round the decimals to

        Returns:
            dict(str,float): the scores
        """
        score_subset = aggregated_scores if score_subset is None else score_subset

        figures = self.calculate_figures()

        if self.aggregation == 'rom':
            scores = calculate_scores_for_lp(figures, score_subset)
        else:
            scores = dict_mean([fold.calculate_scores(score_subset) for fold in self.folds])

        scores = round_scores(scores, rounding_decimals)

        return scores

    def init_lp(self, lp_problem, scores):
        """
        Initializes the linear programming problem for the dataset

        Args:
            lp_problem (pl.LpProblem): a linear programming problem by pulp
            scores (dict(str,float)): the scores intended to match is used to find
                            suitable initial values for the free variables

        Returns:
            pl.LpProblem: the updated linear programming problem
        """

        # initializing the folds
        for fold in self.folds:
            fold.init_lp(lp_problem, scores)

        self.linear_programming = {'tp': pl.lpSum(fold.linear_programming['tp']
                                                    for fold in self.folds),
                                    'tn': pl.lpSum(fold.linear_programming['tn']
                                                    for fold in self.folds),
                                    'p': sum(fold.p for fold in self.folds),
                                    'n': sum(fold.n for fold in self.folds)}

        if self.aggregation == 'rom':
            self.linear_programming = {**self.linear_programming,
                                        **calculate_scores_for_lp({**self.linear_programming})}
        elif self.aggregation == 'mor':
            for key in aggregated_scores:
                norm = 1.0 / len(self.folds)
                self.linear_programming[key] = pl.lpSum(fold.linear_programming[key]
                                                        for fold in self.folds) * norm

        add_bounds(lp_problem,
                    self.linear_programming,
                    self.score_bounds,
                    f'dataset {self.identifier}')

        return lp_problem

    def get_bounds(self, score_subset=None, feasible=True):
        """
        Extracts bounds according to the feasibility flag

        Args:
            score_subset (list|None): the list of scores to return bounds for
            feasibility (bool): if True, the bounds will be feasible, otherwise infeasible
        """
        scores = self.calculate_scores(score_subset)

        return create_bounds(scores, feasible)

    def add_bounds(self, score_bounds):
        """
        Adding bounds to the dataset

        Args:
            score_bounds (dict(str,tuple(float,float))): the score bounds to add

        Returns:
            Dataset: a new dataset object with the score bounds
        """
        return Dataset(identifier=self.identifier,
                        aggregation=self.aggregation,
                        folds=[fold.to_dict(problem_only=False) for fold in self.folds],
                        score_bounds=score_bounds)

    def get_fold_bounds(self, score_subset=None, feasible=True):
        """
        Extracts reasonable bounds from each fold

        Args:
            score_subset (None|list): the scores to extracts bounds for
            feasible (bool): whether the bounds should lead to feasible problems

        Returns:
            list(dict): the list of bounds for each fold
        """
        if not feasible:
            return [fold.get_bounds(score_subset, feasible) for fold in self.folds]

        bounds = dict_minmax([fold.calculate_scores(score_subset) for fold in self.folds])
        for key, value in bounds.items():
            bounds[key] = (value[0]-1e-3, value[1]+1e-3)
        return bounds

    def add_fold_bounds(self, score_bounds):
        """
        Adds bounds to each fold

        Args:
            score_bounds (dict|list): a bound set or a list of bounds

        Returns:
            Dataset: the updated new dataset
        """
        if isinstance(score_bounds, dict):
            score_bounds = [score_bounds] * len(self.folds)

        return Dataset(identifier=self.identifier,
                        aggregation=self.aggregation,
                        score_bounds=self.score_bounds,
                        folds=[fold.to_dict(problem_only=True) | {'score_bounds': s_bounds}
                                for fold, s_bounds in zip(self.folds, score_bounds)])

    def populate(self, lp_problem):
        """
        Populates the object by the elements of the solved/unsolved linear programming
        problem as a figures of tp and tn

        Args:
            lp_problem (pl.LpProblem): the linear programming problem after running the solve
                                        method

        Returns:
            self: the updated object
        """
        return Dataset(identifier=self.identifier,
                        aggregation=self.aggregation,
                        score_bounds=self.score_bounds,
                        folds=[fold.populate(lp_problem).to_dict(problem_only=False)
                                for fold in self.folds])

    def check_bounds(self, numerical_tolerance=NUMERICAL_TOLERANCE):
        """
        Checks if the boundary conditions hold and returns a summary.
        The 'all_bounds' flag indicates the result of bound checks
        for each fold and the dataset.

        Args:
            numerical_tolerance (float): the numerical tolerance

        Returns:
            dict: a summary of the evaluation of the boundary conditions
        """
        scores = self.calculate_scores()
        figures = self.calculate_figures()

        downstream = [fold.check_bounds(numerical_tolerance) for fold in self.folds]
        flag = all(tmp['bounds_flag'] for tmp in downstream)
        check_score_bounds = check_bounds(scores, self.score_bounds, numerical_tolerance)

        all_bounds = (flag
                        and (check_score_bounds is None or check_score_bounds))

        return {'id': self.identifier,
                'figures': figures,
                'scores': scores,
                'score_bounds': self.score_bounds,
                'score_bounds_flag': check_score_bounds,
                'bounds_flag': all_bounds,
                'folds': downstream}

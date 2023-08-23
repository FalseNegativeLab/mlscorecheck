"""
This module implements an abstraction for a dataset
"""

import numpy as np

from ..individual import calculate_scores_for_lp
from ..core import logger
from ..experiments import lookup_dataset

from ._fold import (Fold, random_identifier)
from ._folds import _create_folds_pure
from ._linear_programming import add_bounds, check_bounds

__all__ = ['Dataset']

class Dataset:
    def __init__(self,
                    *,
                    id=None,
                    name=None,
                    p=None,
                    n=None,
                    folds=None,
                    n_folds=None,
                    n_repeats=None,
                    folding='stratified_sklearn',
                    aggregation=None,
                    score_bounds=None,
                    tptn_bounds=None,
                    fold_score_bounds=None,
                    fold_tptn_bounds=None,
                    figures=None,
                    scores=None):

        folds_provided = folds is not None
        pn_provided = p is not None and n is not None
        folds_repeats_provided = n_folds is not None and n_repeats is not None
        name_provided = name is not None

        if not ((folds_provided and not folds_repeats_provided and not name_provided and not pn_provided)\
            or (pn_provided and not folds_provided and not name_provided)\
            or (name_provided and not folds_provided and not pn_provided)):
                raise ValueError('Please specify either folds (without p, n, n_folds, n_repeats, name) '\
                                    'or p and n (without folds, name) '\
                                    'or name (without p, n, folds)')

        if id is None and name is not None:
            self.id = name
        elif id is not None:
            self.id = id
        else:
            logger.info('generating a random id for the dataset')
            self.id = random_identifier(16)

        if pn_provided or name_provided:
            if name_provided:
                logger.info('querying p and n from looking up the dataset')
                tmp = lookup_dataset(name)
                p, n = tmp['p'], tmp['n']
            logger.info('creating a folding based on the specification')
            folds = _create_folds_pure(p, n, n_folds, n_repeats, folding, fold_score_bounds, fold_tptn_bounds, id=self.id.split('.')[-1])

        self.folds = folds

        if aggregation in ('rom', 'mor'):
            self.aggregation = aggregation
        else:
            raise ValueError(f'aggregation {aggregation} is not supported yet')

        self.score_bounds = score_bounds
        self.tptn_bounds = tptn_bounds

        self.figures = figures
        self.scores = scores

        self.linear_programming = None

        self.initialize_folds()

    def to_dict(self, raw_problem=False):
        results = {'id': self.id,
                    'score_bounds': self.score_bounds,
                    'tptn_bounds': self.tptn_bounds,
                    'folds': [fold.to_dict(raw_problem) for fold in self.folds],
                    'aggregation': self.aggregation
                    }

        if raw_problem:
            return results

        return {**results,
                'figures': self.figures,
                'scores': self.scores}

    def initialize_folds(self):
        if not isinstance(self.folds[0], Fold):
            self.folds = [Fold(**fold) for fold in self.folds]

    def sample(self, random_state=None):
        if random_state is None or not isinstance(random_state, np.random.RandomState):
            random_state = np.random.RandomState(random_state)

        for fold in self.folds:
            fold.sample(random_state=random_state)

        self.figures = {'tp': sum(fold.figures['tp'] for fold in self.folds),
                        'tn': sum(fold.figures['tn'] for fold in self.folds),
                        'p': sum(fold.p for fold in self.folds),
                        'n': sum(fold.n for fold in self.folds)}
        return self

    def calculate_scores(self):
        for fold in self.folds:
            fold.calculate_scores()

        if self.aggregation == 'rom':
            self.scores = calculate_scores_for_lp({self})
        else:
            self.scores = {key: np.mean([fold.scores[key] for fold in self.folds]) for key in ['acc', 'sens', 'spec', 'bacc']}

        return self.scores

    def init_lp(self, lp_program):
        for fold in self.folds:
            fold.init_lp(lp_program)

        self.linear_programming = {'tp': sum(fold.linear_programming['tp'] for fold in self.folds),
                                    'tn': sum(fold.linear_programming['tn'] for fold in self.folds),
                                    'p': sum(fold.p for fold in self.folds),
                                    'n': sum(fold.n for fold in self.folds)}

        if self.aggregation == 'rom':
            self.linear_programming = {**self.linear_programming,
                                        **calculate_scores_for_lp({**self.linear_programming})}
        elif self.aggregation == 'mor':
            for key in ['acc', 'sens', 'spec', 'bacc']:
                self.linear_programming[key] = sum(fold.linear_programming[key] for fold in self.folds) * (1.0 / len(self.folds))

        add_bounds(lp_program, self.linear_programming, self.score_bounds)
        add_bounds(lp_program, self.linear_programming, self.tptn_bounds)

        return lp_program

    def populate_with_solution(self, lp_program):
        for fold in self.folds:
            fold.populate_with_solution(lp_program)

        self.figures = {'tp': sum(fold.figures['tp'] for fold in self.folds),
                        'tn': sum(fold.figures['tn'] for fold in self.folds),
                        'p': sum(fold.p for fold in self.folds),
                        'n': sum(fold.n for fold in self.folds)}

        return self

    def check_bounds(self):
        downstream = [fold.check_bounds() for fold in self.folds]
        flag = all(tmp['check_score_bounds'] and tmp['check_tptn_bounds'] for tmp in downstream)

        return {'figures': self.figures,
                'scores': self.scores,
                'score_bounds': self.score_bounds,
                'check_score_bounds': check_bounds(self.scores, self.score_bounds),
                'tptn_bounds': self.tptn_bounds,
                'check_tptn_bounds': check_bounds(self.figures, self.tptn_bounds),
                'downstream_bounds': flag,
                'folds': downstream}


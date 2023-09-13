"""
This module implements some functionalities related to fold structures
"""

import itertools
import copy

import numpy as np

from ..core import init_random_state
from ..experiments import dataset_statistics

from ._fold import Fold
from ._folding_utils import _create_folds

from ._dataset import Dataset

__all__ = ['Folding']

class Folding:
    """
    Abstract representation of a folding
    """
    def __init__(self,
                    n_folds: int=None,
                    n_repeats: int=None,
                    folds: list=None,
                    strategy: str=None):
        """
        Constructor of the folding

        Args:
            n_folds (None|int): the number of folds
            n_repeats (None|int): the number of repeats
            folds (list(dict)): the list of folds
            strategy (str): the folding strategy ('stratified_sklearn')
        """
        if (n_folds is not None) and (folds is not None):
            raise ValueError('specify either n_folds,n_repeats,strategy or folds')
        if (n_folds is None) and (n_repeats is None) and (folds is None):
            raise ValueError('specify either n_folds,strategy or folds')
        if (folds is None) and (strategy is None):
            raise ValueError('specify strategy if folds are not set explicitly')

        self.n_folds = n_folds
        self.n_repeats = n_repeats if n_repeats is not None else 1
        self.folds = folds
        self.strategy = strategy

    def to_dict(self):
        """
        Dictionary representation of the folding

        Returns:
            dict: the representation of the folding
        """
        return {'n_folds': self.n_folds,
                'n_repeats': self.n_repeats,
                'folds': self.folds,
                'strategy': self.strategy}

    def generate_folds(self, dataset: Dataset, aggregation: str):
        """
        Generates fold objects according to the folding

        Args:
            dataset (Dataset): the dataset to generate folds for
            aggregation (str): the type of aggregation ('mor'/'rom')

        Returns:
            list(Fold): the list of fold objects
        """
        if self.folds is not None:
            p = 0
            n = 0
            for fold in self.folds:
                p += fold['p']
                n += fold['n']

            if (p % dataset.p != 0) or (n % dataset.n != 0) or (p // dataset.p != n // dataset.n):
                raise ValueError("The total p and n figures in the folds are not '\
                                    'multiples of the dataset's p and n figures")

            return [Fold(**fold) for fold in self.folds]

        p, n = dataset.p, dataset.n

        if aggregation == 'rom':
            return [Fold(p=p * self.n_repeats,
                        n=n * self.n_repeats)]

        if aggregation == 'mor':
            folds = _create_folds(p=p,
                                    n=n,
                                    n_folds=self.n_folds,
                                    n_repeats=self.n_repeats,
                                    folding=self.strategy,
                                    identifier=dataset.identifier)
            return [Fold(**fold) for fold in folds]

        raise ValueError(f'aggregation mode {aggregation} is not supported')

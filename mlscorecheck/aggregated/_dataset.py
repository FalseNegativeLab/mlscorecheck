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

from ._fold import Fold
from ._folding_utils import _create_folds
from ._linear_programming import add_bounds
from ._utils import check_bounds, aggregated_scores, random_identifier

from ..experiments import dataset_statistics

__all__ = ['Dataset']

class Dataset:
    """
    The abstract representation of a dataset
    """
    def __init__(self,
                    p: int=None,
                    n: int=None,
                    dataset_name: str=None,
                    identifier: str=None):
        """
        Constructor of a dataset

        Args:
            p (None|int): the number of positives
            n (None|int): the number of negatives
            name (None|str): the name of the dataset in the mlscorecheck specification
                            for example, 'common_datasets.ADA'
            identifier (None|str): the identifier of the dataset (randomly generated if
                                    None)
        """
        if (p is None and n is not None) or (p is not None and n is None):
            raise ValueError('specify either p and n or neither of them')
        if (p is None and dataset_name is None):
            raise ValueError('specify either p and n or the name')
        if (p is not None and dataset_name is not None):
            raise ValueError('specify either p and n or the name')

        self.p = p
        self.n = n
        self.dataset_name = dataset_name

        self.resolve_pn()

        if identifier is None:
            self.identifier = (f'{dataset_name}_{random_identifier(3)}'
                                if dataset_name is not None
                                else random_identifier(5))
        else:
            self.identifier = identifier

    def resolve_pn(self):
        """
        Resolves the ``p`` and ``n`` values from the name of the dataset
        """
        if self.p is None:
            dataset = dataset_statistics[self.dataset_name]
            self.p = dataset['p']
            self.n = dataset['n']

    def to_dict(self):
        """
        Dictionary representation of the dataset

        Returns:
            dict: to_dict
        """
        return {'p': self.p if self.dataset_name is None else None,
                'n': self.n if self.dataset_name is None else None,
                'dataset_name': self.dataset_name,
                'identifier': self.identifier}

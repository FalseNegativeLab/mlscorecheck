"""
This module tests the Folding abstraction
"""

import pytest

from mlscorecheck.aggregated import Folding, Dataset, generate_folding, generate_dataset

def test_instantiation():
    """
    Testing the instantiation of a Folding
    """

    with pytest.raises(ValueError):
        Folding(n_folds=5, folds=[])

    with pytest.raises(ValueError):
        Folding()

    with pytest.raises(ValueError):
        Folding(n_folds=5, n_repeats=3)

    folding = Folding(n_folds=5, n_repeats=3, strategy='stratified_sklearn')

    assert folding is not None

def test_to_dict():
    """
    Testing the dictionary representation
    """

    folding = Folding(n_folds=5, n_repeats=3, strategy='stratified_sklearn')

    folding2 = Folding(**folding.to_dict())

    assert folding2.n_folds == 5
    assert folding2.folds is None

def test_generate_folds():
    """
    Testing the generation of folds
    """

    folding = Folding(n_folds=5, n_repeats=3, strategy='stratified_sklearn')
    dataset = Dataset(p=5, n=10)

    assert len(folding.generate_folds(dataset, 'rom')) == 1
    assert len(folding.generate_folds(dataset, 'mor')) == 15

    with pytest.raises(ValueError):
        folding.generate_folds(dataset, 'dummy')

    folding = Folding(folds=[{'p': 5, 'n': 10}, {'p': 10, 'n': 20}])
    assert len(folding.generate_folds(dataset, 'rom')) == 2

    with pytest.raises(ValueError):
        folding.generate_folds(Dataset(p=4, n=21), aggregation='rom')

@pytest.mark.parametrize('random_seed', range(10))
def test_generate_folding(random_seed):
    """
    Testing the folding generation

    Args:
        random_seed (int): the random seed to use
    """

    dataset = generate_dataset(random_state=random_seed)
    folding = generate_folding(dataset, random_state=random_seed)

    assert Folding(**folding) is not None

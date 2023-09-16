"""
This module tests the dataset

The test cases with complicated bound structures are executed
with timeout to prevent hanging.

It is expected, depending on the solver, that some tests times out.

When failure with bounds is tested, the rounding precision is
not tested with 2 decimals, since accidentally with whatever
bounds it is likely to become feasible.
"""

import pytest

from mlscorecheck.aggregated import (Dataset,
                                        generate_dataset)

random_seeds = list(range(20))

def test_dataset_instantiation():
    """
    Testing the instantiation of a dataset
    """

    with pytest.raises(ValueError):
        Dataset(p=5)

    with pytest.raises(ValueError):
        Dataset()

    with pytest.raises(ValueError):
        Dataset(p=5, n=10, dataset_name='common_datasets.ADA')

    dataset = Dataset(dataset_name='common_datasets.ADA')

    assert dataset.p > 0 and dataset.n > 0

    dataset2 = Dataset(**dataset.to_dict())

    assert dataset2.p == dataset.p and dataset2.n == dataset.n

@pytest.mark.parametrize('random_seed', random_seeds)
def test_dataset_generation(random_seed: int):
    """
    Testing the dataset generation

    Args:
        random_seed (int): the random seed to use
    """

    dataset = generate_dataset(random_state=random_seed)

    dataset = Dataset(**dataset)

    assert dataset.p > 0 and dataset.n > 0

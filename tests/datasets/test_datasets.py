"""
This module tests the dataset functionalities
"""

import os
import json

from importlib.resources import files

from mlscorecheck.datasets import (load_json,
                                    dataset_statistics,
                                    load_ml_datasets,
                                    lookup_dataset)

def test_lookup_dataset():
    """
    Testing the lookup dataset functionality
    """
    path = os.path.join('datasets', 'machine_learning', 'common_datasets.json')
    sio = files('mlscorecheck').joinpath(path).read_text()

    data = json.loads(sio)

    for entry in data['datasets']:
        dataset = lookup_dataset('common_datasets.' + entry['name'])

        print(entry, dataset)

        assert entry['p'] == dataset['p']
        assert entry['n'] == dataset['n']

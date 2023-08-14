"""
This module implements some dataset loaders
"""

import os
import json

from importlib.resources import files

__all__ = ['dataset_statistics',
            'load_json',
            'load_ml_datasets',
            'lookup_dataset']

dataset_statistics = {}

def lookup_dataset(dataset):
    return dataset_statistics.get(dataset)

def load_json(directory, file):
    sio = files('mlscorecheck').joinpath(os.path.join('datasets', directory, file)).read_text()

    data = json.loads(sio)

    return data

def load_ml_datasets():
    data = load_json('machine_learning', 'sklearn.json')

    for entry in data['datasets']:
        dataset_statistics['sklearn.' + entry['name']] = {'p': entry['p'], 'n': entry['n']}

    data = load_json('machine_learning', 'common_datasets.json')

    for entry in data['datasets']:
        dataset_statistics['common_datasets.' + entry['name']] = {'p': entry['p'], 'n': entry['n']}

load_ml_datasets()

"""
This module implements some dataset loaders
"""

import os
import json

from importlib.resources import files

__all__ = ['dataset_statistics',
            'load_json',
            'load_ml_datasets',
            'lookup_dataset',
            '_resolve_pn']

dataset_statistics = {}

def _resolve_pn(dataset_conf):
    """
    Resolve the dataset configuration from the integrated statistics

    Args:
        dataset_conf (dict/list(dict)): one or multiple dataset specification(s)
                                with 'dataset' field(s) containing the name of
                                the dataset(s)

    Returns:
        dict: the dataset configuration extended by the 'p' and 'n' figures
    """
    if isinstance(dataset_conf, dict):
        result = {**dataset_conf}
        if result.get('dataset') is not None:
            tmp = lookup_dataset(result['dataset'])
            result['p'] = tmp['p']
            result['n'] = tmp['n']
    elif isinstance(dataset_conf, list):
        result = [_resolve_p_n(dataset) for dataset in dataset_conf]

    return result

def lookup_dataset(dataset):
    if dataset not in dataset_statistics:
        raise ValueError(f"No statistics about dataset {dataset} are available. "\
                            "Didn't you forget to identify like 'common_datasets.ecoli1'?")
    return dataset_statistics[dataset]

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

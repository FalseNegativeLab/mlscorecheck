"""
This module implements the loading of complete experimental settings
"""

import os

__all__ = ['load_drive',
            'load_ehg',
            'load_isic2016',
            'load_isic2017m',
            'load_isic2017sk']

from ..core import load_json

def load_drive() -> dict:
    """
    Loading the drive experiments

    Returns:
        dict: the drive experiments
    """
    prefix = os.path.join('experiments', 'computer_vision', 'drive')
    return {
        'test_fov': load_json(prefix, 'drive_test_fov.json'),
        'test_no_fov': load_json(prefix, 'drive_test_no_fov.json'),
        'train_fov': load_json(prefix, 'drive_train_fov.json'),
        'train_no_fov': load_json(prefix, 'drive_train_no_fov.json')
    }

def load_ehg() -> dict:
    """
    Loading the drive experiments

    Returns:
        dict: the drive experiments
    """
    prefix = os.path.join('experiments', 'machine_learning')
    return load_json(prefix, 'ehg.json')

def load_isic2016() -> dict:
    """
    Loading the ISIC 2016 skin lesion testset

    Returns:
        dict: the testset
    """
    prefix = os.path.join('experiments', 'computer_vision')
    return load_json(prefix, 'isic2016.json')

def load_isic2017m() -> dict:
    """
    Loading the ISIC 2017 skin lesion testset for the binary
    classification task of recognizing melanoma

    Returns:
        dict: the testset
    """
    prefix = os.path.join('experiments', 'computer_vision')
    return load_json(prefix, 'isic2017m.json')

def load_isic2017sk() -> dict:
    """
    Loading the ISIC 2017 skin lesion testset for the binary
    classification task of recognizing seborrheic keratosis

    Returns:
        dict: the testset
    """
    prefix = os.path.join('experiments', 'computer_vision')
    return load_json(prefix, 'isic2017sk.json')

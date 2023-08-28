"""
This module implements the loading of complete experimental settings
"""

import os

__all__ = ['load_drive']

from ..core import load_json

def load_drive():
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

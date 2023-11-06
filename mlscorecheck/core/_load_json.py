"""
This module implements the loading of data
"""

import os
import json

from importlib.resources import files

__all__ = ["load_json"]


def load_json(directory: str, file: str) -> dict:
    """
    Load a JSON file from the package

    Args:
        directory (str): the name of the directory to load from
        file (str): the name of the file to load

    Returns:
        obj: the loaded object
    """
    sio = files("mlscorecheck").joinpath(os.path.join(directory, file)).read_text()

    return json.loads(sio)

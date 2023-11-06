"""
This module tests the dataset functionalities
"""

import os
import json

from importlib.resources import files

import pytest

from mlscorecheck.experiments import lookup_dataset, resolve_pn


def test_lookup_dataset():
    """
    Testing the lookup dataset functionality
    """
    path = os.path.join("experiments", "machine_learning", "common_datasets.json")
    sio = files("mlscorecheck").joinpath(path).read_text()

    data = json.loads(sio)

    for entry in data["datasets"]:
        dataset = lookup_dataset("common_datasets." + entry["name"])

        assert entry["p"] == dataset["p"]
        assert entry["n"] == dataset["n"]


def test_exception():
    """
    Testing the exception throwing
    """

    with pytest.raises(ValueError):
        lookup_dataset("dummy")


def testresolve_pn():
    """
    Testing the resolution of p and n figures
    """

    assert "p" in resolve_pn({"dataset": "common_datasets.ADA"})
    assert (
        len(
            resolve_pn(
                [
                    {"dataset": "common_datasets.ADA"},
                    {"dataset": "common_datasets.ecoli1"},
                ]
            )
        )
        == 2
    )

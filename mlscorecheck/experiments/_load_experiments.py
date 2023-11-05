"""
This module implements the loading of complete experimental settings
"""

import os

__all__ = [
    "get_experiment",
    "load_drive",
    "load_tpehg",
    "load_isic2016",
    "load_isic2017",
    "load_stare",
    "load_chase_db1",
    "load_diaretdb0",
    "load_diaretdb1",
    "load_hrf",
    "load_drishti_gs",
]

from ..core import load_json


experiments = {}


def get_experiment(name):
    """
    Returns the experiment dataset based on the identifier

    Args:
        name (str): the experiment identifier, like 'retina.drive'

    Returns:
        dict: the experiment dataset
    """
    if name in experiments:
        return experiments[name]

    if name == "retina.drive":
        experiments[name] = load_drive()
    elif name == "retina.stare":
        experiments[name] = load_stare()
    elif name == "retina.chase_db1":
        experiments[name] = load_chase_db1()
    elif name == "retina.diaretdb0":
        experiments[name] = load_diaretdb0()
    elif name == "retina.diaretdb1":
        experiments[name] = load_diaretdb1()
    elif name == "retina.hrf":
        experiments[name] = load_hrf()
    elif name == "retina.drishti_gs":
        experiments[name] = load_drishti_gs()
    elif name == "ehg.tpehg":
        experiments[name] = load_tpehg()
    elif name == "skinlesion.isic2016":
        experiments[name] = load_isic2016()
    elif name == "skinlesion.isic2017":
        experiments[name] = load_isic2017()
    else:
        raise ValueError(f"unknown dataset {name}")

    return experiments[name]


def load_chase_db1() -> dict:
    """
    Loading the chase db1 specifications

    Returns:
        dict: the experiment specifications
    """
    prefix = os.path.join("experiments", "retina", "chase_db1")
    return {
        "manual1": load_json(prefix, "manual1.json"),
        "manual2": load_json(prefix, "manual2.json"),
    }


def load_diaretdb0() -> dict:
    """
    Loading the diaretdb0 specifications

    Returns:
        dict: the experiment specifications
    """
    prefix = os.path.join("experiments", "retina", "diaretdb0")
    return load_json(prefix, "diaretdb0.json")


def load_diaretdb1() -> dict:
    """
    Loading the diaretdb1 specifications

    Returns:
        dict: the experiment specifications
    """
    prefix = os.path.join("experiments", "retina", "diaretdb1")
    return load_json(prefix, "diaretdb1.json")


def load_drishti_gs() -> dict:
    """
    Loading the drishti_gs specifications

    Returns:
        dict: the experiment specifications
    """
    prefix = os.path.join("experiments", "retina", "drishti_gs")
    return {
        "train": load_json(prefix, "drishti_gs_train.json")["distributions"],
        "test": load_json(prefix, "drishti_gs_test.json")["distributions"],
    }


def load_hrf() -> dict:
    """
    Loading the hrf specifications

    Returns:
        dict: the experiment specifications
    """
    prefix = os.path.join("experiments", "retina", "hrf")
    return {
        "fov": load_json(prefix, "with_fov.json"),
        "all": load_json(prefix, "without_fov.json"),
    }


def load_stare() -> dict:
    """
    Loading the stare specifications

    Returns:
        dict: the experiment specifications
    """
    prefix = os.path.join("experiments", "retina", "stare")
    return {"ah": load_json(prefix, "ah.json"), "vk": load_json(prefix, "vk.json")}


def load_drive() -> dict:
    """
    Loading the drive experiments

    Returns:
        dict: the drive experiments
    """
    prefix = os.path.join("experiments", "retina", "drive")
    results = {}

    for annotator in [1, 2]:
        for assumption in ["fov", "all"]:
            tmp = {
                "train": load_json(
                    prefix, f"drive_{annotator}_train_{assumption}.json"
                ),
                "test": load_json(prefix, f"drive_{annotator}_test_{assumption}.json"),
            }
            results[(annotator, assumption)] = tmp
    return results


def load_tpehg() -> dict:
    """
    Loading the drive experiments

    Returns:
        dict: the drive experiments
    """
    prefix = os.path.join("experiments", "ehg")
    return load_json(prefix, "tpehg.json")["distribution"]


def load_isic2016() -> dict:
    """
    Loading the ISIC 2016 skin lesion testset

    Returns:
        dict: the testset
    """
    prefix = os.path.join("experiments", "skinlesion", "isic2016")
    return load_json(prefix, "isic2016.json")["distribution"]


def load_isic2017() -> dict:
    """
    Loading the ISIC 2017 skin lesion testset for the binary
    classification task of recognizing melanoma

    Returns:
        dict: the testset
    """
    prefix = os.path.join("experiments", "skinlesion", "isic2017")
    return load_json(prefix, "isic2017.json")["distribution"]

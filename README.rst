.. -*- mode: rst -*-

|GitHub|_ |Codecov|_ |pylint|_ |ReadTheDocs|_ |PythonVersion|_ |PyPi|_ |License|_ |Gitter|_


.. |GitHub| image:: https://github.com/gykovacs/mlscorecheck/workflows/Python%20package/badge.svg?branch=main
.. _GitHub: https://github.com/gykovacs/mlscorecheck/workflows/Python%20package/badge.svg?branch=main

.. |Codecov| image:: https://codecov.io/gh/gykovacs/mlscorecheck/branch/master/graph/badge.svg?token=GQNNasvi4z
.. _Codecov: https://codecov.io/gh/gykovacs/mlscorecheck

.. |pylint| image:: https://img.shields.io/badge/pylint-10.0-brightgreen
.. _pylint: https://img.shields.io/badge/pylint-10.0-brightgreen

.. |ReadTheDocs| image:: https://readthedocs.org/projects/mlscorecheck/badge/?version=latest
.. _ReadTheDocs: https://mlscorecheck.readthedocs.io/en/latest/?badge=latest

.. |PythonVersion| image:: https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10%20%7C%203.11-brightgreen
.. _PythonVersion: https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10%20%7C%203.11-brightgreen

.. |PyPi| image:: https://badge.fury.io/py/mlscorecheck.svg
.. _PyPi: https://badge.fury.io/py/mlscorecheck

.. |License| image:: https://img.shields.io/badge/license-MIT-brightgreen
.. _License: https://img.shields.io/badge/license-MIT-brightgreen

.. |Gitter| image:: https://badges.gitter.im/mlscorecheck.svg
.. _Gitter: https://gitter.im/mlscorecheck?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge

::

mlscorecheck: testing the consistency of machine learning performance scores
****************************************************************************

Introduction
============

Installation
============

The package has only basic requirements when used for consistency checking.

* `numpy`
* `pulp`

.. code-block:: bash

    > pip install numpy pulp

In order to execute the tests, one needs `scikit-learn`, in order to test the computer algebra components, one of `sympy` or `sage` needs to be installed. If one wants to reproduce the score-pair solutions, `sympy` or `sage` needs to be installed.

Installing `sympy` can happen from the usual sources

.. code-block:: bash

    > pip install sympy

Installing sage into a conda environment needs adding the `conda-forge` channel:

.. code-block:: bash

    > conda config --add channels conda-forge
    > conda install sage

Use cases
=========

Specifying a dataset
--------------------

.. code-block:: python

    # one test dataset
    dataset = {"p": 10, "n": 20}
    dataset = {"dataset": "common_datasets.ADA"}

    # one dataset kfold ratio of means
    dataset = {"p": 10, "n": 20, "n_repeats": 5, "n_folds": 3}
    dataset = {"dataset": "common_datasets.ecoli1", "n_repeats": 5, "n_folds": 3}
    dataset = {"fold_configuration": [{"p": 10, "n": 5}, {"p": 5, "n": 20}]

    # multiple dataset ratio of means
    datasets = [{"p": 10, "n": 20},
                {"dataset": "common_datasets.ecoli1"},
                {"fold_configuration": [{"p": 10, "n": 5}]]

    # multiple dataset ratio of means with kfold ratio of means
    datasets = [{"p": 10, "n": 20, "n_repeats": 5, "n_folds": 3},
                {"dataset": "common_datasets.ecoli1"},
                {"fold_configuration": [{"p": 10, "n": 5}, {"p": 5, "n": 20}]]

    # one dataset kfold mean of ratios
    dataset = {"p": 10, "n": 20, "n_repeats": 5, "n_folds": 3, "folding": "stratified-sklearn"}
    dataset = {"dataset": "common_datasets.ecoli1", "n_repeats": 5, "n_folds": 3, "folding": "stratified-sklearn"}
    dataset = {"fold_configuration": [{"p": 10, "n": 5}, {"p": 5, "n": 20}]
    # additionally score_bounds and tptn_bounds for each fold

    # multiple dataset mean of ratios
    datasets = [{"p": 10, "n": 20},
                {"dataset": "common_datasets.ecoli1"},
                {"fold_configuration": [{"p": 10, "n": 5}]}]
    # additionally score_bounds and tptn_bounds for each dataset

    # multiple dataset mean of ratios kfold ratio of means
    datasets = [{"p": 10, "n": 20, "n_repeats": 5, "n_folds": 3},
                {"dataset": "common_datasets.ecoli1"},
                {"fold_configuration": [{"p": 10, "n": 5}, {"p": 5, "n": 20}]}]
    # additionally score_bounds and tptn_bounds for each dataset

    # multiple dataset mean of ratios kfold mean of ratios
    datasets = [{"p": 10, "n": 20, "n_repeats": 5, "n_folds": 3, "folding": "stratified-sklearn"},
                {"dataset": "common_datasets.ecoli1"},
                {"fold_configuration": [{"p": 10, "n": 5}, {"p": 5, "n": 20}]}]
    # additionally score_bounds and tptn_bounds for each dataset and/or fold

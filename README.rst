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

mlscorecheck: testing the consistency of binary classification performance scores
*********************************************************************************

Latest news
===========

* the 0.0.1 version of the package is released
* the paper describing the implemented technique is available as a preprint at:

Citation
========

If you use the package, please consider citing the following paper:

.. code-block:: BibTex

  @article{mlscorecheck,
    author={Gy\"orgy Kov\'acs and Attila Fazekas},
    title={Checking the internal consistency of reported performance scores in binary classification},
    year={2023}
  }

Introduction
============

Binary classification is one of the most basic tasts in machine learning. The evaluation of the performance of binary classification techniques (whether it is original theoretical development or application to a specific field) is driven by performance scores (https://en.wikipedia.org/wiki/Evaluation_of_binary_classifiers). In practice, one of the main challenges of machine learning research is the replication crisis (https://en.wikipedia.org/wiki/Replication_crisis). Despite performance scores provide the basis to estimate the value of a reasearch/study, the reported scores usually suffer from methodological problems, typos and the insufficient description of experimental settings. As the reported scores are usually used to compare and rank techniques, the issues can skew entire fields as pointed out in [RV]_ and [EHG]_.

Most of the performance score are functions of the values of the binary confusion matrix (with four entries: true positives, true negatives, false positives, false negatives). Consequently, the main idea behind the package is that the performance scores cannot take any values independently. For example, the accuracy is the weighted average of the sensitivity and specificity, hence, knowing the statistics of the dataset (number of positives `p` and `n`), these scores can take only specific values simultaneously.

Based on these relations, if at least 3 performance scores are reported, one can construct intervals into which a score given two other score values needs to fall, and can test the internal consistency of the reported figures, given the assumptions on the statistics of the dataset (`p`, `n`) and the evaluation methodology (number of folds and repetitions).

For further documentation, see

* ReadTheDocs full documentation:
* The preprint:

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

In general, there are three inputs to the consistency checking functions:

* the specification of the dataset(s) involved, specified in the form discussed in the previous section;
* the collection of performance scores published: the currently supported performance scores are with their codes in paranthesis (for their specifications see https://en.wikipedia.org/wiki/Evaluation_of_binary_classifiers):

  * accuracy (`acc`)
  * sensitivity (`sens`)
  * specificity (`spec`)
  * positive predictive value (`ppv`)
  * negative predictive value (`npv`)
  * F1-score (`f1`)
  * Fowlkes-Mallows index (`fm`)
* and the estimated numerical uncertainty: the performance scores are usually shared with some finite precision, and are usually rounded/ceiled/floored to `k` digits. Namely, having the accuracy score 0.9489 published, one can suppose that it is rounded, therefore, the numerical uncertainty is 0.00005 (10^(-k)/2). To be more conservative, one can assume that the score was ceiled or follored. In this case the numerical uncertainty becomes 0.0001 (10^(-k)). In both cases, the numerical uncertainty estimates how far the observed score is from the real score.

Specifying datasets
-------------------

Specifying one testset
^^^^^^^^^^^^^^^^^^^^^^

There are multiple ways to specify datasets and entire experiments consisting of multiple datasets evaluated in differing ways of cross-validations.

A simple binary classification test-set consisting of `p` positive samples (usually labelled 1) and `n` negative samples (usually labelled 0) can be specified as

.. code-block:: python

    # one test dataset
    testset = {"p": 10, "n": 20}
    testset = {"name": "common_datasets.ADA"}

Note that in the second case the name of the dataset is specified. ADA is one commonly used dataset in the field of imbalanced learning. In order to prevent the user looking up the details of commonly used datasets, the statistics of many datasets are collected in the package. To see the list of supported datasets and corresponding statistics, issue

.. code-block:: python

    from mlscorecheck.experiments import dataset_statistics
    print(dataset_statistics)

When the name of a dataset is specified, the package looks up the `p` and `n` statistics and substitutes it.

Specifying a dataset with folding
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

There are multiple ways to specify a dataset with some folding structure, either by specifying the parameters of the folding (if it is following a well known strategy, like stratification), or specifying the folds themselves. If `n_repeats` or `n_folds` are not specified, they are considered to be 1. If there is 1 fold, there is no need to specify the folding strategy (`folding`), otherwise the folding strategy needs to be specified. If the `folds` are specified explicitly, there is no need to specify any other parameter (like `p`, `n`, `n_folds`, `n_repeats`). If the `name` of the dataset is specified, `p` and `n` are looked up. For the folds it is possible to specify additional constraints on the `acc`, `sens`, `spec` or `bacc` scores, either by adding the `score_bounds` key to the fold (when `folds` are specified), or setting the `fold_score_bounds` key at the dataset level. Some examples:

.. code-block:: python

    # one dataset kfold with 2 repetitions of stratified folding of 3 folds
    dataset = {"p": 10, "n": 20, "n_repeats": 2, "n_folds": 3, "folding": "stratified_sklearn"}
    dataset = {"dataset": "common_datasets.ecoli1", "n_repeats": 2, "n_folds": 3,
                "folding": "stratified_sklearn"}
    dataset = {"fold_configuration": [{"p": 3, "n": 7}, {"p": 3, "n": 7}, {"p": 4, "n": 6},
                {"p": 3, "n": 7}, {"p": 3, "n": 7}, {"p": 4, "n": 6}]

With score bounds on the folds. Given the score bounds, in the below example, it is a requirement that the accuracy and sensitivity scores both should fall in the range (0.8, 1):

.. code-block:: python

    dataset = {"p": 10, "n": 20, "n_repeats": 2, "n_folds": 3, "folding": "stratified_sklearn",
                "fold_score_bounds": {"acc": (0.8, 1.0), "sens": (0.8, 1.0)}}

    dataset = {"fold_configuration": [{"p": 3, "n": 7,
                                      "score_bounds": {"acc": (0.8, 1.0), "sens": (0.8, 1.0)},
                                      {"p": 3, "n": 7}, {"p": 4, "n": 6}]

The validity of a particular dataset specification can be tested by trying to instantiate a Dataset object:

.. code-block:: python
    from mlscorecheck.aggregated import Dataset
    dataset = {"p": 10, "n": 20, "n_repeats": 2, "n_folds": 3, "folding": "stratified_sklearn"}
    Dataset(**dataset)

If the instantiation is successful, the dataset is specified correctly. Otherwise verbose exceptions will point the user to the inconsistency or lacking parameters.

Checking the consistency of performance scores
----------------------------------------------

Numerous scenarios are supported by the package in which performance scores of binary classification can be produced. In this section we go through them one by one giving some examples of possible use cases.

1 testset with no kfold
^^^^^^^^^^^^^^^^^^^^^^^

This test supports checking the `acc`, `sens`, `spec`, `ppv`, `npv`, `f1`, `fm` scores. The test scenario is having one single test set to which the classifier is applied and the scores are computed from the resulting confusion matrix. For example, given a test image, which is segmented and the scores of the segmentation are calculated and reported.

.. code-block::python
    from mlscorecheck.check import check_1_testset_no_kfold_scores

    result = check_1_testset_no_kfold_scores(
            scores={'acc': 0.62, 'sens': 0.22, 'spec': 0.86, 'f1p': 0.3, 'fm': 0.32},
            eps=1e-2,
            testset={'p': 530, 'n': 902}
        )
    result['inconsistency']
    >> False

    result = check_1_testset_no_kfold_scores(
        scores={'acc': 0.954, 'sens': 0.934, 'spec': 0.985, 'ppv': 0.901},
        eps=1e-3,
        testset={'name': 'common_datasets.ADA'}
    )
    result['inconsistency']
    >> True



1 dataset with kfold ratio-of-means (RoM)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^




1 dataset with kfold mean-of-ratios (MoR)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^




n datasets with k-folds, RoM over datasets and RoM over folds
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^



n datasets with k-folds, MoR over datasets and RoM over folds
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^



n datasets with k-folds, MoR over datasets and MoR over folds
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


Interpreting the results
------------------------

Individual score check
^^^^^^^^^^^^^^^^^^^^^^

Aggregated score check
^^^^^^^^^^^^^^^^^^^^^^

Check bundles
=============

Retinal vessel segmentation
---------------------------


EHG classification
------------------


Contribution
============


References
**********

.. [RV] Kovács, G. and Fazekas, A.: "A new baseline for retinal vessel segmentation: Numerical identification and correction of methodological inconsistencies affecting 100+ papers", Medical Image Analysis, 2022(1), pp. 102300

.. [EHG] Vandewiele, G. and Dehaene, I. and Kovács, G. and Sterckx L. and Janssens, O. and Ongenae, F. and Backere, F. D. and Turck, F. D. and Roelens, K. and Decruyenaere J. and Hoecke, S. V., and Demeester, T.: "Overly optimistic prediction results on imbalanced data: a case study of flaws and benefits when applying over-sampling", Artificial Intelligence in Medicine, 2021(1), pp. 101987

.. -*- mode: rst -*-

|GitHub|_ |Codecov|_ |pylint|_ |ReadTheDocs|_ |PythonVersion|_ |PyPi|_ |License|_ |Gitter|_


.. |GitHub| image:: https://github.com/gykovacs/mlscorecheck/workflows/Python%20package/badge.svg?branch=main
.. _GitHub: https://github.com/gykovacs/mlscorecheck/workflows/Python%20package/badge.svg?branch=main

.. |Codecov| image:: https://codecov.io/gh/FalseNegativeLab/mlscorecheck/graph/badge.svg?token=27TEFPV0P7
.. _Codecov: https://codecov.io/gh/FalseNegativeLab/mlscorecheck

.. |pylint| image:: https://img.shields.io/badge/pylint-10.0-brightgreen
.. _pylint: https://img.shields.io/badge/pylint-10.0-brightgreen

.. |ReadTheDocs| image:: https://readthedocs.org/projects/mlscorecheck/badge/?version=latest
.. _ReadTheDocs: https://mlscorecheck.readthedocs.io/en/latest/?badge=latest

.. |PythonVersion| image:: https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11-brightgreen
.. _PythonVersion: https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10%20%7C%203.11-brightgreen

.. |PyPi| image:: https://badge.fury.io/py/mlscorecheck.svg
.. _PyPi: https://badge.fury.io/py/mlscorecheck

.. |License| image:: https://img.shields.io/badge/license-MIT-brightgreen
.. _License: https://img.shields.io/badge/license-MIT-brightgreen

.. |Gitter| image:: https://badges.gitter.im/mlscorecheck.svg
.. _Gitter: https://app.gitter.im/#/room/!AmkvUevcfkobbwcNWS:gitter.im


mlscorecheck: testing the consistency of machine learning performance scores
****************************************************************************

.. contents::
    :depth: 3

Getting started
===============

The purpose
-----------

Performance scores of a machine learning technique (binary/multiclass classification, regression) are reported on a dataset and look suspicious (exceptionally high scores possibly due to a typo, uncommon evaluation methodology, data leakage in preparation, incorrect use of statistics, etc.). With the tools implemented in the package ``mlscorecheck``, one can test if the reported performance scores are consistent with each other and the assumptions on the experimental setup up.

Testing is as simple as the following example illustrated. Suppose the accuracy, sensitivity and specificity scores are reported for a binary classification testset consisting of p=100 and n=200 samples. All this information is supplied to the suitable test function and the result shows that that inconsistencies were identified: the scores could not be calculated from the confusion matrix of the testset:

.. code-block:: Python

    from mlscorecheck.check.binary import check_1_testset_no_kfold

    result = check_1_testset_no_kfold(testset={'p': 100, 'n': 200},
                                             scores={'acc': 0.9567, 'sens': 0.8545, 'spec': 0.9734},
                                             eps=1e-4)
    result['inconsistency']
    # True

The consistency tests are numerical and **not** statistical: if inconsistencies are identified, it means that either the assumptions on the experimental setup or the reported scores are incorrect.

Latest news
-----------

* the 1.0.1 version of the package is released;
* the paper describing the numerical techniques is available as a preprint at: https://arxiv.org/abs/2310.12527
* the full documentation is available at: https://mlscorecheck.readthedocs.io/en/latest/
* 10 test bundles including retina image processing datasets, preterm delivery prediction from electrohysterograms and skin lesion classification has been added;
* multiclass and regression tests added.

Citation
--------

If you use the package, please consider citing the following paper:

.. code-block:: BibTex

  @article{mlscorecheck,
    author={Attila Fazekas and Gy\"orgy Kov\'acs},
    title={Testing the Consistency of Performance Scores Reported for Binary Classification Problems},
    year={2023}
  }

Installation
------------

The package has only basic requirements when used for consistency testing:

* ``numpy``
* ``pulp``
* ``scikit-learn``

.. code-block:: bash

    pip install numpy pulp

In order to execute the unit tests for the computer algebra components or reproduce the algebraic solutions, either ``sympy`` or ``sage`` needs to be installed. The installation of ``sympy`` can be done in the usual way. To install ``sage`` in a ``conda`` environment, one needs to add the ``conda-forge`` channel first:

.. code-block:: bash

    conda config --add channels conda-forge
    conda install sage

The ``mlscorecheck`` package can be installed from the PyPI repository by issuing

.. code-block:: bash

    pip install mlscorecheck

Alternatively, one can clone the latest version of the package from GitHub and install it into the active virtual environment using the following command:

.. code-block:: bash

    git clone git@github.com:gykovacs/mlscorecheck.git
    cd mlscorecheck
    pip install .


Introduction
============

The evaluation of the performance of machine learning techniques, whether for original theoretical advancements or applications in specific fields, relies heavily on performance scores (https://en.wikipedia.org/wiki/Evaluation_of_binary_classifiers). Although reported performance scores are employed as primary indicators of research value, they often suffer from methodological problems, typos, and insufficient descriptions of experimental settings. These issues contribute to the replication crisis (https://en.wikipedia.org/wiki/Replication_crisis) and ultimately entire fields of research ([RV]_, [EHG]_). Even systematic reviews can suffer from using incomparable performance scores for ranking research papers [RV]_.

In practice, the performance scores cannot take any values independently, the scores reported for the same experiment are constrained by the experimental setup and need to express some internal consistency. For many commonly used experimental setups it is possible to develop numerical techniques to test if the scores could be the outcome of the presumed experiment on the presumed dataset. This package implements such consistency tests for some common experimental setups. We highlight that the developed tests cannot guarantee that the scores are surely calculated by some standards or a presumed evaluation protocol. However, *if the tests fail and inconsistencies are detected, it means that the scores are not calculated by the presumed protocols with certainty*. In this sense, the specificity of the test is 1.0, the inconsistencies being detected are inevitable.

For further information, see

* ReadTheDocs full documentation: https://mlscorecheck.readthedocs.io/en/latest/
* The preprint: https://arxiv.org/abs/2310.12527

Preliminaries
=============

Requirements
------------

In general, there are three inputs to the consistency testing functions:

* **the specification of the experiment**;
* **the collection of available (reported) performance scores**;
* **the estimated numerical uncertainty**: the performance scores are usually shared with some finite precision, being rounded/ceiled/floored to ``k`` decimal places. The numerical uncertainty estimates the maximum difference of the reported score and its true value. For example, having the accuracy score 0.9489 published (4 decimal places), one can suppose that it is rounded, therefore, the numerical uncertainty is 0.00005 (10^(-4)/2). To be more conservative, one can assume that the score was ceiled or floored. In this case, the numerical uncertainty becomes 0.0001 (10^(-4)).

Specification of the experimental setup
---------------------------------------

In this subsection, we illustrate the various ways the experimental setup can be specified.

Specification of one testset or dataset
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

There are multiple ways to specify datasets and entire experiments consisting of multiple datasets evaluated in differing ways of cross-validations.

A simple binary classification testset consisting of ``p`` positive samples (usually labelled 1) and ``n`` negative samples (usually labelled 0) can be specified as

.. code-block:: Python

    testset = {"p": 10, "n": 20}

One can also specify a commonly used dataset by its name and the package will look up the ``p`` and ``n`` counts of the datasets from its internal registry (based on the representations in the ``common-datasets`` package):

.. code-block:: Python

    dataset = {"dataset_name": "common_datasets.ADA"}

To see the list of supported datasets and corresponding counts, issue

.. code-block:: Python

    from mlscorecheck.experiments import dataset_statistics
    print(dataset_statistics)

Specification of a folding
^^^^^^^^^^^^^^^^^^^^^^^^^^

The specification of foldings is needed when the scores are computed in cross-validation scenarios. We distinguish two main cases: in the first case, the number of positive and negative samples in the folds are known, or can be derived from the attributes of the dataset (for example, by stratification); in the second case, the statistics of the folds are not known, but the number of folds and potential repetitions are known.

In the first case, when the folds are known, one can specify them by listing them:

.. code-block:: Python

    folding = {"folds": [{"p": 5, "n": 10},
                            {"p": 4, "n": 10},
                            {"p": 5, "n": 10}]}

This folding can represent the evaluation of a dataset with 14 positive and 30 negative samples in a 3-fold stratified cross-validation scenario.

Knowing that the folding is derived by some standard stratification techniques, one can just specify the parameters of the folding:

.. code-block:: Python

    folding = {"n_folds": 3, "n_repeats": 1, "strategy": "stratified_sklearn"}

In this specification, it is assumed that the samples are distributed into the folds according to the ``sklearn`` stratification implementation.

Finally, if neither the folds nor the folding strategy is known, one can simply specify the folding with its parameters (assuming a repeated k-fold scheme):

.. code-block:: Python

    folding = {"n_folds": 3, "n_repeats": 2}

Note that not all consistency testing functions support the latter case (not knowing the exact structure of the folds).

Specification of an evaluation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A dataset and a folding constitute an *evaluation*, and many of the test functions take evaluations as parameters describing the scenario:

.. code-block:: Python

    evaluation = {"dataset": {"p": 10, "n": 50},
                    "folding": {"n_folds": 5, "n_repeats": 1,
                                "strategy": "stratified_sklearn"}}

A note on the *Score of Means* and *Mean of Scores* aggregations
----------------------------------------------------------------

When it comes to the aggregation of scores (either over multiple folds, multiple datasets or both), there are two approaches in the literature. In the *Mean of Scores* (MoS) scenario, the scores are calculated for each fold/dataset, and the mean of the scores is determined as the score characterizing the entire experiment. In the *Score of Means* (SoM) approach, first the overall confusion matrix is determined, and then the scores are calculated based on these total figures. The advantage of the MoS approach over SoM is that it is possible to estimate the standard deviation of the scores, however, its disadvantage is that the average of non-linear scores might be distorted and some score might become undefined on when the folds are extremely small (typically in the case of small and imbalanced data).

The ``mlscorecheck`` package supports both approaches, however, by design, to increase awareness, different functions are provided for the different approaches, usually indicated by the '_mos' or '_som' suffixes in the function names.

The types of tests
------------------

The consistency tests can be grouped to three classes, and it is the problem and the experimental setup determining which internal implementation is applicable:

- Exhaustive enumeration: primarily applied for binary and multiclass classification, when the scores are calculated from one single confusion matrix. The calculations are speeded up by interval computing techniques. These tests support all 20 performance scores of binary classification.
- Linear programming: when averaging is involved in the calculation of performance scores, due to the non-linearity of most scores, the operation cannot be simplified and the extremely large parameter space prevents exhaustive enumeration. In these scenarios, linear integer programming is exploited. These tests usually support only the accuracy, sensitivity, specificity and balanced accuracy scores.
- Checking the relation of scores: mainly used for regression, when the  domain of the performance scores is continuous, preventing inference from the discrete values.

Binary classification
=====================

Depending on the experimental setup, the consistency tests developed for binary classification problems support a variety of performance scores: when aggregated performance scores (averages on folds or datasets) are reported, only accuracy (``acc``), sensitivity (``sens``), specificity (``spec``) and balanced accuracy (``bacc``) are supported; when cross-validation is not involved in the experimental setup, the list of supported scores reads as follows (with abbreviations in parentheses):

  * accuracy (``acc``),
  * sensitivity (``sens``),
  * specificity (``spec``),
  * positive predictive value (``ppv``),
  * negative predictive value (``npv``),
  * balanced accuracy (``bacc``),
  * f1(-positive) score (``f1``),
  * f1-negative score (``f1n``),
  * f-beta positive (``fbp``),
  * f-beta negative (``fbn``),
  * Fowlkes-Mallows index (``fm``),
  * unified performance measure (``upm``),
  * geometric mean (``gm``),
  * markedness (``mk``),
  * positive likelihood ratio (``lrp``),
  * negative likelihood ratio (``lrn``),
  * Matthews correlation coefficient (``mcc``),
  * bookmaker informedness (``bm``),
  * prevalence threshold (``pt``),
  * diagnostic odds ratio (``dor``),
  * Jaccard index (``ji``),
  * Cohen's kappa (``kappa``)

The tests are designed to detect inconsistencies. If the resulting ``inconsistency`` flag is ``False``, the scores can still be calculated in non-standard ways. However, **if the resulting ``inconsistency`` flag is ``True``, it conclusively indicates that inconsistencies are detected, and the reported scores could not be the outcome of the presumed experiment**.

In the rest of the section, we illustrate some of the test functions, for further details and the full list of supported scenarios, see https://mlscorecheck.readthedocs.io/en/latest/.

1 testset with no k-fold
------------------------

A scenario like this is having one single test set to which classification is applied and the scores are computed from the resulting confusion matrix. For example, given a test image, which is segmented and the scores of the segmentation (as a binary classification of pixels) are calculated and reported.

In the example below, the scores are artificially generated to be consistent, and accordingly, the test did not identify inconsistencies at the ``1e-2`` level of numerical uncertainty.

.. code-block:: Python

    from mlscorecheck.check.binary import check_1_testset_no_kfold

    testset = {'p': 530, 'n': 902}

    scores = {'acc': 0.62, 'sens': 0.22, 'spec': 0.86, 'f1p': 0.3, 'fm': 0.32}

    result = check_1_testset_no_kfold(testset=testset,
                                            scores=scores,
                                            eps=1e-2)
    result['inconsistency']
    # False

The interpretation of the outcome is that given a testset containing 530 positive and 902 negative samples, the reported scores *can* be the outcome of an evaluation. In the ``result`` structure one can find further information about the test. Namely, under the key ``n_valid_tptn_pairs`` one finds the number of ``tp`` and ``tn`` combinations which can lead to the reported performance scores with the given numerical uncertainty.

If one of the scores is altered, like accuracy is changed to 0.92, the configuration becomes infeasible:

.. code-block:: Python

    scores = {'acc': 0.92, 'sens': 0.22, 'spec': 0.86, 'f1p': 0.3, 'fm': 0.32}

    result = check_1_testset_no_kfold(testset=testset,
                                            scores=scores,
                                            eps=1e-2)
    result['inconsistency']
    # True

As the ``inconsistency`` flag shows, here inconsistencies were identified, there are no such ``tp`` and ``tn`` combinations which would end up with the reported scores. Either the assumption on the properties of the dataset, or the scores are incorrect.

1 dataset with k-fold, mean of scores (MoS)/score of means (SoM) aggregation
----------------------------------------------------------------------------

This scenario is the most common in the literature. A classification technique is executed to each fold in a (repeated) k-fold scenario, the scores are calculated for each fold, and the average of the scores is reported with some numerical uncertainty due to rounding/ceiling/flooring. Because of the averaging, this test supports only the linear scores (``acc``, ``sens``, ``spec``, ``bacc``) which usually are among the most commonly reported scores. The test constructs a linear integer program describing the scenario with the true positive and true negative parameters of all folds and checks its feasibility.

In the example below, a consistent set of figures is tested:

.. code-block:: Python

    from mlscorecheck.check.binary import check_1_dataset_known_folds_mos

    dataset = {'p': 126, 'n': 131}
    folding = {'folds': [{'p': 52, 'n': 94}, {'p': 74, 'n': 37}]}

    scores = {'acc': 0.573, 'sens': 0.768, 'bacc': 0.662}

    result = check_1_dataset_known_folds_mos(dataset=dataset,
                                                    folding=folding,
                                                    scores=scores,
                                                    eps=1e-3)
    result['inconsistency']
    # False

As indicated by the output flag, no inconsistencies were identified. The ``result`` dictionary contains some further details of the test. Most notably, under the ``lp_status`` key, one can find the status of the linear programming solver. Additionally, under the ``lp_configuration`` key, one can find the values of all true positive and true negative variables in all folds at the time of the termination of the solver. Furthermore, all scores are calculated for the individual folds and the entire dataset, as well.

If one of the scores is adjusted, for example, sensitivity is changed to 0.568, the configuration becomes infeasible:

.. code-block:: Python

    scores = {'acc': 0.573, 'sens': 0.568, 'bacc': 0.662}
    result = check_1_dataset_known_folds_mos(dataset=dataset,
                                                    folding=folding,
                                                    scores=scores,
                                                    eps=1e-3)
    result['inconsistency']
    # True

Finally, we mention that if there are hints for bounds on the scores in the folds (for example, when the minimum and maximum scores across the folds are reported), one can add these figures to strengthen the test. In the next example, score bounds on accuracy have been added to each fold. This means the test checks if the reported scores can be achieved
with a true positive and true negative configuration with the specified lower and upper bounds for the scores in the individual folds:

.. code-block:: Python

    dataset = {'dataset_name': 'common_datasets.glass_0_1_6_vs_2'}
    folding = {'n_folds': 4, 'n_repeats': 2, 'strategy': 'stratified_sklearn'}

    scores = {'acc': 0.9, 'spec': 0.9, 'sens': 0.6, 'bacc': 0.1, 'f1': 0.95}

    result = check_1_dataset_known_folds_mos(dataset=dataset,
                                                    folding=folding,
                                                    fold_score_bounds={'acc': (0.8, 1.0)},
                                                    scores=scores,
                                                    eps=1e-2,
                                                    numerical_tolerance=1e-6)
    result['inconsistency']
    # True

Note that in this example, although ``f1`` is provided, it is completely ignored as the aggregated tests work only for the four linear scores.

Similar tests are provided for the SoM aggregation as well, for further details see https://mlscorecheck.readthedocs.io/en/latest/.

n testsets without k-fold, SoM/MoS aggregation
----------------------------------------------

In this scenario there are n different testsets, the classifier is evaluated on each testsets, and the scores are aggregated by the SoM aggregation. This scenario is similar to the "1 dataset k-fold SoM" case, except the scores are aggregated over testsets rather than folds. The output of the test is structured similarly as in the "1 dataset k-fold SoM" case. In the following example, a consistent case is tested.

.. code-block:: Python

    from mlscorecheck.check.binary import check_n_testsets_som_no_kfold

    testsets = [{'p': 405, 'n': 223}, {'p': 3, 'n': 422}, {'p': 109, 'n': 404}]
    scores = {'acc': 0.4719, 'npv': 0.6253, 'f1p': 0.3091}

    results = check_n_testsets_som_no_kfold(testsets=testsets,
                                        scores=scores,
                                        eps=0.0001)
    results["inconsistency"]
    # False

If one of the scores is slightly adjusted, for example, ``npv`` changed to 0.6263, the configuration becomes infeasible:

.. code-block:: Python

    scores['npv'] = 0.6263

    results = check_n_testsets_som_no_kfold(testsets=testsets,
                                        scores=scores,
                                        eps=0.0001)
    results["inconsistency"]
    # True

Similar tests are provided for the MoS aggregation as well, for further details see https://mlscorecheck.readthedocs.io/en/latest/.


n datasets with k-fold, MoS over datasets and MoS over folds
------------------------------------------------------------

In this scenario, scores are calculated in the MoS manner for each dataset, and then aggregated again across the datasets. Again, because of the averaging, only the four linear scores (``acc``, ``sens``, ``spec``, ``bacc``) are supported. In the following example a consistent scenario is checked with three datasets and without score bounds specified at any level:

.. code-block:: Python

    from mlscorecheck.check.binary import check_n_datasets_mos_known_folds_mos

    evaluation0 = {'dataset': {'p': 118, 'n': 95},
                    'folding': {'folds': [{'p': 22, 'n': 23}, {'p': 96, 'n': 72}]}}
    evaluation1 = {'dataset': {'p': 781, 'n': 423},
                    'folding': {'folds': [{'p': 300, 'n': 200}, {'p': 481, 'n': 223}]}}
    evaluations = [evaluation0, evaluation1]

    scores = {'acc': 0.61, 'sens': 0.709, 'spec': 0.461, 'bacc': 0.585}

    result = check_n_datasets_mos_known_folds_mos(evaluations=evaluations,
                                                        scores=scores,
                                                        eps=1e-3)
    result['inconsistency']
    # False

Again, the details of the analysis are accessible under the ``lp_status`` and ``lp_configuration`` keys. Adding an adjustment to the scores (turning accuracy to 0.71), the configuration becomes infeasible:

.. code-block:: Python

    scores = {'acc': 0.71, 'sens': 0.709, 'spec': 0.461}

    result = check_n_datasets_mos_known_folds_mos(evaluations=evaluations,
                                                        scores=scores,
                                                        eps=1e-3)
    result['inconsistency']
    # True

If there are hints on the minimum and maximum scores across the datasets, one can add those bounds through the ``dataset_score_bounds`` parameter to strengthen the test.

Similar tests are provided for the SoM aggregation and the mixing of MoS and SoM aggregation, as well, for further details see https://mlscorecheck.readthedocs.io/en/latest/.

Not knowing the k-folding scheme
--------------------------------

In many cases, it is not stated explicitly if stratification was applied or not, only the use of k-fold is phrased in papers. Not knowing the folding structure, the MoS aggregated tests cannot be used. However, if the cardinality of the minority class is not too big (a couple of dozens), then all potential k-fold configurations can be generated, and the MoS tests can be applied to each. If the scores are inconsistent with each, it means that no k-fold could result the scores. There are two functions supporting these exhaustive tests, one for the dataset level, and one for the experiment level.

Given a dataset and knowing that k-fold cross-validation was applied with MoS aggregation, but stratification is not mentioned, the following sample code demonstrates the use of the exhaustive test, with a consistent setup:

.. code-block:: Python

    from mlscorecheck.check.binary import check_1_dataset_unknown_folds_mos

    dataset = {'p': 126, 'n': 131}
    folding = {'n_folds': 2, 'n_repeats': 1}

    scores = {'acc': 0.573, 'sens': 0.768, 'bacc': 0.662}

    result = check_1_dataset_unknown_folds_mos(dataset=dataset,
                                                        folding=folding,
                                                        scores=scores,
                                                        eps=1e-3)
    result['inconsistency']
    # False

If the balanced accuracy score is adjusted to 0.862, the configuration becomes infeasible:

.. code-block:: Python

    scores = {'acc': 0.573, 'sens': 0.768, 'bacc': 0.862}

    result = check_1_dataset_unknown_folds_mos(dataset=dataset,
                                                        folding=folding,
                                                        scores=scores,
                                                        eps=1e-3)
    result['inconsistency']
    # True

In the result of the tests, under the key ``details`` one can find the results for all possible fold combinations.

The following scenario is similar in the sense that MoS aggregation is applied to multiple datasets with unknown folding:

.. code-block:: Python

    from mlscorecheck.check.binary import check_n_datasets_mos_unknown_folds_mos

    evaluation0 = {'dataset': {'p': 13, 'n': 73},
                    'folding': {'n_folds': 4, 'n_repeats': 1}}
    evaluation1 = {'dataset': {'p': 7, 'n': 26},
                    'folding': {'n_folds': 3, 'n_repeats': 1}}
    evaluations = [evaluation0, evaluation1]

    scores = {'acc': 0.357, 'sens': 0.323, 'spec': 0.362, 'bacc': 0.343}

    result = check_n_datasets_mos_unknown_folds_mos(evaluations=evaluations,
                                                            scores=scores,
                                                            eps=1e-3)
    result['inconsistency']
    # False

The setup is consistent. However, if the balanced accuracy is changed to 0.9, the configuration becomes infeasible:

.. code-block:: Python

    scores = {'acc': 0.357, 'sens': 0.323, 'spec': 0.362, 'bacc': 0.9}

    result = check_n_datasets_mos_unknown_folds_mos(evaluations=evaluations,
                                                            scores=scores,
                                                            eps=1e-3)
    result['inconsistency']
    # True

Not knowing the mode of aggregation
-----------------------------------

The biggest challenge with aggregated scores is that the ways of aggregation at the dataset and experiment level are rarely disclosed explicitly. Even in this case the tools presented in the previous section can be used since there are hardly any further ways of meaningful averaging than (MoS on folds, MoS on datasets), (SoM on folds, MoS on datasets), (SoM on folds, SoM on datasets), hence, if a certain set of scores is inconsistent with each of these possibilities, one can safely say that the results do not satisfy the reasonable expectations.

Multiclass classification
=========================

In multiclass classification scenarios single testsets and k-fold cross-validation on a single dataset are supported with both the micro-averaging and macro-averaging aggregation strategies. The list of supported scores depends on the experimental setup, when applicable, all 20 scores listed for binary classification are supported, when the test operates in terms of linear programming, only accuracy, sensitivity, specificity and balanced accuracy are supported.

A note on micro and macro-averaging
-----------------------------------

In a multiclass scenario, the commonly used approach is for measuring the performance of a classification technique is to calculate the micro or macro-averaged scores. In the micro-averaging approach, the confusion matrices of the individual classes are aggregated, and the scores are calculated from the aggregated confusion matrix. In the macro-averaging approach, the scores are calculated for each class, and the average of the scores is reported. The micro-averaging approach is more robust to class imbalance, however, it is not possible to estimate the standard deviation of the scores. The macro-averaging approach is more sensitive to class imbalance, but it is possible to estimate the standard deviation of the scores.

1 testset, no k-fold, micro/macro-averaging
-------------------------------------------

In this scenario, we suppose there is a multiclass classification testset and the class level scores on the testset are aggregated by micro-averaging. The test is based on exhaustive enumeration, so all 20 performance scores are supported. In the first example, we test an artificially generated, consistent scenario:

.. code-block:: Python

    from mlscorecheck.check.multiclass import check_1_testset_no_kfold_micro

    testset = {0: 10, 1: 100, 2: 80}
    scores = {'acc': 0.5158, 'sens': 0.2737, 'spec': 0.6368,
                    'bacc': 0.4553, 'ppv': 0.2737, 'npv': 0.6368}
    results = check_1_testset_no_kfold_micro(testset=testset,
                                            scores=scores,
                                            eps=1e-4)
    results['inconsistency']
    # False

As the test confirms, the setup is consistent. However, if one of the scores is adjusted a little, for example, accuracy is changed to 0.5258, the configuration becomes infeasible:

.. code-block:: Python

    scores['acc'] = 0.5258
    results = check_1_testset_no_kfold_micro(testset=testset,
                                            scores=scores,
                                            eps=1e-4)
    results['inconsistency']
    # True

Similar functionality is provided for macro-averaging, for further details see https://mlscorecheck.readthedocs.io/en/latest/.

1 dataset, known k-folds, SoM/MoS aggregation, micro/macro-averaging
--------------------------------------------------------------------

In this scenario, we assume there is a multiclass classification dataset, which is evaluated in a k-fold cross-validation scenario, the class level scores are calculated by micro-averaging, and the fold level results are aggregated in the score of means fashion. The test is based on exhaustive enumeration, therefore, all 20 performance scores are supported.

In the first example, we test an artificially generated, consistent scenario:

.. code-block:: Python

    from mlscorecheck.check.multiclass import check_1_dataset_known_folds_som_micro

    dataset = {0: 86, 1: 96, 2: 59, 3: 105}
    folding = {'folds': [{0: 43, 1: 48, 2: 30, 3: 52}, {0: 43, 1: 48, 2: 29, 3: 53}]}
    scores =  {'acc': 0.6272, 'sens': 0.2543, 'spec': 0.7514, 'f1p': 0.2543}

    result = check_1_dataset_known_folds_som_micro(dataset=dataset,
                                                        folding=folding,
                                                        scores=scores,
                                                        eps=1e-4)
    result['inconsistency']
    # False

As the test confirms, the scenario is feasible. However, if one of the scores is adjusted a little, for example, sensitivity is changed to 0.2553, the configuration becomes infeasible:

.. code-block:: Python

    scores['sens'] = 0.2553
    result = check_1_dataset_known_folds_som_micro(dataset=dataset,
                                                        folding=folding,
                                                        scores=scores,
                                                        eps=1e-4)
    result['inconsistency']
    # True

Similar functionality is provided for mean of scores aggregation and macro averaging, for further details see https://mlscorecheck.readthedocs.io/en/latest/.

Regression
==========

From the point of view of consistency testing, regression is the hardest problem as the predictions can produce any performance scores. The tests implemented in the package allow testing the relation of the *mean squared error* (``mse``), *root mean squared error* (``rmse``), *mean average error* (``mae``) and *r^2 scores* (``r2``).

1 testset, no k-fold
--------------------

In this scenario, we assume there is a regression testset, and the performance scores are calculated on the testset.

In the first example, we test an artificially generated, consistent scenario:

.. code-block:: Python

    from mlscorecheck.check.regression import check_1_testset_no_kfold

    var = 0.0831619 # the variance of the target values in the testset
    n_samples = 100
    scores =  {'mae': 0.0254, 'r2': 0.9897}

    result = check_1_testset_no_kfold(var=var,
                                        n_samples=n_samples,
                                        scores=scores,
                                        eps=1e-4)
    result['inconsistency']
    # False

As the results show, there is no inconsistency detected. However, if the mae score is adjusted slightly to 0.03, the configuration becomes inconsistent:

.. code-block:: Python

    scores['mae'] = 0.03
    result = check_1_testset_no_kfold(var=var,
                                        n_samples=n_samples,
                                        scores=scores,
                                        eps=1e-4)
    result['inconsistency']
    # True


Test bundles
============

Certain fields have unique, systematic and recurring problems in terms of evaluation methodologies. The aim of this part of the package is to provide bundles of consistency tests for the most typical scenarios of a field.

The list of currently supported problems, datasets and tests are summarized below, for more details see the documentation: https://mlscorecheck.readthedocs.io/en/latest/

The supported scenarios:

* retinal vessel segmentation results on the DRIVE [DRIVE]_ dataset;
* retinal vessel segmentation results on the STARE [STARE]_ dataset;
* retinal vessel segmentation results on the HRF [HRF]_ dataset;
* retinal vessel segmentation results on the CHASE_DB1 [CHASE_DB1]_ dataset;
* retina image labeling using the DIARETDB0 [DIARETDB0]_ dataset;
* retina image labeling and the segmentation of lesions using the DIARETDB1 [DIARETDB1]_ dataset;
* retinal optic disk and optic cup segmentation using the DHRISTI_GS [DRISHTI_GS]_ dataset;
* classification of skin lesion images using the ISIC2016 [ISIC2016]_ dataset;
* classification of skin lesion images using the ISIC2017 [ISIC2017]_ dataset;
* classification of term-preterm delivery in pregnance using EHG signals and the TPEHG [TPEHG]_ dataset.

Contribution
============

We kindly encourage any experts to provide further, field specific dataset and experiment specifications and test bundles to facilitate the reporting of clean and reproducible results in any field related to binary classification!

References
==========

.. [RV] Kovács, G. and Fazekas, A.: "A new baseline for retinal vessel segmentation: Numerical identification and correction of methodological inconsistencies affecting 100+ papers", Medical Image Analysis, 2022(1), pp. 102300

.. [EHG] Vandewiele, G. and Dehaene, I. and Kovács, G. and Sterckx L. and Janssens, O. and Ongenae, F. and Backere, F. D. and Turck, F. D. and Roelens, K. and Decruyenaere J. and Hoecke, S. V., and Demeester, T.: "Overly optimistic prediction results on imbalanced data: a case study of flaws and benefits when applying over-sampling", Artificial Intelligence in Medicine, 2021(1), pp. 101987

.. [DRIVE] Staal, J. and Abramoff, M. D. and Niemeijer, M. and Viergever, M. A. and B. van Ginneken: "Ridge-based vessel segmentation in color images of the retina," in IEEE Transactions on Medical Imaging, vol. 23, no. 4, pp. 501-509, April 2004.

.. [STARE] Hoover, A. D. and Kouznetsova, V. and Goldbaum, M.: "Locating blood vessels in retinal images by piecewise threshold probing of a matched filter response," in IEEE Transactions on Medical Imaging, vol. 19, no. 3, pp. 203-210, March 2000, doi: 10.1109/42.845178.

.. [HRF] Budai A, Bock R, Maier A, Hornegger J, Michelson G.: Robust vessel segmentation in fundus images. Int J Biomed Imaging. 2013;2013:154860. doi: 10.1155/2013/154860. Epub 2013 Dec 12. PMID: 24416040; PMCID: PMC3876700.

.. [CHASE_DB1] Fraz, M. M. et al., "An Ensemble Classification-Based Approach Applied to Retinal Blood Vessel Segmentation," in IEEE Transactions on Biomedical Engineering, vol. 59, no. 9, pp. 2538-2548, Sept. 2012, doi: 10.1109/TBME.2012.2205687.

.. [DIARETDB0] Kauppi, T. and Kalesnykiene, V. and Kämäräinen, J. and Lensu, L. and Sorri, I. and Uusitalo, H. and Kälviäinen, H. and & Pietilä, J. (2007): "DIARETDB 0: Evaluation Database and Methodology for Diabetic Retinopathy Algorithms".

.. [DIARETDB1] Kauppi, Tomi and Kalesnykiene, Valentina and Kamarainen, Joni-Kristian and Lensu, Lasse and Sorri, Iiris and Raninen, A. and Voutilainen, R. and Uusitalo, Hannu and Kälviäinen, Heikki and Pietilä, Juhani. (2007).: "DIARETDB1 diabetic retinopathy database and evaluation protocol". Proc. Medical Image Understanding and Analysis (MIUA). 2007. 10.5244/C.21.15.

.. [DRISHTI_GS] Sivaswamy, J. and Krishnadas, S. R. and Datt Joshi, G. and Jain, M. and Syed Tabish, A. U.: "Drishti-GS: Retinal image dataset for optic nerve head(ONH) segmentation," 2014 IEEE 11th International Symposium on Biomedical Imaging (ISBI), Beijing, China, 2014, pp. 53-56, doi: 10.1109/ISBI.2014.6867807.

.. [ISIC2016] Gutman, D. and Codella, N. C. F. and Celebi, E. and Helba, B. and Marchetti, M. and Mishra, N. and Halpern, A., 2016: "Skin lesion analysis toward melanoma detection: A challenge at the international symposium on biomedical imaging (ISBI) 2016, hosted by the international skin imaging collaboration (ISIC)". doi: 1605.01397

.. [ISIC2017] Codella, N. C. and Gutman, D. and Celebi, M.E. and Helba, B. and Marchetti, M.A. and Dusza, S.W. and Kalloo, A. and Liopyris, K. and Mishra, N. and Kittler, H., et al.: "Skin lesion analysis toward melanoma detection: A challenge at the 2017 international symposium on biomedical imaging (ISBI), hosted by the international skin imaging collaboration (ISIC) Biomedical Imaging (ISBI 2018)", 2018 IEEE 15th International Symposium on, IEEE (2018), pp. 168-172

.. [TPEHG] Fele-Zorz G and Kavsek G and Novak-Antolic Z and Jager F.: "A comparison of various linear and non-linear signal processing techniques to separate uterine EMG records of term and pre-term delivery groups". Med Biol Eng Comput. 2008 Sep;46(9):911-22. doi: 10.1007/s11517-008-0350-y. Epub 2008 Apr 24. PMID: 18437439.

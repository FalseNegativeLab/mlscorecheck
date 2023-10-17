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

mlscorecheck: testing the consistency of binary classification performance scores
*********************************************************************************

The purpose
===========

Performance scores for binary classification are reported on a dataset and look suspicious (exceptionally high scores possibly due to typo, uncommon evaluation methodology, data leakage in preparation, incorrect use of statistics, etc.). With the tools implemented in the package ``mlscorecheck``, one can test if the reported performance scores are consistent with each other and the assumptions on the experimental setup up to the numerical uncertainty due to rounding/truncation/ceiling.

Testing is as simple as the following example shows: the tested scores are inconsistent with a testset of 100 positive and 200 negative entries.

.. code-block:: Python

    from mlscorecheck.check import check_1_testset_no_kfold_scores

    result = check_1_testset_no_kfold_scores(testset={'p': 100, 'n': 200},
                                             scores={'acc': 0.9567, 'sens': 0.8545, 'spec': 0.9734},
                                             eps=1e-4)
    result['inconsistency']
    # True

The consistency tests are numerical and **not** statistical: if inconsistencies are identified, it means that either the assumptions on the experimental setup or the reported scores are incorrect.

Latest news
===========

* the 0.1.1 version of the package is released
* the paper describing the numerical techniques is available as a preprint at:

Citation
========

If you use the package, please consider citing the following paper:

.. code-block:: BibTex

  @article{mlscorecheck,
    author={Gy\"orgy Kov\'acs and Attila Fazekas},
    title={Checking the internal consistency of reported performance scores in binary classification},
    year={2023}
  }

Contents
========

The contents of the repository:

* ``mlscorecheck`` folder: the implementation of the consistency tests;
* ``notebooks/illustration`` folder: the notebooks containing all working sample codes used throughout this README and the ReadTheDocs documentation;
* ``notebooks/utils`` folder: utilities related to generate the algebraic solutions of the score functions, as well as the summary tables used for illustration;
* ``tests`` folder: the unit and functional tests covering each line of code of the package.

Installation
============

The package has only basic requirements when used for consistency testing.

* ``numpy``
* ``pulp``

.. code-block:: bash

    pip install numpy pulp

In order to execute the tests, one also needs ``scikit-learn``, in order to test the computer algebra components or reproduce the algebraic solutions, either ``sympy`` or ``sage`` needs to be installed. The installation of ``sympy`` can be done in the usual way. To install ``sage`` in a ``conda`` environment, one needs to add the ``conda-forge`` channel first:

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

Binary classification is one of the most fundamental tasks in machine learning. The evaluation of the performance of binary classification techniques, whether for original theoretical advancements or applications in specific fields, relies heavily on performance scores (https://en.wikipedia.org/wiki/Evaluation_of_binary_classifiers). Although reported performance scores are employed as primary indicators of research value, they often suffer from methodological problems, typos, and insufficient descriptions of experimental settings. These issues contribute to the replication crisis (https://en.wikipedia.org/wiki/Replication_crisis) and ultimately entire fields of research ([RV]_, [EHG]_). Even systematic reviews can suffer from using incomparable performance scores for ranking research papers [RV]_.

The majority of performance scores are calculated from the binary confusion matrix, or multiple confusion matrices aggregated across folds and/or datasets. For many commonly used experimental setups one can develop numerical techniques to test if there exists any confusion matrix (or matrices), compatible with the experiment and leading to the reported performance scores. This package implements such consistency tests for some common scenarios. We highlight that the developed tests cannot guarantee that the scores are surely calculated by some standards or a presumed evaluation protocol. However, *if the tests fail and inconsistencies are detected, it means that the scores are not calculated by the presumed protocols with certainty*. In this sense, the specificity of the test is 1.0, the inconsistencies being detected are inevitable.

For further information, see

* ReadTheDocs full documentation: https://mlscorecheck.readthedocs.io/en/latest/
* The preprint:

Use cases
=========

In general, there are three inputs to the consistency testing functions:

* **the specification of the experiment**;
* **the collection of available (reported) performance scores**: when aggregated performance scores (averages on folds or datasets) are reported, only accuracy (``acc``), sensitivity (``sens``), specificity (``spec``) and balanced accuracy (``bacc``) are supported; when cross-validation is not involved in the experimental setup, the list of supported scores reads as follows (with abbreviations in parentheses):

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
  * Cohen's kappa (``kappa``);

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

Testing the consistency of performance scores
---------------------------------------------

Numerous experimental setups are supported by the package. In this section we go through them one by one giving some examples of possible use cases.

We emphasize again, that the tests are designed to detect inconsistencies. If the resulting ``inconsistency`` flag is ``False``, the scores can still be calculated in non-standard ways. However, **if the resulting ``inconsistency`` flag is ``True``, it conclusively indicates that inconsistencies are detected, and the reported scores could not be the outcome of the presumed experiment**.

A note on the *Score of Means* and *Mean of Scores* aggregations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When it comes to the aggregation of scores (either over multiple folds, multiple datasets or both), there are two approaches in the literature. In the *Mean of Scores* (MoS) scenario, the scores are calculated for each fold/dataset, and the mean of the scores is determined as the score characterizing the entire experiment. In the *Score of Means* (SoM) approach, first the overall confusion matrix is determined, and then the scores are calculated based on these total figures. The advantage of the MoS approach over SoM is that it is possible to estimate the standard deviation of the scores, however, its disadvantage is that the average of non-linear scores might be distorted and some score might become undefined on when the folds are extremely small (typically in the case of small and imbalanced data).

The two types of tests
^^^^^^^^^^^^^^^^^^^^^^

In the context of a single testset or a Score of Means (SoM) aggregation, which results in one confusion matrix, one can systematically iterate through all potential confusion matrices to assess whether any of them can generate the reported scores within the specified numerical uncertainty. To expedite this process, the test leverages interval arithmetic. The test supports the performance scores ``acc``, ``sens``, ``spec``, ``ppv``, ``npv``, ``bacc``, ``f1``, ``f1n``, ``fbp``, ``fbn``, ``fm``, ``upm``, ``gm``, ``mk``, ``lrp``, ``lrn``, ``mcc``, ``bm``, ``pt``, ``dor``, ``ji``, ``kappa``. Note that when the f-beta positive or f-beta negative scores are used, one also needs to specify the ``beta_positive`` or ``beta_negative`` parameters.

With a MoS type of aggregation, only the averages of scores over folds or datasets are available. In this case, it is feasible to reconstruct fold-level or dataset-level confusion matrices for the linear scores ``acc``, ``sens``, ``spec`` and ``bacc`` using linear integer programming. These tests formulate a linear integer program based on the reported scores and the experimental setup, and check if the program is feasible to produce the reported values within the estimated numerical uncertainties.

1 testset with no k-fold
^^^^^^^^^^^^^^^^^^^^^^^^

A scenario like this is having one single test set to which classification is applied and the scores are computed from the resulting confusion matrix. For example, given a test image, which is segmented and the scores of the segmentation (as a binary classification of pixels) are calculated and reported.

In the example below, the scores are artificially generated to be consistent, and accordingly, the test did not identify inconsistencies at the ``1e-2`` level of numerical uncertainty.

.. code-block:: Python

    from mlscorecheck.check import check_1_testset_no_kfold_scores

    testset = {'p': 530, 'n': 902}

    scores = {'acc': 0.62, 'sens': 0.22, 'spec': 0.86, 'f1p': 0.3, 'fm': 0.32}

    result = check_1_testset_no_kfold_scores(testset=testset,
                                            scores=scores,
                                            eps=1e-2)
    result['inconsistency']
    # False

The interpretation of the outcome is that given a testset containing 530 positive and 902 negative samples, the reported scores *can* be the outcome of an evaluation. In the ``result`` structure one can find further information about the test. Namely, under the key ``n_valid_tptn_pairs`` one finds the number of ``tp`` and ``tn`` combinations which can lead to the reported performance scores with the given numerical uncertainty.

If one of the scores is altered, like accuracy is changed to 0.92, the configuration becomes infeasible:

.. code-block:: Python

    scores = {'acc': 0.92, 'sens': 0.22, 'spec': 0.86, 'f1p': 0.3, 'fm': 0.32}

    result = check_1_testset_no_kfold_scores(testset=testset,
                                            scores=scores,
                                            eps=1e-2)
    result['inconsistency']
    # True

As the ``inconsistency`` flag shows, here inconsistencies were identified, there are no such ``tp`` and ``tn`` combinations which would end up with the reported scores. Either the assumption on the properties of the dataset, or the scores are incorrect.

1 dataset with k-fold, mean-of-scores (MoS)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This scenario is the most common in the literature. A classification technique is executed to each fold in a (repeated) k-fold scenario, the scores are calculated for each fold, and the average of the scores is reported with some numerical uncertainty due to rounding/ceiling/flooring. Because of the averaging, this test supports only the linear scores (``acc``, ``sens``, ``spec``, ``bacc``) which usually are among the most commonly reported scores. The test constructs a linear integer program describing the scenario with the true positive and true negative parameters of all folds and checks its feasibility.

In the example below, a consistent set of figures is tested:

.. code-block:: Python

    from mlscorecheck.check import check_1_dataset_known_folds_mos_scores

    dataset = {'p': 126, 'n': 131}
    folding = {'folds': [{'p': 52, 'n': 94}, {'p': 74, 'n': 37}]}

    scores = {'acc': 0.573, 'sens': 0.768, 'bacc': 0.662}

    result = check_1_dataset_known_folds_mos_scores(dataset=dataset,
                                                    folding=folding,
                                                    scores=scores,
                                                    eps=1e-3)
    result['inconsistency']
    # False

As indicated by the output flag, no inconsistencies were identified. The ``result`` dictionary contains some further details of the test. Most notably, under the ``lp_status`` key, one can find the status of the linear programming solver. Additionally, under the ``lp_configuration`` key, one can find the values of all true positive and true negative variables in all folds at the time of the termination of the solver. Furthermore, all scores are calculated for the individual folds and the entire dataset, as well.

If one of the scores is adjusted, for example, sensitivity is changed to 0.568, the configuration becomes infeasible:

.. code-block:: Python

    scores = {'acc': 0.573, 'sens': 0.568, 'bacc': 0.662}
    result = check_1_dataset_known_folds_mos_scores(dataset=dataset,
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

    result = check_1_dataset_known_folds_mos_scores(dataset=dataset,
                                                    folding=folding,
                                                    fold_score_bounds={'acc': (0.8, 1.0)},
                                                    scores=scores,
                                                    eps=1e-2,
                                                    numerical_tolerance=1e-6)
    result['inconsistency']
    # True

Note that in this example, although ``f1`` is provided, it is completely ignored as the aggregated tests work only for the four linear scores.

1 dataset with kfold ratio-of-means (SoM)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When the scores are calculated in the Score-of-Means (SoM) manner in a k-fold scenario, it means that the total confusion matrix of all folds is calculated first, and then the score formulas are applied to it. The only difference compared to the "1 testset no kfold" scenario is that the number of repetitions of the k-fold scheme multiples the ``p`` and ``n`` statistics of the dataset, but the actual structure of the folds is irrelevant. The result of the analysis is structured similarly to the "1 testset no kfold" case.

For example, testing a consistent scenario:

.. code-block:: Python

    from mlscorecheck.check import check_1_dataset_som_scores

    dataset = {'dataset_name': 'common_datasets.monk-2'}
    folding = {'n_folds': 4, 'n_repeats': 3, 'strategy': 'stratified_sklearn'}

    scores = {'spec': 0.668, 'npv': 0.744, 'ppv': 0.667,
                'bacc': 0.706, 'f1p': 0.703, 'fm': 0.704}

    result = check_1_dataset_som_scores(dataset=dataset,
                                        folding=folding,
                                        scores=scores,
                                        eps=1e-3)
    result['inconsistency']
    # False

If one of the scores is adjusted, for example, negative predictive value is changed to 0.744, the configuration becomes inconsistent:

.. code-block:: Python

    scores = {'spec': 0.668, 'npv': 0.744, 'ppv': 0.667,
            'bacc': 0.706, 'f1p': 0.703, 'fm': 0.704}

    result = check_1_dataset_som_scores(dataset=dataset,
                                        folding=folding,
                                        scores=scores,
                                        eps=1e-3)
    result['inconsistency']
    # True

n datasets with k-folds, SoM over datasets and SoM over folds
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Again, the scenario is similar to the "1 dataset k-fold SoM" scenario, except there is another level of aggregation over datasets, and one single confusion matrix is determined for the entire experiment and the scores are calculated from that. In this scenario a list of evaluations need to be specified. The output of the test is structured similarly as in the "1 dataset k-fold SoM" case, there is a top level ``inconsistency`` flag indicating if inconsistency has been detected. In the following example, a consistent case is prepared with two datasets.

.. code-block:: Python

    from mlscorecheck.check import check_n_datasets_som_kfold_som_scores

    evaluation0 = {'dataset': {'p': 389, 'n': 630},
                    'folding': {'n_folds': 5, 'n_repeats': 2,
                                'strategy': 'stratified_sklearn'}}
    evaluation1 = {'dataset': {'dataset_name': 'common_datasets.saheart'},
                    'folding': {'n_folds': 5, 'n_repeats': 2,
                                'strategy': 'stratified_sklearn'}}
    evaluations = [evaluation0, evaluation1]

    scores = {'acc': 0.631, 'sens': 0.341, 'spec': 0.802, 'f1p': 0.406, 'fm': 0.414}

    result = check_n_datasets_som_kfold_som_scores(scores=scores,
                                                    evaluations=evaluations,
                                                    eps=1e-3)
    result['inconsistency']
    # False

However, if one of the scores is adjusted a little, like accuracy is changed to 0.731, the configuration becomes inconsistent:

.. code-block:: Python

    scores = {'acc': 0.731, 'sens': 0.341, 'spec': 0.802, 'f1p': 0.406, 'fm': 0.414}

    result = check_n_datasets_som_kfold_som_scores(scores=scores,
                                                    evaluations=evaluations,
                                                    eps=1e-3)
    result['inconsistency']
    # True

n datasets with k-folds, MoS over datasets and SoM over folds
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This scenario is about performance scores calculated for each dataset individually by the SoM aggregation in any k-folding strategy, and then the scores are aggregated across the datasets in the MoS manner. Because of the overall averaging, one cannot do inference about the non-linear scores, only the four linear scores are supported (``acc``, ``sens``, ``spec``, ``bacc``), and the scores are checked by linear programming. Similarly as before, the specification of a list of evaluations is needed. In the following example a consistent scenario is tested, with score bounds also specified on the datasets:

.. code-block:: Python

    from mlscorecheck.check import check_n_datasets_mos_kfold_som_scores

    evaluation0 = {'dataset': {'p': 39, 'n': 822},
                    'folding': {'n_folds': 5, 'n_repeats': 3,
                                'strategy': 'stratified_sklearn'}}
    evaluation1 = {'dataset': {'dataset_name': 'common_datasets.winequality-white-3_vs_7'},
                    'folding': {'n_folds': 5, 'n_repeats': 3,
                                'strategy': 'stratified_sklearn'}}
    evaluations = [evaluation0, evaluation1]

    scores = {'acc': 0.312, 'sens': 0.45, 'spec': 0.312, 'bacc': 0.381}

    result = check_n_datasets_mos_kfold_som_scores(evaluations=evaluations,
                                                    dataset_score_bounds={'acc': (0.0, 0.5)},
                                                    eps=1e-4,
                                                    scores=scores)
    result['inconsistency']
    # False

However, if one of the scores is adjusted a little (accuracy changed to 0.412 and the score bounds also changed), the configuration becomes infeasible:

.. code-block:: Python

    scores = {'acc': 0.412, 'sens': 0.45, 'spec': 0.312, 'bacc': 0.381}
    result = check_n_datasets_mos_kfold_som_scores(evaluations=evaluations,
                                                    dataset_score_bounds={'acc': (0.5, 1.0)},
                                                    eps=1e-4,
                                                    scores=scores)
    result['inconsistency']
    # True

The output is structured similarly to the '1 dataset k-folds MoS' case, one can query the status of the solver by the key ``lp_status`` and the actual configuration of the variables by the ``lp_configuration`` key. If there are hints on the minimum and maximum scores across the datasets, one can add those bounds through the ``dataset_score_bounds`` parameter to strengthen the test.

n datasets with k-folds, MoS over datasets and MoS over folds
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In this scenario, scores are calculated in the MoS manner for each dataset, and then aggregated again across the datasets. Again, because of the averaging, only the four linear scores (``acc``, ``sens``, ``spec``, ``bacc``) are supported. In the following example a consistent scenario is checked with three datasets and without score bounds specified at any level:

.. code-block:: Python

    from mlscorecheck.check import check_n_datasets_mos_known_folds_mos_scores

    evaluation0 = {'dataset': {'p': 118, 'n': 95},
                    'folding': {'folds': [{'p': 22, 'n': 23}, {'p': 96, 'n': 72}]}}
    evaluation1 = {'dataset': {'p': 781, 'n': 423},
                    'folding': {'folds': [{'p': 300, 'n': 200}, {'p': 481, 'n': 223}]}}
    evaluations = [evaluation0, evaluation1]

    scores = {'acc': 0.61, 'sens': 0.709, 'spec': 0.461, 'bacc': 0.585}

    result = check_n_datasets_mos_known_folds_mos_scores(evaluations=evaluations,
                                                        scores=scores,
                                                        eps=1e-3)
    result['inconsistency']
    # False

Again, the details of the analysis are accessible under the ``lp_status`` and ``lp_configuration`` keys. Adding an adjustment to the scores (turning accuracy to 0.71), the configuration becomes infeasible:

.. code-block:: Python

    scores = {'acc': 0.71, 'sens': 0.709, 'spec': 0.461}

    result = check_n_datasets_mos_known_folds_mos_scores(evaluations=evaluations,
                                                        scores=scores,
                                                        eps=1e-3)
    result['inconsistency']
    # True

If there are hints on the minimum and maximum scores across the datasets, one can add those bounds through the ``dataset_score_bounds`` parameter to strengthen the test.

Not knowing the mode of aggregation
-----------------------------------

The biggest challenge with aggregated scores is that the ways of aggregation at the dataset and experiment level are rarely disclosed explicitly. Even in this case the tools presented in the previous section can be used since there are hardly any further ways of meaningful averaging than (MoS on folds, MoS on datasets), (SoM on folds, MoS on datasets), (SoM on folds, SoM on datasets), hence, if a certain set of scores is inconsistent with each of these possibilities, one can safely say that the results do not satisfy the reasonable expectations.

Not knowing the k-folding scheme
--------------------------------

In many cases, it is not stated explicitly if stratification was applied or not, only the use of k-fold is phrased in papers. Not knowing the folding structure, the MoS aggregated tests cannot be used. However, if the cardinality of the minority class is not too big (a couple of dozens), then all potential k-fold configurations can be generated, and the MoS tests can be applied to each. If the scores are inconsistent with each, it means that no k-fold could result the scores. There are two functions supporting these exhaustive tests, one for the dataset level, and one for the experiment level.

Given a dataset and knowing that k-fold cross-validation was applied with MoS aggregation, but stratification is not mentioned, the following sample code demonstrates the use of the exhaustive test, with a consistent setup:

.. code-block:: Python

    from mlscorecheck.check import check_1_dataset_unknown_folds_mos_scores

    evaluation = {'dataset': {'p': 126, 'n': 131},
                    'folding': {'n_folds': 2, 'n_repeats': 1}}

    scores = {'acc': 0.573, 'sens': 0.768, 'bacc': 0.662}

    result = check_1_dataset_unknown_folds_mos_scores(evaluation=evaluation,
                                                        scores=scores,
                                                        eps=1e-3)
    result['inconsistency']
    # False

If the balanced accuracy score is adjusted to 0.862, the configuration becomes infeasible:

.. code-block:: Python

    scores = {'acc': 0.573, 'sens': 0.768, 'bacc': 0.862}

    result = check_1_dataset_unknown_folds_mos_scores(dataset=dataset,
                                                        folding=folding,
                                                        scores=scores,
                                                        eps=1e-3)
    result['inconsistency']
    # True

In the result of the tests, under the key ``details`` one can find the results for all possible fold combinations.

The following scenario is similar in the sense that MoS aggregation is applied to multiple datasets with unknown folding:

.. code-block:: Python

    from mlscorecheck.check import check_n_datasets_mos_unknown_folds_mos_scores

    evaluation0 = {'dataset': {'p': 13, 'n': 73},
                    'folding': {'n_folds': 4, 'n_repeats': 1}}
    evaluation1 = {'dataset': {'p': 7, 'n': 26},
                    'folding': {'n_folds': 3, 'n_repeats': 1}}
    evaluations = [evaluation0, evaluation1]

    scores = {'acc': 0.357, 'sens': 0.323, 'spec': 0.362, 'bacc': 0.343}

    result = check_n_datasets_mos_unknown_folds_mos_scores(evaluations=evaluations,
                                                            scores=scores,
                                                            eps=1e-3)
    result['inconsistency']
    # False

The setup is consistent. However, if the balanced accuracy is changed to 0.9, the configuration becomes infeasible:

.. code-block:: Python

    scores = {'acc': 0.357, 'sens': 0.323, 'spec': 0.362, 'bacc': 0.9}

    result = check_n_datasets_mos_unknown_folds_mos_scores(evaluations=evaluations,
                                                            scores=scores,
                                                            eps=1e-3)
    result['inconsistency']
    # True

Test bundles
============

Certain fields have unique, systematic and recurring problems in terms of evaluation methodologies. The aim of this part of the package is to provide bundles of consistency tests for the most typical scenarios of a field.

Experts in various fields are kindly invited to contribute further test bundles to the package.


Retinal vessel segmentation
---------------------------

The segmentation of the vasculature in retinal images [RV]_ gained enormous interest in the recent decades. Typically, the authors have the option to include or exclude certain parts of the images (the pixels outside the Field-of-View), making the reported scores incomparable. (For more details see [RV]_.) To facilitate the meaningful comparison, evaluation and interpretation of reported scores, we provide two functions to check the internal consistency of scores reported for the DRIVE retinal vessel segmentation dataset.

The first function enables the testing of performance scores reported for specific test images. Two tests are executed, one assuming the use of the FoV mask (excluding the pixels outside the FoV) and the other assuming the neglect of the FoV mask (including the pixels outside the FoV). As the following example illustrates, one simply provides the scores and specifies the image (whether it is from the 'test' or 'train' subset and the image identifier) and the consistency results with the two assumptions are returned.

.. code-block:: Python

    from mlscorecheck.bundles import (drive_image, drive_aggregated)

    drive_image(scores={'acc': 0.9478, 'npv': 0.8532, 'f1p': 0.9801, 'ppv': 0.8543},
                eps=1e-4,
                bundle='test',
                identifier='01')
    # {'fov_inconsistency': True, 'no_fov_inconsistency': True}

The interpretation of these results is that the reported scores are inconsistent with any of the reasonable evaluation methodologies.

A similar functionality is provided for the aggregated scores calculated on the DRIVE images, in this case the two assumptions of using the pixels outside the FoV is extended with two assumptions on the way of aggregation.

.. code-block:: Python

    drive_aggregated(scores={'acc': 0.9478, 'sens': 0.8532, 'spec': 0.9801},
                    eps=1e-4,
                    bundle='test')
    # {'mos_fov_inconsistency': True,
    #   'mos_no_fov_inconsistency': True,
    #   'som_fov_inconsistency': True,
    #   'som_no_fov_inconsistency': True}

The results here show that the reported scores could not be the result of any aggregation of any evaluation methodologies.

Preterm delivery prediction from electrohysterogram (EHG) signals
-----------------------------------------------------------------

Electrohysterogram classification for the prediction of preterm delivery in pregnancy became a popular area for the applications of minority oversampling, however, it turned out that there were overly optimistic classification results reported due to systematic data leakage in the data preparation process [EHG]_. In [EHG]_, the implementations were replicated and it was shown that there is a decent gap in terms of performance when the data is prepared properly. However, data leakage changes the statistics of the dataset being cross-validated. Hence, the problematic scores could be identified with the tests implemented in the ``mlscorecheck`` package. In order to facilitate the use of the tools for this purpose, some functionalities have been prepared with the dataset already pre-populated.

For illustration, given a set of scores reported in a real paper, the test below shows that it is not consistent with the dataset:

.. code-block:: Python

    from mlscorecheck.bundles import check_ehg

    scores = {'acc': 0.9552, 'sens': 0.9351, 'spec': 0.9713}

    results = check_ehg(scores=scores, eps=10**(-4), n_folds=10, n_repeats=1)
    results['inconsistency']
    # True

Contribution
============

We kindly encourage any experts to provide further, field specific dataset and experiment specifications and test bundles to facilitate the reporting of clean and reproducible results in any field related to binary classification!

References
**********

.. [RV] Kovács, G. and Fazekas, A.: "A new baseline for retinal vessel segmentation: Numerical identification and correction of methodological inconsistencies affecting 100+ papers", Medical Image Analysis, 2022(1), pp. 102300

.. [EHG] Vandewiele, G. and Dehaene, I. and Kovács, G. and Sterckx L. and Janssens, O. and Ongenae, F. and Backere, F. D. and Turck, F. D. and Roelens, K. and Decruyenaere J. and Hoecke, S. V., and Demeester, T.: "Overly optimistic prediction results on imbalanced data: a case study of flaws and benefits when applying over-sampling", Artificial Intelligence in Medicine, 2021(1), pp. 101987

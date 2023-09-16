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

In a nutshell
=============

One comes across some performance scores of binary classification reported for a dataset and finds them suspicious (typo, unorthodox evaluation methodology, leaky data preparation, etc.). With the tools implemented in the ``mlscorecheck`` package one can test if the reported performance scores are consistent with each other and the assumptions on the experimental setup.

The consistency tests are numerical and **not** statistical: if inconsistencies are identified, it means that either the assumptions on the evaluation protocol or the reported scores are incorrect.

Latest news
===========

* the 0.0.1 version of the package is released
* the paper describing the implemented techniques is available as a preprint at:

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

Binary classification is one of the most fundamental tasks in machine learning. The evaluation of the performance of binary classification techniques (whether it is original theoretical advancement or application to a specific field) is driven by performance scores (https://en.wikipedia.org/wiki/Evaluation_of_binary_classifiers). Despite performance scores provide the basis to estimate the value of reasearch, the reported scores usually suffer from methodological problems, typos and the insufficient description of experimental settings, contributing to the replication crisis (https://en.wikipedia.org/wiki/Replication_crisis) and eventually the derailing of entire fields ([RV]_, [EHG]_). Even systematic reviews can suffer from using incomparable performance scores to rank research papers [RV]_.

Most of the performance score are functions of the values of the binary confusion matrix, having four entries: the number of true positives (``tp``), true negatives (``tn``), false positives (``fp``) and false negatives (``fn``). Since ``tp + fn`` equals the number of positive evaluation samples and ``tn + fp`` equals the number of negative test samples, the matrix has eventually two independent elements. Without the loss of generality, we can pick ``tp`` and ``tn``.

Depending on the experimental setup, one can develop numerical techniques to test if there exists any combination of ``tp`` and ``tn`` which leads to the reported performance scores. This package implements such consistency tests for some common scenarios. We highlight that the developed tests cannot guarantee that the scores are surely calculated by some standards or a presumed evaluation protocol. However, *if the tests fail and inconsistencies are detected, it means that the scores are not calculated by the presumed protocols with certainty*. In this sense, the specificity of the test is 1.0, the inconsistencies being detected are inevitable.

For further information, see

* ReadTheDocs full documentation:
* The preprint:

Installation
============

The package has only basic requirements when used for consistency testing.

* ``numpy``
* ``pulp``

.. code-block:: bash

    > pip install numpy pulp

In order to execute the tests, one also needs ``scikit-learn``, in order to test the computer algebra components or reproduce the algebraic solutions, either ``sympy`` or ``sage`` needs to be installed. The installation of ``sympy`` can be done in the usual way. In order to install ``sage`` into a conda environment one needs adding the ``conda-forge`` channel first:

.. code-block:: bash

    > conda config --add channels conda-forge
    > conda install sage

The ``mlscorecheck`` package itself can be installed from the PyPI repository by issuing

.. code-block:: bash

    > pip install mlscorecheck

Alternatively, the latest (unreleased) version of the package can be cloned from GitHub and installed into the active virtual environment as

.. code-block:: bash

    > git clone git@github.com:gykovacs/mlscorecheck.git
    > cd mlscorecheck
    > pip install .

Use cases
=========

In general, there are three inputs to the consistency testing functions:

* the specification of the dataset(s) involved;
* the collection of available performance scores. When aggregated performance scores (averages on folds or datasets) are reported, only accuracy (``acc``), sensitivity (``sens``), specificity (``spec``) and balanced accuracy (``bacc``) are supported. When cross-validation is not involved in the experimental setup, the list of supported scores reads as follows (with abbreviations in parentheses):

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
  * jaccard index (``ji``),
  * Cohen's kappa (``kappa``),
  * P4 measure (``p4``);

* the estimated numerical uncertainty: the performance scores are usually shared with some finite precision, being rounded/ceiled/floored to ``k`` decimal places. The numerical uncertainty estimates the maximum difference of the reported score and its true value. For example, having the accuracy score 0.9489 published (4 decimal places), one can suppose that it is rounded, therefore, the numerical uncertainty is 0.00005 (10^(-4)/2). To be more conservative, one can assume that the score was ceiled or floored. In this case the numerical uncertainty becomes 0.0001 (10^(-4)).

Specifying the experimental setup
---------------------------------

In this subsection we illustrate the various ways the experimental setup can be specified.

Specifying one testset or dataset
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

There are multiple ways to specify datasets and entire experiments consisting of multiple datasets evaluated in differing ways of cross-validations.

A simple binary classification test-set consisting of ``p`` positive samples (usually labelled 1) and ``n`` negative samples (usually labelled 0) can be specified as

.. code-block:: Python

    testset = {"p": 10, "n": 20}

One can also specify a commonly used dataset by its name and the package will look up the ``p`` and ``n`` statistics of the datasets from its internal registry (based on the representations in the ``common-datasets`` package):

.. code-block:: Python

    dataset = {"dataset_name": "common_datasets.ADA"}

To see the list of supported datasets and corresponding statistics, issue

.. code-block:: Python

    from mlscorecheck.experiments import dataset_statistics
    print(dataset_statistics)

Specifying a folding
^^^^^^^^^^^^^^^^^^^^

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

In this specification it is prescribed that the samples are distributed to the folds according to the ``sklearn`` stratification implementation.

Finally, if nor the folds or the folding strategy is known, one can simply specify the folding with its parameters:

.. code-block:: Python

    folding = {"n_folds": 3, "n_repeats": 2}

Note, that not all consistency testing functions support the latter case (not knowing the folds).

Specifying an evaluation
^^^^^^^^^^^^^^^^^^^^^^^^

A dataset and a folding constitutes an *evaluation*, which can also contain some bounds presumed on the scores calculated in the folds. These bounds can be estimated by the minimum and maximum scores across the folds if they are also reported. Evaluation specifications can be assembled like

.. code-block:: Python

    evaluation = {"dataset": {"p": 10, "n": 50},
                    "folding": {"n_folds": 5, "n_repeats": 1,
                                "strategy": "stratified_sklearn"}}

And with bounds on the scores calculated in the folds:

.. code-block:: Python

    evaluation = {"dataset": {"p": 10, "n": 50},
                    "folding": {"n_folds": 5, "n_repeats": 1,
                                "strategy": "stratified_sklearn"},
                    "fold_score_bounds": {"acc": (0.2, 0.8),
                                            "sens": (0.1, 0.7),
                                            "spec": (0.3, 0.9)}}

Specifying an experiment
^^^^^^^^^^^^^^^^^^^^^^^^

In the context of the package, an experiment is a collection of one or more evaluations, which are presumed to lead to the reported aggregated scores. Experiments are assembled by evaluations, and potential bounds on the scores calculated for the various datasets:

.. code-block:: Python

    experiment = {"evaluations": [{"dataset": {"p": 10, "n": 50},
                                        "folding": {"n_folds": 5, "n_repeats": 1,
                                                    "strategy": "stratified_sklearn"}},
                                    {"dataset": {"p": 30, "n": 30},
                                        "folding": {"n_folds": 5, "n_repeats": 1,
                                                    "strategy": "stratified_sklearn"}]}

With score bounds on the datasets:

.. code-block:: Python

    experiment = {"evaluations": [{"dataset": {"p": 10, "n": 50},
                                    "folding": {"n_folds": 5, "n_repeats": 1,
                                                "strategy": "stratified_sklearn"}},
                                    {"dataset": {"p": 30, "n": 30},
                                    "folding": {"n_folds": 5, "n_repeats": 1,
                                                "strategy": "stratified_sklearn"}],
            "dataset_score_bounds": {"acc": (0.2, 0.8),
                                        "spec": (0.4, 0.95)}}

Checking the consistency of performance scores
----------------------------------------------

Numerous evaluation setups are supported by the package. In this section we go through them one by one giving some examples of possible use cases.

We highlight again, that the tests detect inconsistencies. If the resulting ``inconsistency`` flag is ``False``, the scores can still be calculated in non-standard ways, however, if the ``inconsistency`` flag is ``True``, that is, inconsistencies are detected, then the reported scores are inconsistent with the assumptions with certainty.

A note on the Ratio-of-Means and Mean-of-Ratios aggregations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Most of the performance scores are some sorts of ratios. When it comes to the aggregation of scores (either over multiple folds or multiple datasets or both), there are two approaches in the literature, both having advantages and disadvantages. In the Mean-of-Ratios (MoR) scenario, the scores are calculated for each fold/dataset, and the mean of the scores is determined as the score characterizing the entire experiment. In the Ratio-of-Means (RoM) approach, first the overall confusion matrix (tp, tn, fp, fn) is determined, and then the scores are calculated based on these total figures. The advantage of the MoR approach over RoM is that it is possible to estimate the standard deviation of the scores, however, its disadvantage is that the average of non-linear scores might be distorted.

The two types of tests
^^^^^^^^^^^^^^^^^^^^^^

Having one single testset, or a RoM type of aggregation (leading to one confusion matrix), one can iterate through all potential pairs of ``tp`` and ``tn`` values and check if any of them can produce the reported scores with the given numerical uncertainty. The test is sped up by using interval arithmetic to prevent the evaluation of all possible pairs. This test supports the performance scores ``acc``, ``sens``, ``spec``, ``ppv``, ``npv``, ``bacc``, ``f1``, ``f1n``, ``fbp``, ``fbn``, ``fm``, ``upm``, ``gm``, ``mk``, ``lrp``, ``lrn``, ``mcc``, ``bm``, ``pt``, ``dor``, ``ji``, ``kappa``, ``p4``. Note that when the f-beta positive or f-beta negative scores are used, one also needs to specify the ``beta_positive`` or ``beta_negative`` values.

With a MoR type of aggregation, only the averages of scores over folds or datasets are available. In this case the reconstruction of fold level or dataset level confusion matrices is possible only for the linear scores ``acc``, ``sens``, ``spec`` and ``bacc`` using linear programming. Based on the reported scores and the folding structures, these tests formulate a linear (integer) program of all confusion matrix entries and checks if the program is feasible to result in the reported values with the estimated numerical uncertainties.

1 testset with no kfold
^^^^^^^^^^^^^^^^^^^^^^^

A scenario like this is having one single test set to which classification is applied and the scores are computed from the resulting confusion matrix. For example, given a test image, which is segmented and the scores of the segmentation are calculated and reported.

In the example below, the scores values are generated to be consistent, and accordingly, the test did not identify inconsistencies at the ``1e-2`` level of numerical uncertainty.

.. code-block:: Python

    from mlscorecheck.check import check_1_testset_no_kfold_scores

    >>> result = check_1_testset_no_kfold_scores(
            scores={'acc': 0.62, 'sens': 0.22, 'spec': 0.86, 'f1p': 0.3, 'fm': 0.32},
            eps=1e-2,
            testset={'p': 530, 'n': 902})
    >>> result['inconsistency']
    # False

The interpretation of the outcome is that given a testset containing 530 positive and 902 negative samples, the reported scores *can* be the outcome of an evaluation. In the ``result`` structure one can find further information about the test. Namely, under the key ``n_valid_tptn_pairs`` one finds the number of potential ``tp`` and ``tn`` combinations which can lead to the reported performance scores with the given numerical uncertainty.

In the next example, a consistent set of scores was adjusted randomly to turn them into inconsistent.

.. code-block:: Python

    result = check_1_testset_no_kfold_scores(
        scores={'acc': 0.954, 'sens': 0.934, 'spec': 0.985, 'ppv': 0.901},
        eps=1e-3,
        testset={'name': 'common_datasets.ADA'})
    result['inconsistency']

    # True

As the ``inconsistency`` flag shows, here inconsistencies were identified, there are no such ``tp`` and ``tn`` combinations which would end up with the reported scores. Either the assumption on the properties of the dataset, or the scores are incorrect.

1 dataset with kfold mean-of-ratios (MoR)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This scenario is the most common in the applications and research of machine learning. A classification technique is executed to each fold in a (repeated) k-fold scenario, the scores are calculated for each fold, and the average of the scores is reported with some numerical uncertainty due to rounding/ceiling/flooring. Because of the averaging, this test supports only the linear scores (``acc``, ``sens``, ``spec``, ``bacc``) which usually are among the most commonly reported scores. The test constructs a linear integer program describing the scenario with the ``tp`` and ``tn`` parameters of all folds and checks its feasibility.

In the example below, a consistent set of figures is generated and tested:

.. code-block:: Python

    from mlscorecheck.check import check_1_dataset_kfold_mor_scores

    dataset = {'folds': [{'p': 52, 'n': 94}, {'p': 74, 'n': 37}]}
    scores = {'acc': 0.573, 'sens': 0.768, 'bacc': 0.662}

    result = check_1_dataset_kfold_mor_scores(scores=scores,
                                                eps=1e-3,
                                                dataset=dataset)
    result['inconsistency']

    # False

As one can from the output flag, there are no inconsistencies identified. The ``result`` dict contains some further entries to find further details of the test. Most importantly, under the key ``lp_status`` one can find the status of the linear programming solver, and under the key ``lp_configuration``, one can find the values of all ``tp`` and ``tn`` variables in all folds at the time of the termination of the solver, and additionally, all scores are calculated for the folds and the entire dataset, too:

.. code-block:: bash

    {'id': 'monjhyriadkqzmza',
    'figures': {'p': 126, 'n': 131, 'tp': 93.0, 'tn': 49.0},
    'scores': {'acc': 0.572689127483648,
                'sens': 0.7684511434511435,
                'spec': 0.5556354226566993,
                'bacc': 0.6620432830539213},
    'score_bounds': None,
    'score_bounds_flag': None,
    'bounds_flag': True,
    'folds': [{'identifier': 'pwjncyepgdalgccc',
    'figures': {'tn': 13.0, 'tp': 49.0},
    'scores': {'acc': 0.4246575342465753,
                'sens': 0.9423076923076924,
                'spec': 0.13829787234042554,
                'bacc': 0.5403027823240589},
    'score_bounds': None,
    'score_bounds_flag': None,
    'bounds_flag': True},
    {'identifier': 'nibjsmoafamcpezu',
    'figures': {'tn': 36.0, 'tp': 44.0},
    'scores': {'acc': 0.7207207207207207,
                'sens': 0.5945945945945946,
                'spec': 0.972972972972973,
                'bacc': 0.7837837837837838},
    'score_bounds': None,
    'score_bounds_flag': None,
    'bounds_flag': True}]}

As one can observe, the top level scores match the ones reported to the accuracy of the numerical uncertainty.

As the following example shows, a hand-crafted and insatisfiable set of scores (accuracy must always be between sensitivity ans specificity) leads to the discovery of inconsistency:

.. code-block:: Python

    dataset = {'p': 398,
                'n': 569,
                'n_folds': 4,
                'n_repeats': 2,
                'folding': 'stratified_sklearn'}
    scores = {'acc': 0.91, 'spec': 0.9, 'sens': 0.6}

    result = check_1_dataset_kfold_mor_scores(scores=scores,
                                                eps=1e-2,
                                                dataset=dataset)
    result['inconsistency']

    >> True

Finally, we mention that if there are hints for bounds on the scores in the folds (for example, the minimum and maximum scores across the folds are reported), one can add these figures to strengthen the test. In the next example, the same score bounds on the accuracy have been added to each fold, with the interpretation that beyond matching the overall reported scores, we also require that the accuracy in each fold should be in the range [0.8, 1.0], which becomes unfeasible:

.. code-block:: Python

    dataset = {'name': 'common_datasets.glass_0_1_6_vs_2',
                'n_folds': 4,
                'n_repeats': 2,
                'folding': 'stratified_sklearn',
                'fold_score_bounds': {'acc': (0.8, 1.0)}}
    scores = {'acc': 0.9, 'spec': 0.9, 'sens': 0.6, 'bacc': 0.1, 'f1p': 0.95}

    result = check_1_dataset_kfold_mor_scores(scores=scores,
                                                eps=1e-2,
                                                dataset=dataset)
    result['inconsistency']

    >> True

Note that in this example, although ``f1`` is provided, it is completely ignored as the aggregated tests work only for the four completely linear scores.

1 dataset with kfold ratio-of-means (RoM)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When the scores are calculated in the Ratio-of-Means (RoM) manner in a k-fold scenario, it means that the total confusion matrix (``tp`` and ``tn`` values) of all folds is calculated first, and then some of the formulas (``acc``, ``sens``, ``spec``, ``npv``, ``ppv``, ``f1``, ``fm``) are applied to it. The only difference compared to the "1 testset no kfold" scenario is that the number of repetitions of the k-fold multiples the ``p`` and ``n`` statistics of the dataset, but the actual structure of the folds is irrelevant. The details of the analysis are structured similarly and are accessible under the ``individual_results`` key of the resulting dictionary.

However, having the fold structure enables the testing of the four linear scores (``acc``, ``sens``, ``spec`` and ``bacc``) with potential bounds using linear programming. If any of the four linear scores are supplied and at least one bound is specified, then a linear programming based check similar to the one in the "1 dataset with kfold MoR" scenario is executed. The details of the analysis are structured similarly, and appear under the ``aggregated_results`` key of the resulting dictionary.

In the following example an inconsistent scenario is prepared, and due to the fold level score bounds besides the testing of the individual results, the linear programming based test is also executed.

.. code-block:: Python

    dataset = {'name': 'common_datasets.glass_0_1_6_vs_2',
                'n_folds': 4,
                'n_repeats': 2,
                'folding': 'stratified_sklearn',
                'fold_score_bounds': {'acc': (0.8, 1.0)}}
    scores = {'acc': 0.9, 'npv': 0.9, 'sens': 0.6, 'f1p': 0.95, 'spec': 0.8}

    result = check_1_dataset_kfold_rom_scores(scores=scores,
                                                eps=1e-2,
                                                dataset=dataset)
    result['inconsistency']

    # True

For further details of the analysis the user can access both the details of the individual and the aggregated analysis.

n datasets with k-folds, RoM over datasets and RoM over folds
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This scenario is very similar to the "1 dataset k-fold RoM" scenario, except there is another level of aggregation over datasets, however, still one single confusion matrix is determined first for the entire experiment and the 8 supported scores are calculated from it. In this scenario a list of datasets needs to be specified potentially with folds. If there are score bounds specified for the folds, besides the testing of the individual figures, the aggregated check is also executed. The output of the test is structured similarly as in the "1 dataset k-fold RoM" case, there is a top level ``inconsistency`` flag indicating if inconsistency has been detected. In the following example a consistent case is prepared with two datasets and some mild score bounds.

.. code-block:: Python

    datasets = [{'p': 389,
                    'n': 630,
                    'n_folds': 6,
                    'n_repeats': 3,
                    'folding': 'stratified_sklearn',
                    'fold_score_bounds': {'acc': (0.2, 1)}},
                {'name': 'common_datasets.saheart',
                    'n_folds': 2,
                    'n_repeats': 5,
                    'folding': 'stratified_sklearn'}]
    scores = {'acc': 0.467, 'sens': 0.432, 'spec': 0.488, 'f1p': 0.373}

    result = check_n_datasets_rom_kfold_rom_scores(scores=scores,
                                            datasets=datasets,
                                            eps=1e-3)
    result['inconsistency']

    >> False

The results show that the scores are consistent. Further details are available under the keys ``individual_results`` and ``aggregated_results``. We mention that score bounds at the dataset level could also be specified by adding the ``score_bounds`` key to the dataset specifications.


n datasets with k-folds, MoR over datasets and RoM over folds
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This scenario is about performance scores calculated for each dataset individually by the RoM aggregation in any k-folding strategy, and then the scores are aggregated across the datasets in the MoR manner. Because of the overall averaging, one cannot do inference about the non-linear scores, only the four linear scores are supported (``acc``, ``sens``, ``spec``, ``bacc``), and the scores are checked by linear programming. Similarly as before, the specification of a list of datasets is needed. In the following example an inconsistent scenario is checked:

.. code-block:: Python

    datasets = [{'folds': [{'p': 22, 'n': 90},
                    {'p': 51, 'n': 45},
                    {'p': 78, 'n': 34},
                    {'p': 33, 'n': 89}],
                'fold_score_bounds': {'acc': (0.8, 1.0)},
                'score_bounds': {'acc': (0.85, 1.0)}
                },
                {'name': 'common_datasets.yeast-1-2-8-9_vs_7',
                'n_folds': 8,
                'n_repeats': 4,
                'folding': 'stratified_sklearn',
                'fold_score_bounds': {'acc': (0.8, 1.0)},
                'score_bounds': {'acc': (0.85, 1.0)}
                }]
    scores = {'acc': 0.552, 'sens': 0.555, 'spec': 0.556, 'bacc': 0.555}

    result = check_n_datasets_mor_kfold_rom_scores(datasets=datasets,
                                            eps=1e-3,
                                            scores=scores)
    result['inconsistency']

    # True

The output is structured similarly to the '1 dataset k-folds MoR' case, one can query the status of the solver by the key ``lp_status`` and the actual configuration of the variables by the ``lp_configuration`` key. In this example dataset specification one can observe bounds both at the fold and the dataset level, which must simultaniously hold.

n datasets with k-folds, MoR over datasets and MoR over folds
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The last supported scenario is when scores are calculated in the MoR manner for each dataset, and then aggregated again across the datasets. Again, because of the averaging, only the four linear scores (``acc``, ``sens``, ``spec``, ``bacc``) are supported. Again, the list of datasets involved in the experiment must be specified. In the following example a consistent scenario is checked with three datasets and without score bounds specified at any level:

.. code-block:: Python

    from mlscorecheck.check import check_n_datasets_mor_kfold_mor_scores

    datasets = [{'folds': [{'p': 22, 'n': 23},
                            {'p': 96, 'n': 72}]},
                {'p': 781, 'n': 423, 'n_folds': 1, 'n_repeats': 3},
                {'name': 'common_datasets.glass_0_6_vs_5',
                'n_folds': 6,
                'n_repeats': 1,
                'folding': 'stratified_sklearn'}]
    scores = {'acc': 0.541, 'sens': 0.32, 'spec': 0.728, 'bacc': 0.524}

    result = check_n_datasets_mor_kfold_mor_scores(datasets=datasets,
                                                    scores=scores,
                                                    eps=1e-3)
    result['inconsistency']

    >> False

Being an aggregated test, again, the details of the analysis are accessible under the ``lp_status`` and ``lp_configuration`` keys.

Not knowing the mode of aggregation
-----------------------------------

The biggest challenge with aggregated scores is that the ways of aggregation at the dataset and experiment level are rarely disclosed explicitly. Even in this case the tools presented in the previous section can be used since there are hardly any further ways of meaningful averaging than (MoR on folds, MoR on datasets), (RoM on folds, MoR on datasets), (RoM on folds, RoM on datasets), hence, if a certain set of scores is inconsistent with each of these possibilities, one can safely say that the results do not satisfy the reasonable expectations.

Test bundles
============

Certain fields have unique, systematic and recurring problems in terms of evaluation methodologies. The aim of this part of the package is to provide bundles of consistency tests for the most typical scenarios of a field.

Experts in various fields are kindly invited to contribute further test bundles to the package.


Retinal vessel segmentation
---------------------------

One such field is the segmentation of retinal vessels [RV]_, where the authors have the freedom of either include or exclude certain parts of the images (the pixels outside the Field-of-View) from the evaluation, rendering the reported scores incomparable. In order to facilitate the objective comparison, evaluation and interpretation of reported scores, we provide two functions to check the internal consistency of scores reported for the DRIVE retinal vessel segmentation dataset.

The first function enables the testing of performance scores reported for certain test images, the two tests executed assume the use of the FoV mask (excluding the pixels outside the FoV) and the neglection of the FoV mask (including the pixels outside the FoV). As the following example shows, one simply supplies the scores and specifies the images (whether it is from the 'test' or 'train' subset and the identifier of the image) and gets back if inconsistency is identified with any of the two assumptions.

.. code-block:: Python

    drive_image(scores={'acc': 0.9478, 'npv': 0.8532, 'f1p': 0.9801, 'ppv': 0.8543},
                        eps=1e-4,
                        bundle='test',
                        identifier='01')
    # {'fov_inconsistency': True, 'no_fov_inconsistency': True}

The interpretation of these results is that the reported scores are inconsistent with any of the reasonable evaluation methodolgoies.

A similar functionality is provided for the aggregated scores calculated on the DRIVE images, in this case the two assumptions of using the pixels outside the FoV is extended with two assumptions on the way of aggregation.

.. code-block:: Python

    drive_aggregated(scores={'acc': 0.9478, 'sens': 0.8532, 'spec': 0.9801},
                        eps=1e-4,
                        bundle='test')
    # {'mor_fov_inconsistency': True,
    #   'mor_no_fov_inconsistency': True,
    #   'rom_fov_inconsistency': True,
    #   'rom_no_fov_inconsistency': True}

The results here show that the reported scores could not be the result of any aggregation of any evaluation methodologies.

Contribution
============

We kindly encourage any experts to provide further, field specific dataset and experiment specifications and test bundles to facilitate the reporting of clean and reproducible results in anything related to binary classification!

References
**********

.. [RV] Kovács, G. and Fazekas, A.: "A new baseline for retinal vessel segmentation: Numerical identification and correction of methodological inconsistencies affecting 100+ papers", Medical Image Analysis, 2022(1), pp. 102300

.. [EHG] Vandewiele, G. and Dehaene, I. and Kovács, G. and Sterckx L. and Janssens, O. and Ongenae, F. and Backere, F. D. and Turck, F. D. and Roelens, K. and Decruyenaere J. and Hoecke, S. V., and Demeester, T.: "Overly optimistic prediction results on imbalanced data: a case study of flaws and benefits when applying over-sampling", Artificial Intelligence in Medicine, 2021(1), pp. 101987

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

One comes across some performance scores of binary classification reported for a dataset and finds them suspicious (typo, unorthodox evaluation methodology, etc.). With the tools implemented in the ``mlscorecheck`` package one can test if the scores are consistent with each other and the assumptions on the experimental setup.

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

Binary classification is one of the most basic tasks in machine learning. The evaluation of the performance of binary classification techniques (whether it is original theoretical development or application to a specific field) is driven by performance scores (https://en.wikipedia.org/wiki/Evaluation_of_binary_classifiers). Despite performance scores provide the basis to estimate the value of reasearch, the reported scores usually suffer from methodological problems, typos and the insufficient description of experimental settings, contributing to the replication crisis (https://en.wikipedia.org/wiki/Replication_crisis) and the skewing of entire fields ([RV]_, [EHG]_), as the reported but usually incomparable scores are usually rank approaches and ideas.

Most of the performance score are functions of the values of the binary confusion matrix (with four entries: true positives (tp), true negatives (tn), false positives (fp), false negatives (fn)). Consequently, when multiple performance scores are reported for some experiment, they cannot take any values independently.

Depending on the experimental setup, one can develop techniques to check if the performance scores reported for a dataset are consistent. This package implements such consistency tests for some common scenarios. We highlight that the developed tests cannot guarantee that the scores are surely calculated by some standards. However, if the tests fail and inconsistencies are detected, it means that the scores are not calculated by the presumed protocols with certainty. In this sense, the specificity of the test is 1.0, the inconsistencies being detected are inevitable.

For further documentation, see

* ReadTheDocs full documentation:
* The preprint:

Installation
============

The package has only basic requirements when used for consistency checking.

* ``numpy``
* ``pulp``

.. code-block:: bash

    > pip install numpy pulp

In order to execute the tests, one also needs ``scikit-learn``, in order to test the computer algebra components or reproduce the solutions, either ``sympy`` or ``sage`` needs to be installed. The installation of ``sympy`` can be done in the usual way. In order to install ``sage`` into a conda environment one needs adding the ``conda-forge`` channel first:

.. code-block:: bash

    > conda config --add channels conda-forge
    > conda install sage

Use cases
=========

In general, there are three inputs to the consistency checking functions:

* the specification of the dataset(s) involved;
* the collection of available performance scores. The currently supported scores with their abbreviations in paranthesis are:

  * accuracy (``acc``),
  * sensitivity (``sens``),
  * specificity (``spec``),
  * positive predictive value (``ppv``),
  * negative predictive value (``npv``),
  * F1-score (``f1``),
  * Fowlkes-Mallows index (``fm``);
* the estimated numerical uncertainty: the performance scores are usually shared with some finite precision, being rounded/ceiled/floored to ``k`` decimal places. The numerical uncertainty estimates the maximum difference of the reported score and its true value. For example, having the accuracy score 0.9489 published (4 decimal places), one can suppose that it is rounded, therefore, the numerical uncertainty is 0.00005 (10^(-4)/2). To be more conservative, one can assume that the score was ceiled or floored. In this case the numerical uncertainty becomes 0.0001 (10^(-4)).

Specifying datasets
-------------------

In this subsection we illustrate the various ways datasets can be specified.

Specifying one testset
^^^^^^^^^^^^^^^^^^^^^^

There are multiple ways to specify datasets and entire experiments consisting of multiple datasets evaluated in differing ways of cross-validations.

A simple binary classification test-set consisting of ``p`` positive samples (usually labelled 1) and ``n`` negative samples (usually labelled 0) can be specified as

.. code-block:: Python

    testset = {"p": 10, "n": 20}

One can also specify a commonly used dataset by its name and the package will look up the ``p`` and ``n`` statistics of the datasets from its internal registry:

.. code-block:: Python

    testset = {"name": "common_datasets.ADA"}

To see the list of supported datasets and corresponding statistics, issue

.. code-block:: Python

    from mlscorecheck.experiments import dataset_statistics
    print(dataset_statistics)

Specifying a dataset with folding
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

There are multiple ways to specify a dataset with some folding structure, either by specifying the parameters of the folding (if it is following a well known strategy, like stratification), or specifying the folds themselves. If ``n_repeats`` or ``n_folds`` are not specified, they are considered to be 1. If there is 1 fold, there is no need to specify the folding strategy (``folding``), otherwise the folding strategy needs to be specified. If the ``folds`` are specified explicitly, there is no need to specify any other parameter (like ``p``, ``n``, ``n_folds``, ``n_repeats``). It is possible to specify additional constraints on the ``acc``, ``sens``, ``spec`` or ``bacc`` scores, either by adding the ``score_bounds`` key to the fold (when ``folds`` are specified), or setting the ``fold_score_bounds`` key at the dataset level. For example, multiple ways of specifying datasets with 2 times repeated stratified 3-fold folding structure:

.. code-block:: Python

    # kfold with 2 repetitions of stratified folding of 3 folds
    dataset = {"p": 10, "n": 20, "n_repeats": 2, "n_folds": 3, "folding": "stratified_sklearn"}

    dataset = {"dataset": "common_datasets.ecoli1", "n_repeats": 2, "n_folds": 3, "folding": "stratified_sklearn"}

    dataset = {"folds": [{"p": 3, "n": 7}, {"p": 3, "n": 7}, {"p": 4, "n": 6},
                            {"p": 3, "n": 7}, {"p": 3, "n": 7}, {"p": 4, "n": 6}]

Score bounds can be added in multiple ways:

.. code-block:: Python

    dataset = {"p": 10, "n": 20, "n_repeats": 2, "n_folds": 3, "folding": "stratified_sklearn",
                "fold_score_bounds": {"acc": (0.8, 1.0), "sens": (0.8, 1.0)}}

    dataset = {"folds":
        [{"p": 3, "n": 7, "score_bounds": {"acc": (0.8, 1.0), "sens": (0.8, 1.0)}},
        {"p": 3, "n": 7, "score_bounds": {"acc": (0.8, 1.0), "sens": (0.8, 1.0)}},
        {"p": 4, "n": 6, "score_bounds": {"acc": (0.8, 1.0), "sens": (0.8, 1.0)}}]}

If the specification of a dataset is not consistent or incomplete, the package will guide the user with verbose exceptions on how to fix the specification.

Checking the consistency of performance scores
----------------------------------------------

Numerous experimental protocols are supported by the package in which performance scores of binary classification can be produced. In this section we go through them one by one giving some examples of possible use cases.

We highlight again that the tests detect inconsistencies. If the resulting ``inconsistency`` flag is ``False``, the scores can still be inconsistent, however, if the ``inconsistency`` flag is ``True``, that is, inconsistencies are detected, then the reported scores with the assumptions are inconsistent with certainty.

A note on the Ratio-of-Means and Mean-of-Ratios aggregations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Most of the performance scores are some sorts of ratios. When it comes to the aggregation of scores (either over multiple folds or multiple datasets or both), there are two approaches in the literature, both having advantages and disadvantages. In the Mean-of-Ratios (MoR) scenario, the scores are calculated for each fold/dataset, and the mean of the scores is determined as the score characterizing the entire experiment. In the Ratio-of-Means (RoM) approach, first the overall confusion matrix (tp, tn, fp, fn) is determined, and then the scores are calculated based on these total figures. The advantage of the MoR approach over RoM is that it is possible to estimate the standard deviation of the scores, however, its disadvantage is that the average of non-linear scores might be distorted.

The two types of tests
^^^^^^^^^^^^^^^^^^^^^^

Having one single testset, or a RoM type of aggregation (leading to one confusion matrix) and at least 3 performance scores reported, one can pick two scores and solve the system for the confusion matrix (``tp``, ``tn``) also accounting for the numerical uncertainty of potential rounding/ceiling/flooring, and use the reconstructed confusion matrix to estimate an interval for the third score and check if it is contained in it. This test supports the performance scores ``acc``, ``sens``, ``spec``, ``bacc``, ``npv``, ``ppv``, ``f1p``, ``fm``.

With a MoR type of aggregation, only the averages of scores over folds or datasets are available. In this case the reconstruction of fold level or dataset level confusion matrices is possible only for the linear scores ``acc``, ``sens``, ``spec`` and ``bacc`` using linear programming. Based on the reported scores and the folding structures, these tests formulate a linear (integer) program of all confusion matrix entries and checks if the program is feasible to result in the reported values with the estimated numerical uncertainties.


1 testset with no kfold
^^^^^^^^^^^^^^^^^^^^^^^

This test assumes that at least three of the ``acc``, ``sens``, ``spec``, ``ppv``, ``npv``, ``f1``, ``fm`` scores are reported. A scenario like this is having one single test set to which classification is applied and the scores are computed from the resulting confusion matrix. For example, given a test image, which is segmented and the scores of the segmentation are calculated and reported.

In the example below, the scores are generated to be consistent, and accordingly, the test did not identify inconsistencies at the ``1e-2`` level of numerical uncertainty.

.. code-block:: Python

    from mlscorecheck.check import check_1_testset_no_kfold_scores

    result = check_1_testset_no_kfold_scores(
            scores={'acc': 0.62, 'sens': 0.22, 'spec': 0.86, 'f1p': 0.3, 'fm': 0.32},
            eps=1e-2,
            testset={'p': 530, 'n': 902}
        )
    result['inconsistency']

    # False

The interpretation of the outcome is that given a testset containing 530 positive and 902 negative samples, the reported scores plus/minus ``0.01`` could be the result of a real evaluation. In the ``result`` structure one can find further information about the test. Namely, each pair of scores is used to estimate the range of each other, and under the keys ``tests_succeeded`` and ``tests_failed`` one can find the list of tests which passed and failed. For example, in this particular case, no test has failed. The first entry (``result['tests_succeeded'][0]``) of the succeeded list reads as

.. code-block:: JSON

    {'details': [{'score_0': 'acc',
                'score_0_interval': (0.6099979999999999, 0.6300020000000001),
                'score_1': 'sens',
                'score_1_interval': (0.209998, 0.230002),
                'target_score': 'spec',
                'target_interval': (0.8499979999999999, 0.870002),
                'solution': {'tp': (111.29894, 121.90106),
                            'tn': (751.6160759999999, 790.8639240000001),
                            'tp_formula': 'p*sens',
                            'tn_formula': 'acc*n + acc*p - p*sens'},
                'inconsistency': False,
                'explanation': 'the target score interval ((0.8499979999999999, 0.870002)) and the reconstructed intervals ((0.8332772461197339, 0.8767892727272728)) do intersect',
                'target_interval_reconstructed': (0.8332772461197339, 0.8767892727272728)}],
    'edge_scores': [],
    'underdetermined': False,
    'inconsistency': False}

From the output structure one can read that the accuracy and sensitivity scores are used to reconstruct the interval for specificity (``target_interval_reconstructed``) using the formulas for ``tp`` and ``tn`` under the ``solution`` key. Then, comparing the reconstructed interval with the actual known interval for specificity, one can conclude that they do intersect, hence, the accuracy, sensitivity and specificity scores are not inconsistent.

In the next example, a consistent set of scores was adjusted randomly to turn them into inconsistent.

.. code-block:: Python

    result = check_1_testset_no_kfold_scores(
        scores={'acc': 0.954, 'sens': 0.934, 'spec': 0.985, 'ppv': 0.901},
        eps=1e-3,
        testset={'name': 'common_datasets.ADA'}
    )
    result['inconsistency']

    # True

As the ``inconsistency`` flag shows, here inconsistencies were identified. Looking into the details of the first failed test (``result['tests_failed'][0]``) one can see that

.. code-block:: JSON

    {'details': [{'score_0': 'acc',
                'score_0_interval': (0.9529979999999999, 0.955002),
                'score_1': 'sens',
                'score_1_interval': (0.932998, 0.9350020000000001),
                'target_score': 'spec',
                'target_interval': (0.9839979999999999, 0.986002),
                'solution': {'tp': (960.054942, 962.1170580000002),
                            'tn': (2989.965647999999, 3000.3383520000007),
                            'tp_formula': 'p*sens',
                            'tn_formula': 'acc*n + acc*p - p*sens'},
                'inconsistency': True,
                'explanation': 'the target score interval ((0.9839979999999999, 0.986002)) and the reconstructed intervals ((0.9589370262989092, 0.9622637434252729)) do not intersect',
                'target_interval_reconstructed': (0.9589370262989092, 0.9622637434252729)}],
    'edge_scores': [],
    'underdetermined': False,
    'inconsistency': True}

The interpretation of the output is that given the accuracy and sensitivity scores (and the ``p`` and ``n`` statistics of the dataset), the specificity must fall into the interval ``target_interval_reconstructed``, however, as one can observe the supplied specificity score, it does not, which indicates an inconsistency among the scores.

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

.. code-block:: JSON

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

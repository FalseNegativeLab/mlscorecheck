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

When it comes to the aggregation of scores (either over multiple folds, multiple datasets or both), there are two approaches in the literature. In the *Mean of Scores* (MoS) scenario, the scores are calculated for each fold/dataset, and the mean of the scores is determined as the score characterizing the entire experiment. In the *Score of Means* (SoM) approach, first the overall micro-figures (e.g. the overall confusion matrix in classification, the overall squared error in regression) are determined, and then the scores are calculated based on these total figures. The advantage of the MoS approach over SoM is that it is possible to estimate the standard deviation of the scores, however, its disadvantage is that the average of non-linear scores might be distorted and some score might become undefined on when the folds are extremely small (typically in the case of small and imbalanced data).

The ``mlscorecheck`` package supports both approaches, however, by design, to increase awareness, different functions are provided for the different approaches, usually indicated by the '_mos' or '_som' suffixes in the function names.

The types of tests
------------------

The consistency tests can be grouped to three classes, and it is the problem and the experimental setup determining which internal implementation is applicable:

- Exhaustive enumeration: primarily applied for binary and multiclass classification, when the scores are calculated from one single confusion matrix. The calculations are speeded up by interval computing techniques. These tests support all 20 performance scores of binary classification.
- Linear programming: when averaging is involved in the calculation of performance scores, due to the non-linearity of most scores, the operation cannot be simplified and the extremely large parameter space prevents exhaustive enumeration. In these scenarios, linear integer programming is exploited. These tests usually support only the accuracy, sensitivity, specificity and balanced accuracy scores.
- Checking the relation of scores: mainly used for regression, when the  domain of the performance scores is continuous, preventing inference from the discrete values.

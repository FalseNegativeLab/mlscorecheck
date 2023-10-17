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

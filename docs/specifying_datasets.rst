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

There are multiple ways to specify a dataset with some folding structure, either by specifying the parameters of the folding (if it is following a well known strategy, like stratification), or specifying the folds themselves. If ``n_repeats`` or ``n_folds`` are not specified, they are considered to be 1. If there is 1 fold, there is no need to specify the folding strategy (``folding``), otherwise the folding strategy needs to be specified. If the ``folds`` are specified explicitly, there is no need to specify any other parameter (like ``p``, ``n``, ``n_folds``, ``n_repeats``). It is possible to specify additional constraints on the ``acc``, ``sens``, ``spec`` or ``bacc`` scores, either by adding the ``score_bounds`` key to the fold (when ``folds`` are specified), or setting the ``fold_score_bounds`` key at the dataset level. In the example below we specify 3 datasets, each with 2 times repeated stratified 3-fold folding structure (note that when the folds are explicitly listed, the repetitions are not grouped):

.. code-block:: Python

    dataset = {"p": 10,
                "n": 20,
                "n_repeats": 2,
                "n_folds": 3,
                "folding": "stratified_sklearn"}

    dataset = {"dataset": "common_datasets.ecoli1",
                "n_repeats": 2,
                "n_folds": 3,
                "folding": "stratified_sklearn"}

    dataset = {"folds": [{"p": 3, "n": 7}, {"p": 3, "n": 7},
                        {"p": 4, "n": 6}, {"p": 3, "n": 7},
                        {"p": 3, "n": 7}, {"p": 4, "n": 6}]

Score bounds can be added in multiple ways, adding the same score bounds to each fold:

.. code-block:: Python

    dataset = {"p": 10, "n": 20, "n_repeats": 2, "n_folds": 3, "folding": "stratified_sklearn",
                "fold_score_bounds": {"acc": (0.8, 1.0), "sens": (0.8, 1.0)}}

Adding different bounds to the folds

.. code-block:: Python

    dataset = {"folds":
        [{"p": 3, "n": 7, "score_bounds": {"acc": (0.8, 1.0), "sens": (0.8, 1.0)}},
        {"p": 3, "n": 7, "score_bounds": {"sens": (0.8, 1.0)}},
        {"p": 4, "n": 6, "score_bounds": {"acc": (0.8, 1.0)}}]}

If the specification of a dataset is incomplete or not consistent, the package will guide the user with verbose exceptions on how to fix the specification.

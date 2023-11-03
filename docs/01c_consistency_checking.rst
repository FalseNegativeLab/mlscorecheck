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

    >>> from mlscorecheck.check.binary import check_1_testset_no_kfold

    >>> testset = {'p': 530, 'n': 902}

    >>> scores = {'acc': 0.62, 'sens': 0.22, 'spec': 0.86, 'f1p': 0.3, 'fm': 0.32}

    >>> result = check_1_testset_no_kfold(testset=testset,
                                                scores=scores,
                                                eps=1e-2)
    >>> result['inconsistency']
    # False

The interpretation of the outcome is that given a testset containing 530 positive and 902 negative samples, the reported scores *can* be the outcome of an evaluation. In the ``result`` structure one can find further information about the test. Namely, under the key ``n_valid_tptn_pairs`` one finds the number of ``tp`` and ``tn`` combinations which can lead to the reported performance scores with the given numerical uncertainty.

If one of the scores is altered, like accuracy is changed to 0.92, the configuration becomes infeasible:

.. code-block:: Python

    >>> scores = {'acc': 0.92, 'sens': 0.22, 'spec': 0.86, 'f1p': 0.3, 'fm': 0.32}

    >>> result = check_1_testset_no_kfold(testset=testset,
                                                scores=scores,
                                                eps=1e-2)
    >>> result['inconsistency']
    # True

As the ``inconsistency`` flag shows, here inconsistencies were identified, there are no such ``tp`` and ``tn`` combinations which would end up with the reported scores. Either the assumption on the properties of the dataset, or the scores are incorrect.

1 dataset with k-fold, mean-of-scores (MoS)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This scenario is the most common in the literature. A classification technique is executed to each fold in a (repeated) k-fold scenario, the scores are calculated for each fold, and the average of the scores is reported with some numerical uncertainty due to rounding/ceiling/flooring. Because of the averaging, this test supports only the linear scores (``acc``, ``sens``, ``spec``, ``bacc``) which usually are among the most commonly reported scores. The test constructs a linear integer program describing the scenario with the true positive and true negative parameters of all folds and checks its feasibility.

In the example below, a consistent set of figures is tested:

.. code-block:: Python

    >>> from mlscorecheck.check.binary import check_1_dataset_known_folds_mos

    >>> dataset = {'p': 126, 'n': 131}
    >>> folding = {'folds': [{'p': 52, 'n': 94}, {'p': 74, 'n': 37}]}

    >>> scores = {'acc': 0.573, 'sens': 0.768, 'bacc': 0.662}

    >>> result = check_1_dataset_known_folds_mos(dataset=dataset,
                                                        folding=folding,
                                                        scores=scores,
                                                        eps=1e-3)
    >>> result['inconsistency']
    # False

As indicated by the output flag, no inconsistencies were identified. The ``result`` dictionary contains some further details of the test. Most notably, under the ``lp_status`` key, one can find the status of the linear programming solver. Additionally, under the ``lp_configuration`` key, one can find the values of all true positive and true negative variables in all folds at the time of the termination of the solver. Furthermore, all scores are calculated for the individual folds and the entire dataset, as well.

If one of the scores is adjusted, for example, sensitivity is changed to 0.568, the configuration becomes infeasible:

.. code-block:: Python

    >>> scores = {'acc': 0.573, 'sens': 0.568, 'bacc': 0.662}
    >>> result = check_1_dataset_known_folds_mos(dataset=dataset,
                                                        folding=folding,
                                                        scores=scores,
                                                        eps=1e-3)
    >>> result['inconsistency']
    # True

Finally, we mention that if there are hints for bounds on the scores in the folds (for example, when the minimum and maximum scores across the folds are reported), one can add these figures to strengthen the test. In the next example, score bounds on accuracy have been added to each fold. This means the test checks if the reported scores can be achieved
with a true positive and true negative configuration with the specified lower and upper bounds for the scores in the individual folds:

.. code-block:: Python

    >>> dataset = {'dataset_name': 'common_datasets.glass_0_1_6_vs_2'}
    >>> folding = {'n_folds': 4, 'n_repeats': 2, 'strategy': 'stratified_sklearn'}

    >>> scores = {'acc': 0.9, 'spec': 0.9, 'sens': 0.6, 'bacc': 0.1, 'f1': 0.95}

    >>> result = check_1_dataset_known_folds_mos(dataset=dataset,
                                                        folding=folding,
                                                        fold_score_bounds={'acc': (0.8, 1.0)},
                                                        scores=scores,
                                                        eps=1e-2,
                                                        numerical_tolerance=1e-6)
    >>> result['inconsistency']
    # True

Note that in this example, although ``f1`` is provided, it is completely ignored as the aggregated tests work only for the four linear scores.

1 dataset with k-folds, score-of-means (SoM)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When the scores are calculated in the Score-of-Means (SoM) manner in a k-fold scenario, it means that the total confusion matrix of all folds is calculated first, and then the score formulas are applied to it. The only difference compared to the "1 testset no kfold" scenario is that the number of repetitions of the k-fold scheme multiples the ``p`` and ``n`` statistics of the dataset, but the actual structure of the folds is irrelevant. The result of the analysis is structured similarly to the "1 testset no kfold" case.

For example, testing a consistent scenario:

.. code-block:: Python

    >>> from mlscorecheck.check.binary import check_1_dataset_kfold_som

    >>> dataset = {'dataset_name': 'common_datasets.monk-2'}
    >>> folding = {'n_folds': 4, 'n_repeats': 3, 'strategy': 'stratified_sklearn'}

    >>> scores = {'spec': 0.668, 'npv': 0.744, 'ppv': 0.667,
                    'bacc': 0.706, 'f1p': 0.703, 'fm': 0.704}

    >>> result = check_1_dataset_kfold_som(dataset=dataset,
                                            folding=folding,
                                            scores=scores,
                                            eps=1e-3)
    >>> result['inconsistency']
    # False

If one of the scores is adjusted, for example, negative predictive value is changed to 0.754, the configuration becomes inconsistent:

.. code-block:: Python

    >>> {'spec': 0.668, 'npv': 0.754, 'ppv': 0.667,
            'bacc': 0.706, 'f1p': 0.703, 'fm': 0.704}

    >>> result = check_1_dataset_kfold_som(dataset=dataset,
                                            folding=folding,
                                            scores=scores,
                                            eps=1e-3)
    >>> result['inconsistency']
    # True

n testsets without k-folding, SoM over the testsets
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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

n testsets without k-folding, MoS over the testsets
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This scenario is analogous to the "n testsets without k-folding, SoM" scenario, except the aggregation over the testsets is carried out with the MoS approach. The output is structured similarly to the output of the "1 dataset k-fold MoS" scenario. In the first example, a feasible scenario is tested.

.. code-block:: Python

    from mlscorecheck.check.binary import check_n_testsets_mos_no_kfold

    testsets = [{'p': 349, 'n': 50},
                {'p': 478, 'n': 323},
                {'p': 324, 'n': 83},
                {'p': 123, 'n': 145}]

    scores = {'acc': 0.6441, 'sens': 0.6706, 'spec': 0.3796, 'bacc': 0.5251}
    results = check_n_testsets_mos_no_kfold(testsets=testsets,
                                                    scores=scores,
                                                    eps=0.0001)
    results["inconsistency"]
    # False

If one of the scores is slightly adjusted, for example, ``sens`` is updated to 0.6756, the configuration becomes infeasible.

.. code-block:: Python

    scores['sens'] = 0.6756

    results = check_n_testsets_mos_no_kfold(testsets=testsets,
                                                    scores=scores,
                                                    eps=0.0001)
    results["inconsistency"]
    # True

n datasets with k-folds, SoM over datasets and SoM over folds
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Again, the scenario is similar to the "1 dataset k-fold SoM" scenario, except there is another level of aggregation over datasets, and one single confusion matrix is determined for the entire experiment and the scores are calculated from that. In this scenario a list of evaluations need to be specified. The output of the test is structured similarly as in the "1 dataset k-fold SoM" case, there is a top level ``inconsistency`` flag indicating if inconsistency has been detected. In the following example, a consistent case is prepared with two datasets.

.. code-block:: Python

    >>> from mlscorecheck.check.binary import check_n_datasets_som_kfold_som

    >>> evaluation0 = {'dataset': {'p': 389, 'n': 630},
                        'folding': {'n_folds': 5, 'n_repeats': 2,
                                    'strategy': 'stratified_sklearn'}}
    >>> evaluation1 = {'dataset': {'dataset_name': 'common_datasets.saheart'},
                        'folding': {'n_folds': 5, 'n_repeats': 2,
                                    'strategy': 'stratified_sklearn'}}
    >>> evaluations = [evaluation0, evaluation1]

    >>> scores = {'acc': 0.631, 'sens': 0.341, 'spec': 0.802, 'f1p': 0.406, 'fm': 0.414}

    >>> result = check_n_datasets_som_kfold_som(scores=scores,
                                                        evaluations=evaluations,
                                                        eps=1e-3)
    >>> result['inconsistency']
    # False

However, if one of the scores is adjusted a little, like accuracy is changed to 0.731, the configuration becomes inconsistent:

.. code-block:: Python

    >>> scores = {'acc': 0.731, 'sens': 0.341, 'spec': 0.802, 'f1p': 0.406, 'fm': 0.414}

    >>> result = check_n_datasets_som_kfold_som(scores=scores,
                                                        evaluations=evaluations,
                                                        eps=1e-3)
    >>> result['inconsistency']
    # True

n datasets with k-folds, MoS over datasets and SoM over folds
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This scenario is about performance scores calculated for each dataset individually by the SoM aggregation in any k-folding strategy, and then the scores are aggregated across the datasets in the MoS manner. Because of the overall averaging, one cannot do inference about the non-linear scores, only the four linear scores are supported (``acc``, ``sens``, ``spec``, ``bacc``), and the scores are checked by linear programming. Similarly as before, the specification of a list of evaluations is needed. In the following example a consistent scenario is tested, with score bounds also specified on the datasets:

.. code-block:: Python

    >>> from mlscorecheck.check.binary import check_n_datasets_mos_kfold_som

    >>> evaluation0 = {'dataset': {'p': 39, 'n': 822},
                        'folding': {'n_folds': 5, 'n_repeats': 3,
                                    'strategy': 'stratified_sklearn'}}
    >>> evaluation1 = {'dataset': {'dataset_name': 'common_datasets.winequality-white-3_vs_7'},
                        'folding': {'n_folds': 5, 'n_repeats': 3,
                                    'strategy': 'stratified_sklearn'}}
    >>> evaluations = [evaluation0, evaluation1]

    >>> scores = {'acc': 0.312, 'sens': 0.45, 'spec': 0.312, 'bacc': 0.381}

    >>> result = check_n_datasets_mos_kfold_som(evaluations=evaluations,
                                                        dataset_score_bounds={'acc': (0.0, 0.5)},
                                                        eps=1e-4,
                                                        scores=scores)
    >>> result['inconsistency']
    # False

However, if one of the scores is adjusted a little (accuracy changed to 0.412 and the score bounds also changed), the configuration becomes infeasible:

.. code-block:: Python

    >>> scores = {'acc': 0.412, 'sens': 0.45, 'spec': 0.312, 'bacc': 0.381}
    >>> result = check_n_datasets_mos_kfold_som(evaluations=evaluations,
                                                        dataset_score_bounds={'acc': (0.5, 1.0)},
                                                        eps=1e-4,
                                                        scores=scores)
    >>> result['inconsistency']
    # True

The output is structured similarly to the '1 dataset k-folds MoS' case, one can query the status of the solver by the key ``lp_status`` and the actual configuration of the variables by the ``lp_configuration`` key. If there are hints on the minimum and maximum scores across the datasets, one can add those bounds through the ``dataset_score_bounds`` parameter to strengthen the test.

n datasets with k-folds, MoS over datasets and MoS over folds
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In this scenario, scores are calculated in the MoS manner for each dataset, and then aggregated again across the datasets. Again, because of the averaging, only the four linear scores (``acc``, ``sens``, ``spec``, ``bacc``) are supported. In the following example a consistent scenario is checked with three datasets and without score bounds specified at any level:

.. code-block:: Python

    >>> from mlscorecheck.check.binary import check_n_datasets_mos_known_folds_mos

    >>> evaluation0 = {'dataset': {'p': 118, 'n': 95},
                    'folding': {'folds': [{'p': 22, 'n': 23}, {'p': 96, 'n': 72}]}}
    >>> evaluation1 = {'dataset': {'p': 781, 'n': 423},
                    'folding': {'folds': [{'p': 300, 'n': 200}, {'p': 481, 'n': 223}]}}
    >>> evaluations = [evaluation0, evaluation1]

    >>> scores = {'acc': 0.61, 'sens': 0.709, 'spec': 0.461, 'bacc': 0.585}

    >>> result = check_n_datasets_mos_known_folds_mos(evaluations=evaluations,
                                                            scores=scores,
                                                            eps=1e-3)
    >>> result['inconsistency']
    # False

Again, the details of the analysis are accessible under the ``lp_status`` and ``lp_configuration`` keys. Adding an adjustment to the scores (turning accuracy to 0.71), the configuration becomes infeasible:

.. code-block:: Python

    >>> scores = {'acc': 0.71, 'sens': 0.709, 'spec': 0.461}

    >>> result = check_n_datasets_mos_known_folds_mos(evaluations=evaluations,
                                                        scores=scores,
                                                        eps=1e-3)
    >>> result['inconsistency']
    # True

If there are hints on the minimum and maximum scores across the datasets, one can add those bounds through the ``dataset_score_bounds`` parameter to strengthen the test.

Not knowing the mode of aggregation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The biggest challenge with aggregated scores is that the ways of aggregation at the dataset and experiment level are rarely disclosed explicitly. Even in this case the tools presented in the previous section can be used since there are hardly any further ways of meaningful averaging than (MoS on folds, MoS on datasets), (SoM on folds, MoS on datasets), (SoM on folds, SoM on datasets), hence, if a certain set of scores is inconsistent with each of these possibilities, one can safely say that the results do not satisfy the reasonable expectations.

Not knowing the k-folding scheme
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In many cases, it is not stated explicitly if stratification was applied or not, only the use of k-fold is phrased in papers. Not knowing the folding structure, the MoS aggregated tests cannot be used. However, if the cardinality of the minority class is not too big (a couple of dozens), then all potential k-fold configurations can be generated, and the MoS tests can be applied to each. If the scores are inconsistent with each, it means that no k-fold could result the scores. There are two functions supporting these exhaustive tests, one for the dataset level, and one for the experiment level.

Given a dataset and knowing that k-fold cross-validation was applied with MoS aggregation, but stratification is not mentioned, the following sample code demonstrates the use of the exhaustive test, with a consistent setup:

.. code-block:: Python

    >>> from mlscorecheck.check.binary import check_1_dataset_unknown_folds_mos

    >>> dataset = {'p': 126, 'n': 131}
    >>> folding = {'n_folds': 2, 'n_repeats': 1}

    >>> scores = {'acc': 0.573, 'sens': 0.768, 'bacc': 0.662}

    >>> result = check_1_dataset_unknown_folds_mos(dataset=dataset,
                                                        folding=folding,
                                                        scores=scores,
                                                        eps=1e-3)
    >>> result['inconsistency']
    # False

If the balanced accuracy score is adjusted to 0.862, the configuration becomes infeasible:

.. code-block:: Python

    >>> scores = {'acc': 0.573, 'sens': 0.768, 'bacc': 0.862}

    >>> result = check_1_dataset_unknown_folds_mos(dataset=dataset,
                                                        folding=folding,
                                                        scores=scores,
                                                        eps=1e-3)
    >>> result['inconsistency']
    # True

In the result of the tests, under the key ``details`` one can find the results for all possible fold combinations.

The following scenario is similar in the sense that MoS aggregation is applied to multiple datasets with unknown folding:

.. code-block:: Python

    >>> from mlscorecheck.check.binary import check_n_datasets_mos_unknown_folds_mos

    >>> evaluation0 = {'dataset': {'p': 13, 'n': 73},
                    'folding': {'n_folds': 4, 'n_repeats': 1}}
    >>> evaluation1 = {'dataset': {'p': 7, 'n': 26},
                    'folding': {'n_folds': 3, 'n_repeats': 1}}
    >>> evaluations = [evaluation0, evaluation1]

    >>> scores = {'acc': 0.357, 'sens': 0.323, 'spec': 0.362, 'bacc': 0.343}

    >>> result = check_n_datasets_mos_unknown_folds_mos(evaluations=evaluations,
                                                            scores=scores,
                                                            eps=1e-3)
    >>> result['inconsistency']
    # False

The setup is consistent. However, if the balanced accuracy is changed to 0.9, the configuration becomes infeasible:

.. code-block:: Python

    >>> scores = {'acc': 0.357, 'sens': 0.323, 'spec': 0.362, 'bacc': 0.9}

    >>> result = check_n_datasets_mos_unknown_folds_mos(evaluations=evaluations,
                                                            scores=scores,
                                                            eps=1e-3)
    >>> result['inconsistency']
    # True

Preterm delivery prediction from electrohysterogram signals (TPEHG dataset)
---------------------------------------------------------------------------

Electrohysterogram classification for the prediction of preterm delivery in pregnancy became a popular area for the applications of minority oversampling, however, it turned out that there were overly optimistic classification results reported due to systematic data leakage in the data preparation process [EHG]_. In [EHG]_, the implementations were replicated and it was shown that there is a decent gap in terms of performance when the data is prepared properly. However, data leakage changes the statistics of the dataset being cross-validated. Hence, the problematic scores could be identified with the tests implemented in the ``mlscorecheck`` package. In order to facilitate the use of the tools for this purpose, some functionalities have been prepared with the dataset already pre-populated.

The test bundle implemented in the ``mlscorecheck`` package is based on the TPEHG dataset [TPEHG]_, containing 262 negative and 38 positive samples. In the lack of predefined train/test splits, the dataset is usually evaluated in a k-fold cross-validation scenario with unknown fold structures.

For illustration, given a set of scores reported in a real paper, the test below shows that it is not consistent with the dataset:

.. code-block:: Python

    >>> from mlscorecheck.check.bundles.ehg import check_tpehg
    >>> # the 5-fold cross-validation scores reported in the paper
    >>> scores = {'acc': 0.9447, 'sens': 0.9139, 'spec': 0.9733}
    >>> eps = 0.0001
    >>> results = check_tpehg(scores=scores,
                                eps=eps,
                                n_folds=5,
                                n_repeats=1)
    >>> results['inconsistency']
    # True

As the results show, the reported scores are inconsistent with the assumption of being yielded in a 5-fold cross-validation experiment on the TPEHG dataset.

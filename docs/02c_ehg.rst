Preterm delivery prediction from electrohysterogram (EHG) signals
-----------------------------------------------------------------

Electrohysterogram classification for the prediction of preterm delivery in pregnancy became a popular area for the applications of minority oversampling, however, it turned out that there were overly optimistic classification results reported due to systematic data leakage in the data preparation process [EHG]_. In [EHG]_, the implementations were replicated and it was shown that there is a decent gap in terms of performance when the data is prepared properly. However, data leakage changes the statistics of the dataset being cross-validated. Hence, the problematic scores could be identified with the tests implemented in the ``mlscorecheck`` package. In order to facilitate the use of the tools for this purpose, some functionalities have been prepared with the dataset already pre-populated.

For illustration, given a set of scores reported in a real paper, the test below shows that it is not consistent with the dataset:

.. code-block:: Python

    >>> from mlscorecheck.bundles import check_ehg

    >>> scores = {'acc': 0.9552, 'sens': 0.9351, 'spec': 0.9713}

    >>> results = check_ehg(scores=scores, eps=10**(-4), n_folds=10, n_repeats=1)
    >>> results['inconsistency']
    # True

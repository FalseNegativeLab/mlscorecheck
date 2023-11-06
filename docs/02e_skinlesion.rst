Classification of skin lesions
------------------------------

The test bundle for the consistency testing of skin lesion classification results supports the ISIC2016 [ISIC2016]_ and ISIC2017 [ISIC2017]_ datasets.

ISIC2016 dataset
^^^^^^^^^^^^^^^^

The test set of the ISIC2016 [ISIC2016]_ dataset consists of 75 positive (malignant) and 304 negative (non-malignant) images and constitutes a binary classification problem.

In the following example, we illustrate the consistency testing of performance scores reported for this binary classification task:

.. code-block:: Python

    >>> from mlscorecheck.check.bundles.skinlesion import check_isic2016
    >>> scores = {'acc': 0.7916, 'sens': 0.2933, 'spec': 0.9145}
    >>> results = check_isic2016(scores=scores, eps=1e-4)
    >>> results['inconsistency']
    # False

As the results show, inconsistencies were not identified, the reported scores are compatible with the experiment.

ISIC2017 dataset
^^^^^^^^^^^^^^^^

The ISIC2017 [ISIC2017]_ dataset is a multiclass classification problem, consisting of three classes: melanoma, nevus, and seborrheic keratosis. The test set consists of 393 nevus, 117 melanoma and 90 seborrheic keratosis images (600 images in total). In practice, authors report the performance scores for the binary classification task of distinguishing one particular class from the other two. For example, the performance scores for the melanoma class describe the performance of distinguishing melanoma from nevus and seborrheic keratosis.

The consistency tests support these types of evaluations. In the following example, we illustrate the consistency testing of performance scores reported for the melanoma class:

.. code-block:: Python

    >>> from mlscorecheck.check.bundles.skinlesion import check_isic2017
    >>> scores = {'acc': 0.6183, 'sens': 0.4957, 'ppv': 0.2544, 'f1p': 0.3362}
    >>> results = check_isic2017(target='M',
                        against=['SK', 'N'],
                        scores=scores,
                        eps=1e-4)
    >>> results['inconsistency']
    # False

As the results show, the no inconsistencies were identified, the reported scores are compatible with the experiment.

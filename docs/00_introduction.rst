.. raw:: html
    :file: ga4.html

Introduction
============

The purpose
-----------

Performance scores of a machine learning technique (binary/multiclass classification, regression) are reported on a dataset and look suspicious (exceptionally high scores possibly due to a typo, uncommon evaluation methodology, data leakage in preparation, incorrect use of statistics, etc.). With the tools implemented in the package ``mlscorecheck``, one can test if the reported performance scores are consistent with each other and the assumptions on the experimental setup up.

Testing is as simple as the following example illustrated. Suppose the accuracy, sensitivity and specificity scores are reported for a binary classification testset consisting of p=100 and n=200 samples. All this information is supplied to the suitable test function and the result shows that that inconsistencies were identified: the scores could not be calculated from the confusion matrix of the testset:

.. code-block:: Python

    from mlscorecheck.check.binary import check_1_testset_no_kfold

    result = check_1_testset_no_kfold(testset={'p': 100, 'n': 200},
                                             scores={'acc': 0.9567, 'sens': 0.8545, 'spec': 0.9734},
                                             eps=1e-4)
    result['inconsistency']
    # True

The consistency tests are numerical and **not** statistical: if inconsistencies are identified, it means that either the assumptions on the experimental setup or the reported scores are incorrect.

In more detail
--------------

The evaluation of the performance of machine learning techniques, whether for original theoretical advancements or applications in specific fields, relies heavily on performance scores (https://en.wikipedia.org/wiki/Evaluation_of_binary_classifiers). Although reported performance scores are employed as primary indicators of research value, they often suffer from methodological problems, typos, and insufficient descriptions of experimental settings. These issues contribute to the replication crisis (https://en.wikipedia.org/wiki/Replication_crisis) and ultimately entire fields of research ([RV]_, [EHG]_). Even systematic reviews can suffer from using incomparable performance scores for ranking research papers [RV]_.

In practice, the performance scores cannot take any values independently, the scores reported for the same experiment are constrained by the experimental setup and need to express some internal consistency. For many commonly used experimental setups it is possible to develop numerical techniques to test if the scores could be the outcome of the presumed experiment on the presumed dataset. This package implements such consistency tests for some common experimental setups. We highlight that the developed tests cannot guarantee that the scores are surely calculated by some standards or a presumed evaluation protocol. However, *if the tests fail and inconsistencies are detected, it means that the scores are not calculated by the presumed protocols with certainty*. In this sense, the specificity of the test is 1.0, the inconsistencies being detected are inevitable.

For further information, see the preprint: https://arxiv.org/abs/2310.12527

Citation
========

If you use the package, please consider citing the following paper:

.. code-block:: BibTex

    @article{mlscorecheck,
        author={Attila Fazekas and Gy\"orgy Kov\'acs},
        title={Testing the Consistency of Performance Scores Reported for Binary Classification Problems},
        year={2023}
    }

Latest news
===========

* the 1.0.1 version of the package is released;
* the paper describing the numerical techniques is available as a preprint at: https://arxiv.org/abs/2310.12527
* 10 test bundles including retina image processing datasets, preterm delivery prediction from electrohysterograms and skin lesion classification has been added;
* multiclass and regression tests added.

Installation
============

Requirements
------------

The package has only basic requirements when used for consistency testing:

* ``numpy``
* ``pulp``
* ``scikit-learn``

.. code-block:: bash

    > pip install numpy pulp

In order to execute the unit tests for the computer algebra components or reproduce the algebraic solutions, either ``sympy`` or ``sage`` needs to be installed. The installation of ``sympy`` can be done in the usual way. To install ``sage`` in a ``conda`` environment, one needs to add the ``conda-forge`` channel first:

.. code-block:: bash

    > conda config --add channels conda-forge
    > conda install sage

Installing the package
----------------------

For consistency testing, the package can be installed from the PyPI repository as:

.. code-block:: bash

    > pip install mlscorecheck

For develompent purposes, one can clone the source code from the repository as

.. code-block:: bash

    > git clone git@github.com:gykovacs/mlscorecheck.git

and install the source code into the actual virtual environment as

.. code-block:: bash

    > cd mlscorecheck
    > pip install -e .

In order to use and test all functionalities (including the algebraic and symbolic computing parts), please install the ``requirements.txt``:

.. code-block:: bash

    > pip install -r requirements.txt

Contribution
============

To contribute, please start a discussion in the GitHub repository at https://github.com/gykovacs/mlscorecheck.

.. raw:: html
    :file: ga4.html

Introduction
=============

The purpose
-----------

Performance scores for binary classification are reported on a dataset and look suspicious (exceptionally high scores possibly due to typo, uncommon evaluation methodology, data leakage in preparation, incorrect use of statistics, etc.). With the tools implemented in the package ``mlscorecheck``, one can test if the reported performance scores are consistent with each other and the assumptions on the experimental setup up to the numerical uncertainty due to rounding/truncation/ceiling.

The consistency tests are numerical and **not** statistical: if inconsistencies are identified, it means that either the assumptions on the evaluation protocol or the reported scores are incorrect.

In more detail
--------------

Binary classification is one of the most fundamental tasks in machine learning. The evaluation of the performance of binary classification techniques, whether for original theoretical advancements or applications in specific fields, relies heavily on performance scores (https://en.wikipedia.org/wiki/Evaluation_of_binary_classifiers). Although reported performance scores are employed as primary indicators of research value, they often suffer from methodological problems, typos, and insufficient descriptions of experimental settings. These issues contribute to the replication crisis (https://en.wikipedia.org/wiki/Replication_crisis) and ultimately entire fields of research ([RV]_, [EHG]_). Even systematic reviews can suffer from using incomparable performance scores for ranking research papers [RV]_.

The majority of performance scores are calculated from the binary confusion matrix, or multiple confusion matrices aggregated across folds and/or datasets. For many commonly used experimental setups one can develop numerical techniques to test if there exists any confusion matrix (or matrices), compatible with the experiment and leading to the reported performance scores. This package implements such consistency tests for some common scenarios. We highlight that the developed tests cannot guarantee that the scores are surely calculated by some standards or a presumed evaluation protocol. However, *if the tests fail and inconsistencies are detected, it means that the scores are not calculated by the presumed protocols with certainty*. In this sense, the specificity of the test is 1.0, the inconsistencies being detected are inevitable.

For further information, see the preprint:

Citation
========

If you use the package, please consider citing the following paper:

.. code-block:: BibTex

    @article{mlscorecheck,
    author={Gy\"orgy Kov\'acs and Attila Fazekas},
    title={Checking the internal consistency of reported performance scores in binary classification},
    year={2023}
    }

Latest news
===========

* the 0.1.0 version of the package is released
* the paper describing the implemented techniques is available as a preprint at: https://arxiv.org/abs/2310.12527

Installation
============

Requirements
------------

The package has only basic requirements when used for consistency testing.

* ``numpy``
* ``pulp``

.. code-block:: bash

    > pip install numpy pulp

In order to execute the tests, one also needs ``scikit-learn``, in order to test the computer algebra components or reproduce the algebraic solutions, either ``sympy`` or ``sage`` needs to be installed. The installation of ``sympy`` can be done in the usual way. To install ``sage`` in a ``conda`` environment, one needs to add the ``conda-forge`` channel first:

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

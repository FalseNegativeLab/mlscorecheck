In a nutshell
=============

One comes across some performance scores of binary classification reported for a dataset and finds them suspicious (typo, unorthodox evaluation methodology, etc.). With the tools implemented in the ``mlscorecheck`` package one can test if the scores are consistent with each other and the assumptions on the experimental setup.

The consistency tests are numerical and **not** statistical: if inconsistencies are identified, it means that either the assumptions on the evaluation protocol or the reported scores are incorrect.

In more detail
==============

Binary classification is one of the most basic tasks in machine learning. The evaluation of the performance of binary classification techniques (whether it is original theoretical development or application to a specific field) is driven by performance scores (https://en.wikipedia.org/wiki/Evaluation_of_binary_classifiers). Despite performance scores provide the basis to estimate the value of reasearch, the reported scores usually suffer from methodological problems, typos and the insufficient description of experimental settings, contributing to the replication crisis (https://en.wikipedia.org/wiki/Replication_crisis) and the skewing of entire fields ([RV]_, [EHG]_), as the reported but usually incomparable scores are usually rank approaches and ideas.

Most of the performance score are functions of the values of the binary confusion matrix (with four entries: true positives (``tp``), true negatives (``tn``), false positives (``fp``), false negatives (``fn``)). Consequently, when multiple performance scores are reported for some experiment, they cannot take any values independently.

Depending on the experimental setup, one can develop techniques to check if the performance scores reported for a dataset are consistent. This package implements such consistency tests for some common scenarios. We highlight that the developed tests cannot guarantee that the scores are surely calculated by some standards. However, if the tests fail and inconsistencies are detected, it means that the scores are not calculated by the presumed protocols with certainty. In this sense, the specificity of the test is 1.0, the inconsistencies being detected are inevitable.

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

* the 0.0.1 version of the package is released
* the paper describing the implemented techniques is available as a preprint at:

Installation
============

Requirements
------------

The package has only basic requirements when used for consistency checking.

* ``numpy``
* ``pulp``

In order to execute the tests, one also needs ``scikit-learn``, in order to test the computer algebra components or reproduce the solutions, either ``sympy`` or ``sage`` needs to be installed. The installation of ``sympy`` can be done in the usual way. In order to install ``sage`` into a conda environment one needs adding the ``conda-forge`` channel first:

.. code-block:: bash

    > conda config --add channels conda-forge
    > conda install sage

Installing the package
----------------------

In order to use for consistency testing, the package can be installed from the PyPI repository as:

.. code-block:: bash

    > pip install mlscorecheck

For develompent purposes, one can clone the source code from the repository as

.. code-block:: bash

    > git clone git@github.com:gykovacs/mlscorecheck.git

And install the source code into the actual virtual environment as

.. code-block:: bash

    > cd mlscorecheck
    > pip install -e .

In order to use and test all functionalities (including the symbolic computing part), please install the ``requirements.txt``:

.. code-block:: bash

    > pip install -r requirements.txt

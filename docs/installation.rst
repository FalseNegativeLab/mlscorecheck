Requirements
============

The package has only basic requirements when used for consistency checking.

* ``numpy``
* ``pulp``

In order to execute the tests, one also needs ``scikit-learn``, in order to test the computer algebra components or reproduce the solutions, either ``sympy`` or ``sage`` needs to be installed. The installation of ``sympy`` can be done in the usual way. In order to install ``sage`` into a conda environment one needs adding the ``conda-forge`` channel first:

.. code-block:: bash

    > conda config --add channels conda-forge
    > conda install sage

Installing the package
======================

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

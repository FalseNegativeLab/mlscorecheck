Installation
============

The package has only basic requirements when used for consistency checking.

* ``numpy``
* ``pulp``

.. code-block:: bash

    > pip install numpy pulp

In order to execute the tests, one also needs ``scikit-learn``, in order to test the computer algebra components or reproduce the solutions, either ``sympy`` or ``sage`` needs to be installed. The installation of ``sympy`` can be done in the usual way. In order to install ``sage`` into a conda environment one needs adding the ``conda-forge`` channel first:

.. code-block:: bash

    > conda config --add channels conda-forge
    > conda install sage

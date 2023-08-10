.. -*- mode: rst -*-

..
  |CircleCI|_

|GitHub|_ |Codecov|_ |pylint|_ |ReadTheDocs|_ |PythonVersion|_ |PyPi|_ |License|_ |Gitter|_


..
  .. |CircleCI| image:: https://circleci.com/gh/analyticalmindsltd/smote_variants.svg?style=svg
  .. _CircleCI: https://circleci.com/gh/analyticalmindsltd/smote_variants

.. |GitHub| image:: https://github.com/analyticalmindsltd/smote_variants/workflows/Python%20package/badge.svg?branch=master
.. _GitHub: https://github.com/analyticalmindsltd/smote_variants/workflows/Python%20package/badge.svg?branch=master

.. |Codecov| image:: https://codecov.io/gh/analyticalmindsltd/smote_variants/branch/master/graph/badge.svg?token=GQNNasvi4z
.. _Codecov: https://codecov.io/gh/analyticalmindsltd/smote_variants

.. |pylint| image:: https://img.shields.io/badge/pylint-10.0-brightgreen
.. _pylint: https://img.shields.io/badge/pylint-10.0-brightgreen

.. |ReadTheDocs| image:: https://readthedocs.org/projects/smote-variants/badge/?version=latest
.. _ReadTheDocs: https://smote-variants.readthedocs.io/en/latest/?badge=latest

.. |PythonVersion| image:: https://img.shields.io/badge/python-3.7%20%7C%203.8%20%7C%203.9%20%7C%203.10-brightgreen
.. _PythonVersion: https://img.shields.io/badge/python-3.7%20%7C%203.8%20%7C%203.9%20%7C%203.10-brightgreen

.. |PyPi| image:: https://badge.fury.io/py/smote-variants.svg
.. _PyPi: https://badge.fury.io/py/smote-variants

.. |License| image:: https://img.shields.io/badge/license-MIT-brightgreen
.. _License: https://img.shields.io/badge/license-MIT-brightgreen

.. |Gitter| image:: https://badges.gitter.im/smote_variants.svg
.. _Gitter: https://gitter.im/smote_variants?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge

::

mlscorecheck: testing the consistency of machine learning performance scores
============================================================================

Installation
------------

The package has only basic requirements when used for consistency checking.

* `numpy`
* `pulp`

When one wants to reproduce the formulas used in the package for the reconstruction of
scores, one of the currently supported two symbolic computation systems also needs to
be installed:

* `sympy`
* `sage`

Installing sage into a conda environment:

.. code-block::

    conda config --add channels conda-forge
    conda install sage

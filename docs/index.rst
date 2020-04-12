Welcome to statinf's documentation!
===================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   :glob:


STATINF
=======

.. image:: https://pepy.tech/badge/statinf
   :target: https://pepy.tech/project/statinf
.. image:: https://badge.fury.io/py/statinf.svg
   :target: https://pypi.org/project/statinf/

This library aims at re-implementing standard statistical tools (such as OLS, logistic regression, Neural Network) and is built on top of `numpy`_
for handling data and `theano`_ for differential computing.
The objective is to implement new methodologies from research projects on these models.
The library also provides a data generator for linear and binary data.


The library is pip-installable and the source code is available on my Git.
For any question or suggestion of improvement, please `contact me <mailto:florian.website.mail@gmail.com>`_.


Installation
============

You can get STATINF from `PyPI <https://pypi.org/project/statinf/>`_ with:

.. code-block:: console

   pip install statinf



The library is supported on Windows, Linux and MacOS.

STATINF tries to use the least number of dependencies possible:

* `pandas <https://pandas.pydata.org/>`_: used to convert data frames into arrays.
* `numpy`_ : main library for data handling and matrix operations.
* `scipy <https://www.scipy.org/>`_: probability density estimations.
* `theano`_: tensor operations and back-propagation for Deep Learning models.
* `matplotlib <https://matplotlib.org/>`_: plots of training performances.
* `pycof <https://www.florianfelice.com/pycof>`_: basic information printing.



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


Content
=======

.. toctree::
   :maxdepth: 4

   modules/statinf


.. _numpy: https://numpy.org/
.. _theano: http://deeplearning.net/software/theano/
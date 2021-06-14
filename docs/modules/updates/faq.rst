###
FAQ
###

*****************************************************
I have an error when using a function, what can I do?
*****************************************************

If you encounter an unexpected error (coming from the source code), we recommend to first check that you are using the latest version of STATINF.

.. code::

    pip3 show statinf

If the version is not the latest one, please upgrade it.

.. code::

    pip3 install --upgrade statinf

If you still encounter the same error with the most recent version, please raise an `issue`_.

----


*****************************
Why is my GPU not recognized?
*****************************

STATINF uses `jax`_ for the backend computations.
By default, once installed GPU might not be enabled and computations will fall back on CPUs.
This will be seen when importing STATINF, the below warning message may appear:

.. code-block:: python

    >>> from statinf.ml import MLP, Layer
    WARNING:absl:No GPU/TPU found, falling back to CPU. (Set TF_CPP_MIN_LOG_LEVEL=0 and rerun for more info.)

You will need to install the updated version of `jaxlib` based on your cuda version.
You can find your CUDA version with the command:

.. code::
    
    nvcc --version


Based on the `JAX documentation <https://github.com/google/jax#pip-installation>`_, run:

.. code::

    pip install --upgrade pip
    pip install --upgrade jax jaxlib==0.1.67+cuda111 -f https://storage.googleapis.com/jax-releases/jax_releases.html


Once installed, if when importing STATINF you see the error :code:`RuntimeError: CUDA operation failed: cudaGetErrorString symbol not found`, the problem will be coming from an incorrect version of cuda specified when upgrading jaxlib.

In that case, the JAX documentation suggests that for CUDA 11.1, 11.2, or 11.3, use :code:`cuda111`, for CUDA 11.0, use :code:`cuda110`, for CUDA 10.2, use :code:`cuda102`, for CUDA 10.1, use :code:`cuda101`.
See `this issue <https://github.com/google/jax/issues/5080>`_ for more details.

If you are not able to fix the issue, please `contact me <mailto:florian.website.mail@gmail.com>`_ or raise an `issue`_.

----


**********************************
How can I get a new model/feature?
**********************************

We are happy to support and implement new models or features in STATINF.
If you want to request a new implementation, please `contact me <mailto:florian.website.mail@gmail.com>`_ or raise an `issue`_.


----


****************************
Can I contribute to STATINF?
****************************

Similar to adding new features, we are happy to onboard new contributors to the project.
If you have ideas for improvement, new features to add or if you simply want to participate, please `contact me <mailto:florian.website.mail@gmail.com>`_.



.. _git: https://github.com/florianfelice/statinf/
.. _issue: https://github.com/florianfelice/statinf/issues
.. _jax: https://jax.readthedocs.io
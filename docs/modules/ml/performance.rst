Model performance
=================

Binary accuracy
---------------

.. automodule:: statinf.ml.performance
    :members:
    :undoc-members:
    :show-inheritance:


Example
-------

.. code-block:: python

    from statinf.ml.performance import BinaryPerformance

    ## Load the performance class
    perf = BinaryPerformance(y_true=data[Y], y_pred=predicted)

    ## Model accuracy
    print(perf.accuracy())
    ## Confusion matrix
    print(perf.confusion())
    
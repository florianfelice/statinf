###############
Library updates
###############


This section aims at showing the latest release of the library.
We show most important releases that included new features.
Library versions in between are used to fix bugs and implement improvement suggested by users' feedback.


----


*************************************************************************************
1.0.23 - May 17, 2020 - New model :func:`~GLM` and improved features for :func:`~OLS`
*************************************************************************************

* **GLM**

STATINF now provides a module for Generalized Linear Models: :py:meth:`statinf.regressions.glm.GLM`.
The module currently implements binomial and gaussian families of distribution for Logit and Probit models.
We will soon extend to other families.

The module will also propose different heteroskedastic-robut covariance estimates such as Sandwich,
:math:`HC0`, :math:`HC1`, :math:`HC2` and :math:`HC3` (non-robust is currently available).
The module fits the model with the Newton-Raphson's method.
It proposes a function for Partial and Average Partial Effect (:py:meth:`statinf.regressions.glm.GLM.partial_effects`).


* **OLS**

The :obj:`argument` in :py:meth:`statinf.regressions.LinearModels.OLS` now supports variable transformations.
You can run you regressions and created interaction variables in the forumla.
This can be particularly useful for `difference in differences <https://en.wikipedia.org/wiki/Difference_in_differences>`_.

Also, the function :py:meth:`statinf.regressions.LinearModels.OLS.predict` has been upgraded and can not return confidence intervals for prediction.
The function will continue to return a :obj:`numpy.array` for standard predictions and will return a :obj:`pandas.DataFrame` for confidence intervals.

.. math:: \left[ \hat{Y} \pm z_{1 - \frac{\alpha}{2}} \dfrac{\sigma}{\sqrt{n - 1}} \right]


* **BinaryPerformance**

The syntax issues encountered in :py:meth:`statinf.ml.performance.BinaryPerformance` in version 1.1.16 have been fixed.
The module is now fully working.


.. note::
    A new version of STATINF 1.1.0 will soon be released with a stable version of all existing components.
    Stay tuned!


^^^^^^^^^^^^^^
How to use it?
^^^^^^^^^^^^^^

.. code::

    from statinf.regressions import GLM


.. code::

    from statinf.regressions import GLM
    
    # We set the Logit formula
    formula = "Y ~ X0 + X1 + X2 + X3 + X4"
    # We fit the GLM with the data, the formula and without intercept
    logit = GLM(formula, data, fit_intercept=False, family='binomial')
    logit.fit(cov_type='nonrobust', plot=False)

    logit.summary()


^^^^^^^^^^^^^^^^^^
How to install it?
^^^^^^^^^^^^^^^^^^

.. code::

    pip3 install statinf==1.0.23


See more details: :py:meth:`statinf.regressions.LinearModels.GLM` and :py:meth:`statinf.regressions.LinearModels.OLS`


----


*************************************************
1.0.21 - Apr 26, 2020 - New module :func:`~stats`
*************************************************

STATINF now comes with an advanced statistics modules.

It allows to get traditional descriptive statistics such as :py:meth:`statinf.stats.descriptive.var`, :py:meth:`statinf.stats.descriptive.cov`
but also :py:meth:`statinf.stats.descriptive.pearson` and :py:meth:`statinf.stats.descriptive.spearman`.

This modules also provides statistical tests :py:meth:`statinf.stats.tests.ttest` for one sample and :py:meth:`statinf.stats.tests.ttest_2samp`
as well as :py:meth:`statinf.stats.tests.kstest`.


^^^^^^^^^^^^^^
How to use it?
^^^^^^^^^^^^^^

.. code::

    from statinf import stats
    
    # Normality test
    stats.kstest(data.X)

    # 1-sample test for a mean
    stats.ttest([30.02, 29.99, 30.11, 29.97, 30.01, 29.99], mu=30)

    # 2-sample test for comparing means
    stats.ttest(a, b)


^^^^^^^^^^^^^^^^^^
How to install it?
^^^^^^^^^^^^^^^^^^

.. code::

    pip3 install statinf==1.0.21


See more details and examples: :py:meth:`statinf.stats.tests` and :py:meth:`statinf.stats.descriptive`


----


******************************************************
1.0.19 - Apr 17, 2020 - Update for :func:`OLS` summary
******************************************************

The summary :py:meth:`statinf.regressions.LinearModels.OLS` model has been improved with better layout and the possibility to return a DataFrame.
This will allow users to extract individual components (such as variable p-value, coefficients, ...) for automation purposes.



^^^^^^^^^^^^^^
How to use it?
^^^^^^^^^^^^^^

.. code::

    from statinf.regressions import OLS

    # We set the OLS formula
    formula = "Y ~ X1 + X2 + X3 + X0"
    # We fit the OLS with the data, the formula and without intercept
    ols = OLS(formula, data, fit_intercept=False)

    ols.summary()


^^^^^^^^^^^^^^^^^^
How to install it?
^^^^^^^^^^^^^^^^^^

.. code::

    pip3 install statinf==1.0.19


See more details: :py:meth:`statinf.regressions.LinearModels.OLS`
More detailled example: `OLS example <../econometrics/ols/ols.html#example>`_


----


*************************************************************
1.0.16 - Mar 22, 2020 - New module :func:`~BinaryPerformance`
*************************************************************

A new modules has been implemented for evaluation binary classification problem with :py:meth:`statinf.ml.performance.BinaryPerformance`.
The module (model agnostic) takes the prediction from a model and can return: model accuracy, confusion matrix, F1-score, precision and recall.

.. note::
    Syntax issues have been identified in this module. This will be corrected in version 1.1.22.


^^^^^^^^^^^^^^
How to use it?
^^^^^^^^^^^^^^

.. code::

    from statinf.ml.performance import BinaryPerformance

    ## Load the performance class
    perf = BinaryPerformance(y_true=data[Y].values, y_pred=predicted)

    ## Model accuracy
    print(perf.accuracy())
    ## Confusion matrix
    print(perf.confusion())


^^^^^^^^^^^^^^^^^^
How to install it?
^^^^^^^^^^^^^^^^^^

.. code::

    pip3 install statinf==1.0.16


See more details: :py:meth:`statinf.ml.performance.BinaryPerformance`


----


************************************************
1.0.12 - Mar 10, 2020 - New optimizers available
************************************************

Multiple optimizers have been added on top of the traditional :func:`~SGD`.
:py:meth:`statinf.ml.neuralnetwork.MLP` now supports :py:meth:`statinf.ml.optimizers.Adam`, :py:meth:`statinf.ml.optimizers.AdaMax`,
:py:meth:`statinf.ml.optimizers.AdaGrad`, :py:meth:`statinf.ml.optimizers.SGD`, :py:meth:`statinf.ml.optimizers.MomentumSGD` and
:py:meth:`statinf.ml.optimizers.RMSprop`.


^^^^^^^^^^^^^^
How to use it?
^^^^^^^^^^^^^^

.. code::

    from statinf.ml import MLP, Layer

    # Initialize the network and its architecture
    nn = MLP(loss='mse')
    nn.add(Layer(4, 1, activation='linear'))

    # Train the neural network
    nn.train(data=data, X=X, Y=Y, epochs=1, learning_rate=0.001, optimizer='adam')



^^^^^^^^^^^^^^^^^^
How to install it?
^^^^^^^^^^^^^^^^^^

.. code::

    pip3 install statinf==1.0.12


See more details: :py:meth:`statinf.ml.optimizers`




----


***********************************************************************
1.0.7 - Feb 1, 2020 - New model :func:`~MLP`
***********************************************************************

STATINF now offers a Neural Network framework with :py:meth:`statinf.ml.neuralnetwork.MLP`.
It is powered by `theano`_ and `numpy`_.
It allows to run a model on multiple CPUs or with GPUs (see `Theano documentation <http://deeplearning.net/software/theano/tutorial/using_gpu.html>`_).


^^^^^^^^^^^^^^
How to use it?
^^^^^^^^^^^^^^

.. code::

    from statinf.ml import MLP, Layer

    # Initialize the network and its architecture
    nn = MLP(loss='mse')
    nn.add(Layer(4, 1, activation='linear'))

    # Train the neural network
    nn.train(data=data, X=X, Y=Y, epochs=1, learning_rate=0.001)



^^^^^^^^^^^^^^^^^^
How to install it?
^^^^^^^^^^^^^^^^^^

.. code::

    pip3 install statinf==1.0.7


See more details: :py:meth:`statinf.ml.neuralnetwork.MLP`
More examples: `Getting Stated - Deep Learning <../deeplearning/example.html>`_.



.. _git: https://github.com/florianfelice/PYCOF/
.. _issue: https://github.com/florianfelice/PYCOF/issues

.. _theano: http://deeplearning.net/software/theano/
.. _numpy: https://numpy.org/

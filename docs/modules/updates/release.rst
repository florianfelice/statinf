###############
Library updates
###############


This section aims at showing the latest release of the library.
We show most important releases that included new features.
Library versions in between are used to fix bugs and implement improvement suggested by users' feedback.

----


********************************************************************************************************
1.2.5 - May 5, 2023 - New distribution :meth:`statinf.distributions.discrete.NegativeBinomial` available
********************************************************************************************************

Statinf's distributions module now includes a class for the Negative Binomial distribution.
In the same spirit as already available distributions, the module allows to estimate parameters from data via Maximum Likelihood Estimation (MLE) and generate samples from the estimated (or user-provided) parameters.

In this release, the computations for the normalizing factor in the Conway-Maxwell Poisson distribution (:py:meth:`statinf.distributions.discrete.CMPoisson.Z`) has been improved.
This now offers stable results of the computation heavy factor improving the results from the parameters estimation.
The new implementation also simplifies computations of the infinite sum of factorials by using mathematical tricks.
Exploiting the fact that :math:`a^{b} = \exp(b \cdot \log(a))`, we can simplify factors such as :math:`(j!)^{\nu}` to be :math:`\exp \left(\nu \cdot \sum_{i=1}^{j} {\log(i)} \right)`.
This significantly speeds up computations while leading to the same results.

The main changes are:

* :py:meth:`statinf.distributions.discrete.NegativeBinomial` for the generic Poisson distribution
* :py:meth:`statinf.distributions.discrete.CMPoisson.Z` improved implementation of the normalizing factor


^^^^^^^^^^^^^^
How to use it?
^^^^^^^^^^^^^^


.. code::

    from statinf.distributions import NegativeBinomial

    # Let us generate a random sample of size 1000
    x = NegativeBinomial(n_=5, p_=0.15).sample(size=1000)
    
    # We can also estimate the parameters from the generated sample
    # We just need to initialize the class...
    nb = NegativeBinomial()
    
    # ... and we can fit from the generated sample. The function returns a dictionary
    nb.fit(x)
    # The class stores the value of the estimated parameters, 
    # wo we can generate more samples using the fitted parameters
    y = nb.sample(200)


^^^^^^^^^^^^^^^^^^
How to install it?
^^^^^^^^^^^^^^^^^^

.. code::

    pip3 install statinf==1.2.4


See more details: :py:meth:`statinf.distributions.discrete.NegativeBinomial` and :py:meth:`statinf.distributions.discrete.CMPoisson.Z`.

----

********************************************************************************************************************
1.2.0 - February 5, 2023 - New module :meth:`statinf.distributions` available for discrete distributions
********************************************************************************************************************

Statinf now includes a module for proability distributions.
The module allows to estimate parameters from data via Maximum Likelihood Estimation (MLE) and generate samples from the estimated (or user-provided) parameters.
The module currently include discrete distributions, continuous distributions will be added later.

The main additions are:

* :py:meth:`statinf.distributions.discrete.Poisson` for the generic Poisson distribution
* :py:meth:`statinf.distributions.discrete.CMPoisson` for the Conway-Maxwell Poisson distribution, a generalization of the Poisson

More discrete and continuous distributions will be added soon.

^^^^^^^^^^^^^^
How to use it?
^^^^^^^^^^^^^^


.. code::

    from statinf.distributions import Poisson
    
    # Let us generate a random sample of size 1000
    x = Poisson(lambda_=2.5).sample(size=1000)
    
    # We can also estimate the parameter from the generated sample
    # We just need to initialize the class...
    poiss = Poisson()
    
    # ... and we can fit from the generated sample. The function returns a dictionary
    poiss.fit(x)
    
    # The class stores the value of the estimated parameters,
    # so we can generate more samples using the fitted parameters
    y = poiss.sample(200)


^^^^^^^^^^^^^^^^^^
How to install it?
^^^^^^^^^^^^^^^^^^

.. code::

    pip3 install statinf==1.2.0


See more details: :py:meth:`statinf.distributions.discrete.Discrete`.

----


*********************************************************************************************
1.1.0 - March 7, 2021 - New backend dependency for :py:meth:`statinf.ml.neuralnetwork` module
*********************************************************************************************

Jax is now the new dependency for Statinf Deep Learning module.
The module supports the same functionalities but the backend has moved from `theano`_ to `jax`_.
The library offers auto-gradient computations that better scale with GPUs for high performance.

The main changes are:

* :py:meth:`statinf.ml.optimizers.SGD` offer the argument `alpha` as the momentum parameter and replaces the function :py:meth:`statinf.ml.optimizers.MomentumSGD`.
* :py:meth:`statinf.ml.neuralnetwork.MLP.fit` replaces the previous :py:meth:`train` function and offers the same functionalities.
* The value for the parameters of the optimizer can now be passed through the :py:meth:`statinf.ml.neuralnetwork.MLP.fit` function.

More functionalities will soon be available.

^^^^^^^^^^^^^^
How to use it?
^^^^^^^^^^^^^^


.. code::

    from statinf.ml import MLP, Layer

    # Initialize the network and its architecture
    nn = MLP(loss='mse')
    nn.add(Layer(4, 1, activation='linear'))

    # Train the neural network
    nn.fit(data=data, X=X, Y=Y, epochs=1, learning_rate=0.001)


^^^^^^^^^^^^^^^^^^
How to install it?
^^^^^^^^^^^^^^^^^^

.. code::

    pip3 install statinf==1.1.0


See more details: :py:meth:`statinf.ml.neuralnetwork.MLP`.



----


**********************************************************************************************
1.0.28 - September 27, 2020 - Time series module :py:meth:`statinf.stats.timeseries` available
**********************************************************************************************

New functions for time series are now available in the :obj:`statinf.stats` module.

The current functions available are:

* :py:meth:`statinf.stats.timeseries.adf_test` for Augmented Dickey-Fuller test.

* :py:meth:`statinf.stats.timeseries.coint_test` for cointegration test.

More functionalities will soon be available.

^^^^^^^^^^^^^^
How to use it?
^^^^^^^^^^^^^^


.. code::

    from statinf.stats import adf_test, coint_test
    
    import requests
    import pandas as pd

    # Function to generate data
    def get_bitfinex_asset(asset, ts_start=None, ts_end=None):
        # Defaults from 1 January 2018, 00:00:00
        ts_ms_start = 1514768400000 if ts_start is None else ts_start
        ts_ms_end = int(datetime.datetime.now().timestamp()*100) if ts_end is None else ts_end
        url = 'https://api.bitfinex.com/v2/candles/trade:1D:t' + asset + '/hist'
        params = { 'start': ts_ms_start, 'end': ts_ms_end, 'sort': 1}
        r = requests.get(url, params=params)
        data = r.json()
        return pd.DataFrame(data)[2]

    # Create the data series
    series = get_bitfinex_asset(asset='BTCUSD')
    series2 = get_bitfinex_asset(asset='NEOUSD')

    # Test stationarity of the first series with ADF test
    ts.adf_test(series, trend='ct')
    # Test cointegration of both series
    ts.coint_test(series, series2, trend='ct')



^^^^^^^^^^^^^^^^^^
How to install it?
^^^^^^^^^^^^^^^^^^

.. code::

    pip3 install statinf==1.0.28


See more details: :py:meth:`statinf.stats.timeseries.adf_test` and :py:meth:`statinf.stats.timeseries.coint_test`.



----


***********************************************************************************
1.0.27 - September 13, 2020 - New module :py:meth:`statinf.data.ProcessData.Scaler`
***********************************************************************************

A new functionality is now available for Machine Learning models to scale data.
The class :py:meth:`statinf.data.ProcessData.Scaler` includes the below methods:

* :obj:`MinMax` Scales the data to range between 0 and 1.

.. math:: x_{\text{scaled}} = \dfrac{x - \min(x)}{\max(x) - \min(x)}


* :obj:`Normalize` Scales the data to have mean 0 and standard deviation 1 (user can chose to center or reduce or not).

.. math:: x_{\text{scaled}} = \dfrac{x - \bar{x}}{\sqrt{\mathbb{V}(x)}}


^^^^^^^^^^^^^^
How to use it?
^^^^^^^^^^^^^^



.. code::

    from statinf.data import Scaler
    
    # Load the Scaler class
    scaler = Scaler(data=df, columns=['X1', 'X2'])
    # Scale our dataset with MinMax method
    scaled_df = scaler.MinMax()
    print(scaled_df)
    # Unscale data
    unscaled_df = scaler.unscaleMinMax(scaled_df)
    print(unscaled_df)


^^^^^^^^^^^^^^^^^^
How to install it?
^^^^^^^^^^^^^^^^^^

.. code::

    pip3 install statinf==1.0.27


See more details: :py:meth:`statinf.data.Scaler`.


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


See more details: :py:meth:`statinf.regressions.LinearModels.GLM` and :py:meth:`statinf.regressions.LinearModels.OLS`.


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


See more details and examples: :py:meth:`statinf.stats.tests` and :py:meth:`statinf.stats.descriptive`.


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


See more details: :py:meth:`statinf.ml.performance.BinaryPerformance`.


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


See more details: :py:meth:`statinf.ml.optimizers`.




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


See more details: :py:meth:`statinf.ml.neuralnetwork.MLP`.

More examples: `Getting Stated - Deep Learning <../deeplearning/example.html>`_.



.. _git: https://github.com/florianfelice/PYCOF/
.. _issue: https://github.com/florianfelice/PYCOF/issues

.. _theano: http://deeplearning.net/software/theano/
.. _jax: https://jax.readthedocs.io
.. _numpy: https://numpy.org/

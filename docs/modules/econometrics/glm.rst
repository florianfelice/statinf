Generalized Linear Models
=========================


GLM
---

.. automodule:: statinf.regressions.glm
    :members:
    :undoc-members:
    :show-inheritance:


Example
-------

.. code-block:: python

    from statinf.regressions import GLM
    
    # We set the Logit formula
    formula = "Y ~ X0 + X1 + X2 + X3 + X4"
    # We fit the GLM with the data, the formula and without intercept
    logit = GLM(formula, data, fit_intercept=False, family='binomial')
    logit.fit(cov_type='nonrobust', plot=False)

    logit.summary()

Output will be:

.. code-block::

    ==================================================================================
    |                                  Logit summary                                 |
    ==================================================================================
    | McFadden's R²   =          0.67128 | McFadden's R² Adj.  =              0.6424 |
    | Log-Likelihood  =          -227.62 | Null Log-Likelihood =             -692.45 |
    | LR test p-value =              0.0 | Covariance          =           nonrobust |
    | n               =              999 | p                   =                  5  |
    | Iterations      =                8 | Convergence         =                True |
    ==================================================================================
    | Variables         | Coefficients   | Std. Errors  | t-values   | Probabilities |
    ==================================================================================
    | X0                |       -1.13024 |      0.10888 |    -10.381 |     0.0   *** |
    | X1                |        0.02963 |      0.07992 |      0.371 |   0.711       |
    | X2                |       -1.40968 |       0.1261 |    -11.179 |     0.0   *** |
    | X3                |         0.5253 |      0.08966 |      5.859 |     0.0   *** |
    | X4                |        0.14705 |      0.25018 |      0.588 |   0.557       |
    ==================================================================================
    | Significance codes: 0. < *** < 0.001 < ** < 0.01 < * < 0.05 < . < 0.1 < '' < 1 |
    ==================================================================================

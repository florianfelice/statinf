Linear regression
=================


OLS
---

.. automodule:: statinf.regressions.LinearModels
    :members:
    :undoc-members:
    :show-inheritance:


Examples
--------

OLS
***

.. code-block:: python

    import statinf.data as gd
    from statinf.regressions import OLS

    # Generate a synthetic dataset
    data = gd.generate_dataset(coeffs=[1.2556, -0.465, 1.665414, 2.5444, -7.56445], n=1000, std_dev=2.6)
    
    # We set the OLS formula
    formula = "Y ~ X0 + X1 + X2 + X3 + X4 + X1*X2 + exp(X2)"
    
    # We fit the OLS with the data, the formula and without intercept
    ols = OLS(formula, data, fit_intercept=False)

    ols.summary()

Output will be:

.. code-block::

    ==================================================================================
    |                                  OLS summary                                   |
    ==================================================================================
    | R²             =             0.9846 | R² Adj.      =                   0.98449 |
    | n              =                999 | p            =                         7 |
    | Fisher value   =           10568.56 |                                          |
    ==================================================================================
    | Variables         | Coefficients   | Std. Errors  | t-values   | Probabilities |
    ==================================================================================
    | X0                |         1.2898 |      0.03218 |     40.085 |     0.0   *** |
    | X1                |       -0.50096 |      0.03187 |    -15.718 |     0.0   *** |
    | X2                |        1.62202 |      0.04264 |     38.039 |     0.0   *** |
    | X3                |        2.56471 |      0.03196 |     80.252 |     0.0   *** |
    | X4                |       -7.58065 |      0.03226 |   -234.983 |     0.0   *** |
    | X1*X2             |       -0.03968 |      0.03438 |     -1.154 |   0.249       |
    | exp(X2)           |        0.00301 |      0.01692 |      0.178 |   0.859       |
    ==================================================================================
    | Significance codes: 0. < *** < 0.001 < ** < 0.01 < * < 0.05 < . < 0.1 < '' < 1 |
    ==================================================================================


You can also predict new values with their confidence interval

.. code-block:: python

    # Generate a new synthetic dataset
    test_data = generate_dataset(coeffs=[1.2556, -0.465, 1.665414, 2.5444, -7.56445], n=1000, std_dev=2.6)

    # Predict with 95% confidence interval
    ols.predict(test_data, conf_level=.95)

Output will be:

.. code-block::

        Prediction  LowerBound  UpperBound
    0    -19.252926  -19.265841  -19.240012
    1      4.988078    4.975164    5.000993
    2     10.824623   10.811708   10.837537
    3     -2.725563   -2.738477   -2.712649
    4      4.057040    4.044125    4.069954


LinearBayes
***********

.. code-block:: python

    from statinf.regressions import LinearBayes
Linear regression
=================


OLS
---

.. automodule:: statinf.regressions.LinearModels
    :members:
    :undoc-members:
    :show-inheritance:


Example
-------

.. code-block:: python

    import statinf.GenerateData as gd
    from statinf.regressions import OLS

    # Generate a synthetic dataset
    data = generate_dataset(coeffs=[1.2556, -0.465, 1.665414, 2.5444, -7.56445], n=1000, std_dev=2.6)
    
    # We set the OLS formula
    formula = "Y ~ X1 + X2 + X3 + X0 + X1*X2 + exp(X2)"
    # We fit the OLS with the data, the formula and without intercept
    ols = OLS(formula, data, fit_intercept=False)

    ols.summary()

Output will be:

.. code-block::

    =========================================================================
                                OLS summary                               
    =========================================================================
    | R² = 0.98540                  | Adjusted-R² = 0.98530
    | n  =    999                   | p =     7
    | Fisher = 11160.21338
    =========================================================================
    Variables  Coefficients  Standard Errors    t-values  Probabilities
        X0      1.311338         0.031827   41.202252       0.000000
        X1     -0.469525         0.030507  -15.390557       0.000000
        X2      1.633329         0.040214   40.615687       0.000000
        X3      2.562506         0.030486   84.056517       0.000000
        X4     -7.487243         0.032274 -231.987140       0.000000
     X1*X2      0.046324         0.030565    1.515569       0.129945
    exp(X2)     0.020938         0.015474    1.353057       0.176344


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
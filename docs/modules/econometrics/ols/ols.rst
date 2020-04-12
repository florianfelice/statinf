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
    formula = "Y ~ X1 + X2 + X3 + X0"
    # We fit the OLS with the data, the formula and without intercept
    ols = OLS(formula, data, fit_intercept=False)

    ols.summary()

Output will be:

.. code-block::

    =========================================================================
                                OLS summary                               
    =========================================================================
    | R² = 0.98484                  | Adjusted-R² = 0.98477
    | n  =    999                   | p =     5
    | Fisher = 16146.04006                         
    =========================================================================
    Variables  Coefficients  Standard Errors    t values  Probabilites
        X0      1.284372         0.032941   38.989887           0.0
        X1     -0.477014         0.031606  -15.092496           0.0
        X2      1.645034         0.034310   47.945992           0.0
        X3      2.571289         0.031940   80.504863           0.0
        X4     -7.634125         0.032077 -237.994821           0.0

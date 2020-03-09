import numpy as np
import pandas as pd

def generate_dataset(coeffs, n, std_dev, intercept=0., distribution='normal', binary=False, seed=None, **kwargs):
    """
    Generate a synthetic dataset for linear or binary data.

    Args:
    coeffs (list): Coefficients for the linear piece.
    n (int): Number of observations/rows required.
    std_dev (float): Standard deviation for the normally distributed data.
    intercept (float): Value of the intercept (defaults 0.0).
    distribution (str): Probability distribution from which we draw the data (defaults 'normal').
    binary (bool): Is the output data (y) binary (defaults False).
    """

    rdm = np.random.RandomState(seed) if seed else np.random
    # We calculate the number of predictors, and create a coefficient matrix
    # With `p` rows and 1 column, for matrix multiplication
    p = len(coeffs)
    params = pd.DataFrame({'coeff': coeffs, 'std_dev': std_dev})

    # Similar as before, but with `n` rows and `p` columns this time
    x = []
    for index, row in params.iterrows():
        if distribution.lower() == 'normal':
            x += [rdm.normal(size=n, **kwargs)]
        if distribution.lower() == 'uniform':
            x += [rdm.uniform(size=n, **kwargs)]
    X = np.array(x)
    e = rdm.normal(loc=0., scale=1., size=n)

    # Since x is a n*p matrix, and coefficients is a p*1 matrix
    # we can use matrix multiplication to get the value of y for each
    # set of values x1, x2 .. xp
    # We need to transpose it to get a 1*n array from a n*1 matrix to use in the regression model
    y = X.T.dot(coeffs) + e + intercept

    if binary:
        y = [1 if y_i > 0 else 0 for y_i in y]
    
    df = pd.DataFrame(X.T)
    # Assign column names
    for col in df.columns:
        df.rename(columns = {col: 'X' + str(col)}, inplace=True)
    
    # Append Y
    df['Y'] = y
    #
    return df

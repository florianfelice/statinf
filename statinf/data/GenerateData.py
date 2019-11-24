import numpy as np
import pandas as pd

def generate_dataset(coeffs, n, std_dev, intercept=1.0):
    # We calculate the number of predictors, and create a coefficient matrix
    # With `p` rows and 1 column, for matrix multiplication
    p = len(coeffs)
    params = pd.DataFrame({'coeff': coeffs, 'std_dev': std_dev})
    # Similar as before, but with `n` rows and `p` columns this time
    x = []
    for index, row in params.iterrows():
        x += [np.random.normal(row.coeff, row.std_dev, n)]
    X = np.array(x)
    e = np.random.randn(n) * 1.0
    # Since x is a n*p matrix, and coefficients is a p*1 matrix
    # we can use matrix multiplication to get the value of y for each
    # set of values x1, x2 .. xp
    # We need to transpose it to get a 1*n array from a n*1 matrix to use in the regression model
    y = np.matmul(X.T, coeffs).transpose() + e + intercept
    
    df = pd.DataFrame(X.T)
    # Assign column names
    for col in df.columns:
        df.rename(columns = {col: 'X' + str(col)}, inplace=True)
    
    # Append Y
    df['Y'] = y
    #
    return df



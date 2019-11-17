import numpy as np
import pandas as pd

def generate_dataset(coeffs, n, std_dev, intercept = 0.0):
    # We calculate the number of predictors, and create a coefficient matrix
    # With `p` rows and 1 column, for matrix multiplication
    p = len(coeffs)
    coeff_mat = np.array(coeffs).reshape(p, 1)
    # Similar as before, but with `n` rows and `p` columns this time
    x = np.random.random_sample((n, p))* 100
    e = np.random.randn(n) * std_dev
    # Since x is a n*p matrix, and coefficients is a p*1 matrix
    # we can use matrix multiplication to get the value of y for each
    # set of values x1, x2 .. xp
    # We need to transpose it to get a 1*n array from a n*1 matrix to use in the regression model
    y = np.matmul(x, coeff_mat).transpose() + e + intercept
    
    df = pd.DataFrame(x)
    # Assign column names
    for col in df.columns:
        df.rename(columns = {col: 'X' + str(col)}, inplace=True)
    
    # Append Y
    df['Y'] = y[0]
    #
    return df



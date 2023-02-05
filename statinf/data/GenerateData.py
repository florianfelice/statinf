import numpy as np
import pandas as pd

def generate_dataset(coeffs, n, std_dev, intercept=0., distribution='normal', binary=False, seed=None, **kwargs):
    """Generate an artificial dataset

    :param coeffs: List of coefficients to use for computing the ouytput variable.
    :type coeffs: :obj:`list`
    :param n: Number of observations to generate.
    :type n: :obj:`int`
    :param std_dev: Standard deviation of the distribution.
    :type std_dev: :obj:`list`
    :param intercept: Value of the intercept to be set, defaults to 0.
    :type intercept: :obj:`float`, optional
    :param distribution: Type of distribution to use for generating the input variables, defaults to 'normal'. Can be:

        * `normal`: :math:`X \\sim \\mathcal{N}(\\mu, \\sigma^{2})`
        * `unirform`: :math:`X \\sim \\mathcal{U}_{[\\text{low}, \\text{high}]}`

    :type distribution: :obj:`str`, optional
    :param binary: Define if output is binary, defaults to False.
    :type binary: :obj:`bool`, optional
    :param seed: Random seed, defaults to None.
    :type seed: :obj:`int`, optional

    :param \\*\\*kwargs: Arguments to be passed in the distribution function. Can be:

        * `normal`: :obj:`loc` = :math:`\\mu` and :obj:`scale` = :math:`\\sigma^{2}`
        * `uniform`: :obj:`low` and :obj:`high`

    :return: DataFrame with output variable named as :obj:`Y` and covariates as :obj:`X0`, :obj:`X1`, :obj:`X2`, ...
    :rtype: :obj:`pandas.DataFrame`
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
        df.rename(columns={col: 'X' + str(col)}, inplace=True)

    # Append Y
    df.loc[:, 'Y'] = y

    return df

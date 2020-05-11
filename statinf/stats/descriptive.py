import numpy as np
from ..data import rankdata


# ##### Variance

def var(x, std=False, df=1):
    """Compute the variance of a variable.

    :param x: Input variable. Format can be :obj:`numpy.array`, :obj:`list` or :obj:`pandas.Series`.
    :type x: :obj:`numpy.array`
    :param std: Returns standard deviation, i.e. :math:`\\sqrt{\\mathbb{V}(\\mathbf{X})}`, defauls to False.
    :type std: :obj:`bool`, optional
    :param df: Degrees of freedom, defaults to 1.
    :type df: :obj:`int`, optional

    :formula: .. math:: \\mathbb{V}(\\mathbf{X}) = \\dfrac{1}{n - 1} \\sum_{i = 1}^{n} (X_{i} - \\bar{X})^{2}

    :example:

    >>> from statinf import stats
    >>> x = [0.023699, 0.021436, 0.0200109, 0.0202762, 0.0165271, 0.01027]
    >>> stats.var(x)
    ... 2.2492979044000003e-05

    :return: Variance.
    :rtype: :obj:`float`
    """
    x = np.asarray(x)
    n = len(x)
    x_bar = x.mean()
    s2 = ((x - x_bar) ** 2).sum() / (n-df)
    if std:
        return np.sqrt(s2)
    else:
        return s2


# ##### Covariance

def cov(x, y):
    """Compute the covariance between two variables.

    :param x: Input variable. Format can be :obj:`numpy.array`, :obj:`list` or :obj:`pandas.Series`.
    :type x: :obj:`numpy.array`
    :param y: Input variable. Format can be :obj:`numpy.array`, :obj:`list` or :obj:`pandas.Series`.
    :type y: :obj:`numpy.array`

    :formula: .. math:: Cov(\\mathbf{X}, \\mathbf{Y}) = \\dfrac{ \\sum_{i = 1}^{n} (X_{i} - \\bar{X}) (Y_{i} - \\bar{Y}) }{n - 1}

    :example:

    >>> from statinf import stats
    >>> x = [0.023699, 0.021436, 0.0200109, 0.0202762, 0.0165271, 0.01027]
    >>> y = [9.4228, 9.27951, 9.167963, 9.68820, 9.56490, 7.543]
    >>> stats.cov(x, y)
    ... 0.003047229298620001

    :reference: * DeGroot, M. H., & Schervish, M. J. (2012). Probability and statistics. Pearson Education.

    :return: Covariance value.
    :rtype: :obj:`float`
    """
    x = np.asarray(x)
    y = np.asarray(y)
    x - x.mean()
    y - y.mean()

    x_xbar = x - x.mean()
    y_ybar = y - y.mean()

    num = (x_xbar * y_ybar).sum()
    cov = num / (len(x) - 1)
    return cov


# ##### Pearson's correlation coefficient

def pearson(x, y):
    """Compute the Pearson's coefficient correlation

    :param x: Input variable. Format can be :obj:`numpy.array`, :obj:`list` or :obj:`pandas.Series`.
    :type x: :obj:`numpy.array`
    :param y: Input variable. Format can be :obj:`numpy.array`, :obj:`list` or :obj:`pandas.Series`.
    :type y: :obj:`numpy.array`

    :formula: .. math:: \\rho = \\dfrac{ Cov(X, Y) }{\\sigma_{X} \\sigma_{Y}}

        where :math:`\\sigma_{Z} = \\sqrt{\\mathbb{V}(Z)}`

    :example:

    >>> from statinf import stats
    >>> x = [0.023699, 0.021436, 0.0200109, 0.0202762, 0.0165271, 0.01027]
    >>> y = [9.4228, 9.27951, 9.167963, 9.68820, 9.56490, 7.543]
    >>> stats.pearson(x, y)
    ... 0.9750052703452801

    :reference: * DeGroot, M. H., & Schervish, M. J. (2012). Probability and statistics. Pearson Education.

    :return: Pearson's coefficient correlation.
    :rtype: :obj:`float`
    """
    x = np.asarray(x)
    y = np.asarray(y)
    cv = cov(x, y)
    stdx = x.std()
    stdy = y.std()
    return cv / (stdx * stdy)


# ##### Spearman's rank correlation coefficient

def spearman(x, y):
    """Spearman's rank correlation coefficient.

    :param x: Input variable. Format can be :obj:`numpy.array`, :obj:`list` or :obj:`pandas.Series`.
    :type x: :obj:`numpy.array`
    :param y: Input variable. Format can be :obj:`numpy.array`, :obj:`list` or :obj:`pandas.Series`.
    :type y: :obj:`numpy.array`

    :formula: .. math:: \\rho = 1 - \\dfrac{ 6 \\sum_{i=1}^{n} d^{2} }{ n (n^{2} - 1)}

    :example:

    >>> from statinf import stats
    >>> x = [0.023699, 0.021436, 0.0200109, 0.0202762, 0.0165271, 0.01027]
    >>> y = [9.4228, 9.27951, 9.167963, 9.68820, 9.56490, 7.543]
    >>> stats.spearman(x, y)
    ... 0.37142857142857144

    :return: Spearman's rank correlation coefficient.
    :rtype: :obj:`float`
    """
    n = len(x)
    rk_x = rankdata(x)
    rk_y = rankdata(y)
    d = (rk_y - rk_x) ** 2

    rho = 1 - (6 * d.sum()) / (n * (n ** 2 - 1))

    return rho

import numpy as np

def gaussian(X, mu, cov):
    """
        Returns the pdf of a Gaussian kernel

        :param X: Input data.
        :type X: :obj:`numpy.ndarray`
        :param mu: Mean of the gaussian distribution.
        :type mu: :obj:`numpy.ndarray`
        :param cov: Covariance matrix.
        :type cov: :obj:`numpy.ndarray`

        :formula:

            .. math:: \\dfrac{1}{(2 \\pi)^{\\frac{n}{2}} \\sqrt{\\det{\\sigma}}} e^{-\\dfrac{1}{2} \\dfrac{X'X}{\\sigma^{2}}}

        :references: * Murphy, K. P. (2012). Machine learning: a probabilistic perspective. MIT press.
    """
    X_centered = X - mu
    n = X.shape[1]
    _a = (1 / ((2 * np.pi) ** (n / 2) * np.linalg.det(cov) ** 0.5))
    _b = np.exp(-0.5 * np.dot(np.dot(X_centered, np.linalg.inv(cov)), X_centered.T))

    return np.diagonal(_a * _b).reshape(-1, 1)

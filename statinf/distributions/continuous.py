# import numpy as np
# import scipy
# # from pymc3.distributions.continuous import Continuous


# class Gaussian(Continuous):
#     def __init__(self, mu_=None, sigma_=None, *args, **kwargs):
#         """Class for the continuous Gaussian (Normal) distribution.

#         :param mu_: Location parameter, commonly known as mean and estimated by :math:`\\mathbf{E}(X)`, defaults to None.
#         :type mu_: :obj:`float`, optional
#         :param sigma_: Scale parameter, commonly known as standard deviation and estimated by :math:`\\sqrt{\\mathbf{V}(X)}`, defaults to None.
#         :type sigma_: :obj:`float`, optional
#         """
#         super(Gaussian, self).__init__(*args, **kwargs)
#         self.mu_ = mu_
#         self.sigma_ = sigma_

#     @staticmethod
#     def nll(mu_, sigma_, data):
#         """Negative log-likelihood function. Its form is given by:

#         .. math::

#             \\log \\mathcal{L}(\\mu, \\sigma) = \\frac{n}{2} \\log{2 \\ pi} + \\frac{n}{2} \\log{\\sigma^{2}} + \\frac{1}{2\\sigma^{2}} \\sum_{i=1}{n}{(x_{i} - \\mu)^{2}}

#         :param mu_: Location parameter, commonly known as mean: :math:`\\mathbf{E}(X) = \\mu`
#         :type mu_: :obj:`float`
#         :param sigma_: Scale parameter, commonly known as standard deviation: :math:`\\sqrt{\\mathbf{V}(X)} = \\sigma`
#         :type sigma_: :obj:`float`
#         :param data: Observations to measure
#         :type data: :obj:`numpy.array`

#         :return: Estimated negative log-likelihood
#         :rtype: :obj:`float`
#         """
#         ll = np.sum(scipy.stats.norm.logpdf(data, loc=mu_, scale=sigma_))

#         return -ll

#     def logp(self, value):
#         return self.nll(mu_=self.mu_, sigma_=self.sigma_, value=value)

#     def random(self, point=None, size=None):
#         return np.random.normal(loc=self.mu_, scale=self.sigma_, size=size)

#     def fit(self, data, init_params=[1, 0.], verbose=False):

#         res = scipy.optimize.minimize(
#             fun=lambda params, data: self.nll(mu_=params[0], sigma_=params[1], data=data),
#             x0=np.array(init_params),
#             args=(data,),
#             method='L-BFGS-B'
#         )
#         self.mu_ = res.x[0]
#         self.sigma_ = res.x[1]
#         self.nll = res.fun

#         if verbose:
#             print(res)

#         return {'mu_': self.mu_, 'sigma_': self.sigma_, 'nll': self.nll}

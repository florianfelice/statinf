import numpy as np
import pandas as pd
import math
import warnings
import scipy.stats as scp
import matplotlib.pyplot as plt

from ..ml.initializations import init_params
from ..ml.activations import logit

from ..data.ProcessData import parse_formula
from ..misc import ConvergenceWarning, summary, get_significance
# from .glm_test import GLM

# TODO: HC covariance
# TODO: check probit

class GLM:

    def __init__(self, formula, data, family='binomial', fit_intercept=False, initializer='zeros'):
        """Generalized Linear Model implemented with Newton-Raphson's method.

        :param formula: Regression formula to be run of the form :obj:`y ~ x1 + x2`. See :py:meth:`statinf.data.ProcessData.parse_formula`.
        :type formula: :obj:`str`
        :param data: Input data with Pandas format.
        :type data: :obj:`pandas.DataFrame`
        :param family: Family distribution of the dependent variable, defaults to binomial.
        :type family: :obj:`str`, optional
        :param fit_intercept: Used for adding intercept in the regression formula, defaults to False.
        :type fit_intercept: :obj:`bool`, optional
        :param initializer: Method for initializing the first parameters (see :py:meth:`statinf.ml.initializations`), defaults to 'zeros'.
        :type initializer: :obj:`str`, optional

        .. note:: The modules allows Binomial (for Logit) and Gaussian (for Probit) distributions.
            It will soon be extended to other distributions.
        """
        # Parse the formula
        self.df, self.X_col, self.Y_col = parse_formula(data=data, formula=formula, check_values=True, return_all=True)
        self.formula        = formula
        self.fit_intercept  = fit_intercept
        self.n              = self.df.shape[0]
        self.p              = len(self.X_col) + 1 if fit_intercept else len(self.X_col)
        self.X              = self.df[self.X_col].to_numpy()
        self.Y              = np.array(self.df[self.Y_col].values).reshape(self.n, 1)
        self.cov_type       = 'nonrobust'
        self.family         = family
        self.log_likelihood = np.Inf
        self.log_likelihood_hist = []
        self.gradient_hist  = []
        self.variance_hist  = []
        self.coefs_hist     = []
        self.hessian_hist   = []

        # Initialize parameters
        self.beta = init_params(rows=self.p, cols=1, method=initializer, tensor=False)

    def _get_X(self, new_data=None):
        """Function to process data (and append unit variable if fit_intercept)

        :param new_data: Data to transform, defaults to None (gets train data provided in __init__).
        :type new_data: :obj:`pandas.DataFrame`, optional

        :return: Formatted data
        :rtype: :obj:`numpy.ndarray`
        """
        if new_data is None:
            X = self.X
        else:
            # If new_data, parse the formula to get transformations
            df = parse_formula(data=new_data, formula=self.formula, check_values=True)
            X = df[self.X_col].to_numpy()

        # If fit_intercept, append unit variable
        if self.fit_intercept:
            X = np.hstack((np.ones((self.n, 1), dtype=self.X.dtype), self.X))

        return X

    def _prob(self, X):
        """ Prob
        """
        if self.family == 'binomial':
            prob = logit(X, weights=self.beta)  # (X.dot(self.beta))
        elif self.family == 'gaussian':
            prob = scp.norm.cdf(X.dot(self.beta), loc=0, scale=1)
        else:
            raise ValueError('Family distribution is not valid.')

        return prob

    def _log_likelihood(self, X=None, Y=None):
        """
        Compute Log-Likelihood
        """
        X_data = self._get_X() if X is None else X
        Y_data = self.Y if Y is None else Y

        # Individual log-likelihood
        log_likelihood_id = np.array(Y_data * np.log(self._prob(X_data)) + (1 - Y_data) * np.log(1 - self._prob(X_data)))
        # Overall log-likelihood
        log_likelihood = log_likelihood_id.sum()
        return log_likelihood

    def _gradient(self):
        """
        Gradient (Overall) not the score function
        """
        X = self._get_X()
        gradient = X.T.dot((self._prob(X) - self.Y)) / self.n  # gradient of the likelihood
        return gradient

    def _hessian(self):
        """
        Hessian - Fisher Information
        """
        X = self._get_X()
        pi = self._prob(X)
        S = np.multiply(pi, (1 - pi)) * np.identity(self.n)
        hessian = (1 / self.n) * X.T.dot(S).dot(X)
        return hessian

    def _sandwich(self):
        X = self._get_X()
        pi = self._prob(X)
        res_2 = (self.Y - pi)**2 * np.identity(self.n)
        XX_1 = np.linalg.inv(X.T.dot(X))
        Xres_2X = X.T.dot(res_2.dot(X))
        _sand = XX_1.dot(Xres_2X).dot(XX_1)
        return _sand

    def variance(self, cov_type='nonrobust'):
        """Compute the covariance matrix for the fitted model.

        :param cov_type: Type of the covariance matrix, defaults to nonrobust.
        :type cov_type: :obj:`str`, optional

        :formulae: * **Non-robust covariance matrix**:

            .. math:: \\sigma_{\\beta} = {(X'SX)}^{-1}

            * **Sandwich covariance matrix**:

            .. math:: \\sigma_{\\beta} = {(X'X)}^{-1} (X'\\hat{S}X) {(X'X)}^{-1}

            with :math:`S = \\text{diag}(\\hat{p}_i(1 - \\hat{p}_i))` and
            :math:`\\hat{S} = \\text{diag} \\left( (Y_i - \\hat{p}_i)^{2} \\right)`

        .. note:: Only non-robust covariance matrix is currently available.
            Sandwich estimate and :math:`HC0`, :math:`HC1`, :math:`HC2`, :math:`HC3` will soon be implemented.

        :return: Fisher information matrix
        :rtype: :obj:`numpy.array`
        """
        self.cov_type = cov_type

        if self.cov_type == 'nonrobust':
            # Uses Fisher information matrix
            _variance = np.linalg.inv(self._hessian())
        # elif self.cov_type == 'sandwich':
        #     _variance = self._sandwich()
        else:
            error_msg = f"""Value for cov_type not valid. Got '{self.cov_type}'.
            Please chose 'nonrobust'.
            """
            raise ValueError(error_msg)
        return np.array(_variance)

    def _std_errors(self):
        """Compute the covariance matrix used for standard errors
        """
        _variance = self.variance()
        _std = np.sqrt((1 / self.n) * np.diag(_variance))
        return np.array(_std).reshape(self.p, 1)

    def fit(self, maxit=15, cov_type='nonrobust', improvement_threshold=0.0005, keep_hist=True, plot=False):
        """Fits the GLM regression model using Newton-Raphson method.

        :param maxit: Maximum number of iterations, defaults to 15.
        :type maxit: :obj:`int`, optional
        :param cov_type: Type of the covariance matrix (non-robust or sandwich), defaults to nonrobust.
        :type cov_type: :obj:`str`, optional
        :param improvement_threshold: Threshold from which we consider the likelihood improved, defaults to 0.0005.
        :type improvement_threshold: :obj:`float`, optional
        :param keep_hist: Keeps training history (gradients, hessian, etc...), can be turned off for saving memory, defaults to True.
        :type keep_hist: :obj:`bool`, optional
        :param plot: Plots evolution of log-likelihood through the different iterations (requires :obj:`keep_hist = True`), defaults to False.
        :type plot: :obj:`bool`, optional

        :formulae: * **Log-likelihood**:

            .. math:: l(\\beta) = y_{i} \\log \\left[ G(\\mathbf{x_i} \\beta) \\right] + (1 - y_{i}) \\log \\left[1 - G(\\mathbf{x_i} \\beta) \\right]

            * **Newton's method**:

            .. math:: \\hat{\\beta}_{s+1} = \\hat{\\beta}_{s} - H^{-1}_{s} G_{s}

            with

                .. math:: G_{s} = \\dfrac{\\partial}{\\partial \\beta_{s}} l(\\beta) = \\sum_{i=1}^{N} x_i (y_{i} - G(x_{i} \\beta))

            and

                .. math:: H_{s} = \\dfrac{\\partial^2}{\\partial \\beta_{s}^2} l(\\beta) = X'S_{s}X

            where :math:`S = \\text{diag} \\left( (Y_i - \\hat{p}_i)^{2} \\right)` and
            :math:`G` denotes the link function (:py:meth:`statinf.ml.activations.sigmoid` for logit or gaussian c.d.f for probit).

        :references: * Wooldridge, J. M. (2010). Econometric analysis of cross section and panel data.
            * Cameron, A. C., & Trivedi, P. K. (2005). Microeconometrics: methods and applications. Cambridge university press.
            * McCullagh, P. (2018). Generalized linear models. Routledge.
        """
        self.cov_type = cov_type.lower()
        self.it = 0
        diff = np.inf
        old_log_like = np.inf

        while (self.it < maxit) and (np.abs(diff) >= improvement_threshold):
            self.it      += 1
            log_like      = self._log_likelihood()
            gradient      = self._gradient()
            variance      = self.variance(cov_type=cov_type.lower())
            self.beta    -= variance.dot(gradient)
            diff          = old_log_like - log_like
            old_log_like  = log_like

            if keep_hist:
                self.log_likelihood = log_like
                self.log_likelihood_hist.append(log_like)
                self.gradient_hist.append(gradient)
                self.variance_hist.append(variance)

        # Check whether convergence was reached
        self.converged = np.abs(diff) < improvement_threshold
        if (self.converged is False) or (math.isnan(log_like)):
            warnings.warn('Model did not converge.', ConvergenceWarning)

        if plot:
            # Plot the training and test loss
            plt.title('GLM training history', loc='center')
            plt.plot(self.log_likelihood_hist, label='Log-likelihood')
            # plt.plot(self.test_losses, label='Test loss')
            plt.legend()
            plt.show()

    def r_squared(self):
        """Mc Fadden's pseudo-:math:`R^{2}` -- Goodness of fit

        :formula: .. math:: R^{2} = 1 - \\dfrac{LL(\\hat{\\beta})}{LL(\\bar{Y})}

        :return: Goodness of fit.
        :rtype: :obj:`float`
        """
        null_mod = self.__class__(formula=f'{self.Y_col} ~ 1', data=self.df)
        _r2 = 1 - self._log_likelihood() / null_mod._log_likelihood()
        self.r_squared = _r2
        return _r2

    def adjusted_r_squared(self):
        """Mc Fadden's pseudo-:math:`R^{2}` adjusted -- Adjusted goodness of fit

        :formula: .. math:: R^{2}_{adj} = 1 - \\dfrac{LL(\\hat{\\beta}) - p}{LL(\\bar{Y})}

        :return: Adjusted goodness of fit.
        :rtype: :obj:`float`
        """
        null_mod = self.__class__(formula=f'{self.Y_col} ~ 1', data=self.df)
        ar2 = 1 - (self._log_likelihood() - self.p) / null_mod._log_likelihood()
        self.adjusted_r_squared = ar2
        return ar2

    def summary(self, return_df=False):
        """Statistical summary for GLM model

        :param return_df: Return the summary as a Pandas DataFrame, else returns a string, defaults to False.
        :type return_df: :obj:`bool`

        :formulae: * **Mc Fadden's** :math:`R^{2}`:

            .. math:: R^{2} = 1 - \\dfrac{LL(\\hat{\\beta})}{LL(\\bar{Y})}

            where :math:`LL` represents the log-likelihood.

        :references: * Student. (1908). The probable error of a mean. Biometrika, 1-25.
            * Wooldridge, J. M. (2016). `Introductory econometrics: A modern approach
              <https://faculty.arts.ubc.ca/nfortin/econ495/IntroductoryEconometrics_AModernApproach_FourthEdition_Jeffrey_Wooldridge.pdf>`_.
              Nelson Education.
            * Agresti, A. (2003). Categorical data analysis (Vol. 482). John Wiley & Sons.
            * Murphy, K. P. (2012). Machine learning: a probabilistic perspective. MIT press.

        :return: Model's summary.
        :rtype: :obj:`pandas.DataFrame` or :obj:`str`
        """
        # Fit model if not already done
        if self.log_likelihood == []:
            self.fit()

        # Initialize
        betas = self.beta
        X = self._get_X()

        t_values = betas / self._std_errors()

        p_values = [round(2 * (1 - scp.t.cdf(np.abs(i[0]), (len(X) - 1))), 5) for i in t_values]
        z = scp.norm.ppf(0.975)

        summary_df = pd.DataFrame()
        summary_df["Variables"] = ['(Intercept)'] + self.X_col if self.fit_intercept else self.X_col
        summary_df["Coefficients"] = betas
        summary_df["Standard Errors"] = self._std_errors()
        summary_df["t-values"] = t_values
        summary_df["Probabilities"] = p_values
        summary_df["Significance"] = summary_df["Probabilities"].map(lambda x: get_significance(x))
        summary_df["CI_lo"] = summary_df["Coefficients"] - z * summary_df["Standard Errors"]
        summary_df["CI_hi"] = summary_df["Coefficients"] + z * summary_df["Standard Errors"]

        # Null model for null log likelihood
        null_mod = self.__class__(formula=f'{self.Y_col} ~ 1', data=self.df)

        # Key metrics for summary
        _ll = round(self._log_likelihood(), 2)
        nll = round(null_mod._log_likelihood(), 2)
        _r2 = round(1 - _ll / nll, 5)
        ar2 = round(1 - (_ll - self.p) / nll, 5)
        _n_ = self.n
        _p_ = self.p
        _it = self.it
        _cv = '     True' if self.converged else "    False"
        _ct = ' ' + self.cov_type

        # Likelihood Ratio test
        G = 2 * (_ll - nll)
        lrp = round(scp.chi2.sf(G, self.p), 3)

        if return_df:
            return(summary_df)
        else:
            max_var = np.max([len(v) for v in summary_df.Variables])

            add_sp = ' ' * np.max([max_var - 17, 0])
            add_sep = '=' * np.max([max_var - 17, 0])
            space = np.max([max_var, 17])

            summ = f"============================================================================================================={add_sep}\n"
            summ += f'|                                              Logit summary                                                {add_sp}|\n'
            summ += f"============================================================================================================={add_sep}\n"
            summ += f"| McFadden's R²          =              {_r2:10} | McFadden's R² Adj.         =                {ar2:10} {add_sp}|\n"
            summ += f"| Log-Likelihood         =              {_ll:10} | Null Log-Likelihood        =                {nll:10} {add_sp}|\n"
            summ += f"| LR test p-value        =              {lrp:10} | Covariance                 =                {_ct:10} {add_sp}|\n"
            summ += f"| n                      =              {_n_:10} | p                          =                {_p_:10} {add_sp}|\n"
            summ += f"| Iterations             =              {_it:10} | Convergence                =                 {_cv:5} {add_sp}|\n"
            summ += summary(summary_df)
            return(summ)

    def predict(self, new_data, return_proba=False):
        """Predicted :math:`\\hat{Y}` values for for a new dataset

        :param new_data: New data to evaluate with pandas data-frame format.
        :type new_data: :obj:`pandas.DataFrame`
        :param return_proba: Whether to return probabilities or binary values, defaults to False
        :type return_proba: :obj:`bool`, optional

        :formula: .. math:: f(X) = \\dfrac{1}{e^{-\\beta X} + 1}

        :return: Predictions :math:`\\hat{Y}`
        :rtype: :obj:`numpy.ndarray`
        """
        # Fit model if not already done
        if self.log_likelihood == []:
            self.fit()

        # Tranform data
        new_X = self._get_X(new_data=new_data)

        pred = self._prob(new_X)
        if return_proba:
            return np.asarray([x[0] for x in pred])
        else:
            return np.asarray([1 if x[0] > 0.5 else 0 for x in pred])

    def partial_effects(self, variables, new_data=None, average=False):
        """Computes Partial and Average Partial Effects (APE).

        :param variables: List of variables for which to compute the PE/APE.
        :type variables: :obj:`list`
        :param new_data: Data to use for computations, optional, defaults to None (uses training set).
        :type new_data: :obj:`pandas.DataFrame`
        :param average: Whether to compute Average Partial Effects or not, defaults to False.
        :type average: :obj:`bool`, optional

        :formula: .. math:: PE(X_{i}) = \\beta_{i} \\dfrac{e^{-\\beta X}}{(e^{-\\beta X} + 1)^{2}}

        if :obj:`average = True`, :math:`APE = \\bar{PE}`

        :raises TypeError: If argument :obj:`variables` is neither :obj:`str` nor :obj:`list`.

        :return: Dictionnary including Partial Effects (PE) or Average Partial Effects (APE).
        :rtype: :obj:`dict`
        """

        assert self.family == 'binomial', 'APE module only available for Logit model.'

        # Fit model if not already done
        if self.log_likelihood == []:
            self.fit()

        # Get data
        X = self._get_X(new_data)

        # Get coefficients as a DataFrame (easier to handle for fitlering)
        if self.fit_intercept:
            # Append the name of the intercept
            cols = ['(Intercept)'] + self.X_col
        else:
            cols = self.X_col

        coefs = pd.DataFrame({'Coefficients': [x[0] for x in self.beta]}, index=cols)

        # Compute exp(bX)
        exp_bx = np.exp(-X.dot(self.beta))

        # Check type for argument variables
        if type(variables) == str:
            var = [variables]
        elif type(variables) == list:
            var = variables
        else:
            raise TypeError('Type for argument variables is not valid.')

        # Initialize the dictionnary for output
        pes = {}

        # Compute PEs (or APEs) for each variable
        for v in var:
            b = coefs.Coefficients[v].min()
            pe = b * exp_bx / ((exp_bx + 1)**2)
            pe_out = pe.mean() if average else pe

            pes.update({v: pe_out})

        return pes

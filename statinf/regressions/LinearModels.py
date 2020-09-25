import numpy as np
from scipy import stats as scp
import pandas as pd
import math
from scipy.stats import norm

from ..data.ProcessData import parse_formula
from ..misc import summary, get_significance

# TODO: Add Log-Likehood + AIC + BIC
# TODO: Add dask for GPU usage


class OLS:

    def __init__(self, formula, data, fit_intercept=False):
        """Ordinary Least Squares regression

        :param formula: Regression formula to be run of the form :obj:`y ~ x1 + x2`. See :func:`~parse_formula` in `ProcessData <../../data/process.html#statinf.data.ProcessData.parse_formula>`_
        :type formula: :obj:`str`
        :param data: Input data with Pandas format.
        :type data: :obj:`pandas.DataFrame`
        :param fit_intercept: Used for adding intercept in the regression formula, defaults to False.
        :type fit_intercept: :obj:`bool`, optional
        """

        super(OLS, self).__init__()
        # Parse formula
        df, self.X_col, self.Y_col = parse_formula(data=data, formula=formula, check_values=True, return_all=True)
        # self.no_space_formula = formula.replace(' ', '')
        # self.Y_col = self.no_space_formula.split('~')[0]
        # self.X_col = self.no_space_formula.split('~')[1].split('+')
        self.formula = formula
        # Subset X
        self.X = df[self.X_col].to_numpy()
        # Target variable
        self.Y = df[self.Y_col].to_numpy()
        # Degrees of freedom of the population
        self.dft = self.X.shape[0]
        # Degree of freedom of the residuals
        self.dfe = self.X.shape[0] - self.X.shape[1]
        # Size of the population
        self.n = self.X.shape[0]
        # Number of explanatory variables / estimates
        self.p = self.X.shape[1]
        # Use intercept or only explanatory variables
        self.fit_intercept = fit_intercept
        # Beta standard errors for confidence intervals
        self.std_err = None

    def _get_X(self):
        if self.fit_intercept:
            return(np.hstack((np.ones((self.n, 1), dtype=self.X.dtype), self.X)))
        else:
            return(self.X)

    def get_betas(self):
        """Computes the estimates for each explanatory variable

        :formula: .. math:: \\beta = (X'X)^{-1} X'Y

        :return: Estimated coefficients
        :rtype: :obj:`numpy.array`
        """
        XtX = self._get_X().T.dot(self._get_X())
        XtX_1 = np.linalg.inv(XtX)
        XtY = self._get_X().T.dot(self.Y)
        beta = XtX_1.dot(XtY)
        return(beta)

    def fitted_values(self):
        """Computes the estimated values of Y

        :formula: .. math:: \\hat{Y} = X \\beta

        :return: Fitted values for Y.
        :rtype: :obj:`numpy.array`
        """
        betas = self.get_betas()
        Y_hat = np.zeros(len(self.Y))
        for i in range(len(betas)):
            Y_hat += (betas[i] * self._get_X().T[i])
        return(Y_hat)

    def _get_error(self):
        """Compute the error term/residuals

        :formula: .. math:: \\epsilon = Y - \\hat{Y}

        :return: Estimated residual term.
        :rtype: :obj:`numpy.array`
        """
        res = self.Y - self.fitted_values()
        return(res)

    def rss(self):
        """Residual Sum of Squares

        :formula: .. math:: RSS = \\sum_{i=1}^{n} (y_{i} - \\hat{y}_{i})^{2}

            where :math:`y_{i}` denotes the true/observed value of :math:`y` for individual :math:`i` and :math:`\\hat{y}_{i}` denotes the predicted value of :math:`y` for individual :math:`i`.

        :return: Residual Sum of Squares.
        :rtype: :obj:`float`
        """
        return((self._get_error()**2).sum())

    def tss(self):
        """Total Sum of Squares

        :formula: .. math:: TSS = \\sum_{i=1}^{n} (y_{i} - \\bar{y})^{2}

            where :math:`y_{i}` denotes the true/observed value of :math:`y` for individual :math:`i` and :math:`\\bar{y}_{i}` denotes the average value of :math:`y`.

        :return: Total Sum of Squares.
        :rtype: :obj:`float`
        """

        y_bar = self.Y.mean()
        total_squared = (self.Y - y_bar) ** 2
        return(total_squared.sum())

    def r_squared(self):
        """:math:`R^{2}` -- Goodness of fit

        :formula: .. math:: R^{2} = 1 - \\dfrac{RSS}{TSS}

        :return: Goodness of fit.
        :rtype: :obj:`float`
        """

        return(1 - self.rss() / self.tss())

    def adjusted_r_squared(self):
        """Adjusted-:math:`R^{2}` -- Goodness of fit

        :formula: .. math:: R^{2}_{adj} = 1 - (1 - R^{2}) \\dfrac{n - 1}{n - p - 1}

            where :math:`p` denotes the number of estimates (i.e. explanatory variables) and :math:`n` the sample size

        :references: Theil, Henri (1961). Economic Forecasts and Policy.

        :return: Adjusted goodness of fit.
        :rtype: :obj:`float`
        """

        adj_r_2 = 1 - (1 - self.r_squared()) * (self.n - 1) / (self.n - self.p - 1)
        return(adj_r_2)

    def _fisher(self):
        """Fisher test

        :formula: .. math:: \\mathcal{F} = \\dfrac{TSS - RSS}{\\frac{RSS}{n - p}}

            where :math:`p` denotes the number of estimates (i.e. explanatory variables) and :math:`n` the sample size

        :references: Shen, Q., & Faraway, J. (2004). `An F test for linear models with functional responses <https://www.jstor.org/stable/24307230>`_. Statistica Sinica, 1239-1257.

        :return: Value of the :math:`\\mathcal{F}`-statistic.
        :rtype: :obj:`float`
        """
        MSE = (self.tss() - self.rss()) / (self.p - 1)
        MSR = self.rss() / self.dfe
        return(MSE / MSR)

    def _std_err(self):
        """Standard error function

        :formula:

            .. math:: \\mathbb{V}(\\beta) = \\sigma^{2} X'X

            where :math:`\\sigma^{2} = \\frac{RSS}{n - p -1}`
        
        :return: Standard error of the estimates.
        :rtype: :obj:`numpy.array`
        """
        X = self._get_X()

        sigma_2 = (sum((self._get_error())**2)) / (len(X) - len(X[0]))
        variance_beta = sigma_2 * (np.linalg.inv(np.dot(X.T, X)).diagonal())
        self.std_err = np.sqrt(variance_beta)
        return self.std_err
    
    def _loglikelihood(self):
        """Standard error function

        :formula:

            .. math:: l = \\dfrac{n}{2} \\log{2 \\pi} - \\dfrac{n}{2} \\log{\\dfrac{RSS}{n}} - \\dfrac{n}{2}
        
        :return: Log-likelihood.
        :rtype: :obj:`float`
        """
        _ll = - (self.n / 2) * np.log(2 * math.pi) - (self.n / 2) * np.log(self.rss()/self.n) - (self.n / 2)
        self.loglikelihood = _ll
        return _ll
    
    def _aic(self, metric='aic'):
        """Akaike Information Criterion and Bayesian Information Criterion

        :param metric: Define what metric to return, defaults to 'aic'.
        :type return_df: :obj:`str`

        :formula: * AIC:

            .. math:: AIC = - 2 \\log{L(\\theta)} + 2 p

            * BIC:

            .. math:: AIC = - 2 \\log{L(\\theta)} + \\log{n} p

            where :math:`p` is the number of covariates and :math:`L` the likelihood function.

        :references: * Cameron, A. C., & Trivedi, P. K. (2009). Microeconometrics using stata (Vol. 5, p. 706). College Station, TX: Stata press.

        :return: Information criterion.
        :rtype: :obj:`float`
        """
        self.aic = -2 * self._loglikelihood() + 2 * self.p
        self.bic = -2 * self._loglikelihood() + np.log(self.n) * self.p
        
        if metric.lower() == 'bic':
            return self.bic
        else:
            return self.aic

    def summary(self, return_df=False):
        """Statistical summary for OLS

        :param return_df: Return the summary as a Pandas DataFrame, else returns a string, defaults to False.
        :type return_df: :obj:`bool`

        :formulae: * Fisher test:

            .. math:: \\mathcal{F} = \\dfrac{TSS - RSS}{\\frac{RSS}{n - p}}

            where :math:`p` denotes the number of estimates (i.e. explanatory variables) and :math:`n` the sample size


            * Covariance matrix:

            .. math:: \\mathbb{V}(\\beta) = \\sigma^{2} X'X

            where :math:`\\sigma^{2} = \\frac{RSS}{n - p -1}`


            * Coefficients' significance:

            .. math:: p = 2 \\left( 1 - T_{n} \\left( \\dfrac{\\beta}{\\sqrt{\\mathbb{V}(\\beta)}} \\right) \\right)

            where :math:`T` denotes the Student cumulative distribution function (c.d.f.) with :math:`n` degrees of freedom


        :references: * Student. (1908). The probable error of a mean. Biometrika, 1-25.
            * Shen, Q., & Faraway, J. (2004). `An F test for linear models with functional responses <https://www.jstor.org/stable/24307230>`_. Statistica Sinica, 1239-1257.
            * Wooldridge, J. M. (2016). `Introductory econometrics: A modern approach <https://faculty.arts.ubc.ca/nfortin/econ495/IntroductoryEconometrics_AModernApproach_FourthEdition_Jeffrey_Wooldridge.pdf>`_. Nelson Education.
            * Cameron, A. C., & Trivedi, P. K. (2009). Microeconometrics using stata (Vol. 5, p. 706). College Station, TX: Stata press.

        :return: Model's summary.
        :rtype: :obj:`pandas.DataFrame` or :obj:`str`
        """
        # Initialize
        betas = self.get_betas()
        X = self._get_X()

        self.std_err = self._std_err()
        t_values = betas / self.std_err

        p_values = [2 * (1 - scp.t.cdf(np.abs(i), (len(X) - 1))) for i in t_values]
        z = norm.ppf(0.975)

        summary_df = pd.DataFrame()
        summary_df["Variables"] = ['(Intercept)'] + self.X_col if self.fit_intercept else self.X_col
        summary_df["Coefficients"] = betas
        summary_df["Standard Errors"] = self.std_err
        summary_df["t-values"] = t_values
        summary_df["Probabilities"] = p_values
        summary_df["Significance"] = summary_df["Probabilities"].map(lambda x: get_significance(x))
        summary_df["CI_lo"] = summary_df["Coefficients"] - z * summary_df["Standard Errors"]
        summary_df["CI_hi"] = summary_df["Coefficients"] + z * summary_df["Standard Errors"]

        _r2 = round(self.r_squared(), 5)
        ar2 = round(self.adjusted_r_squared(), 5)
        _n_ = self.n
        _p_ = self.p
        #
        fis = round(self._fisher(), 3)
        llf = round(self._loglikelihood(), 3)
        aic = round(self._aic(), 3)
        
        #
        if return_df:
            return(summary_df)
        else:
            max_var = np.max([len(v) for v in summary_df.Variables])

            add_sp = ' ' * np.max([max_var - 17, 0])
            add_sep = '=' * np.max([max_var - 17, 0])

            summ = f"============================================================================================================={add_sep}\n"
            summ += f'|                                                OLS summary                                                {add_sp}|\n'
            summ += f"============================================================================================================={add_sep}\n"
            summ += f"| n                   =                  {_n_:10} | p                 =                        {_p_:10} {add_sp}|\n"
            summ += f"| R²                  =                  {_r2:10} | R² Adj.           =                        {ar2:10} {add_sp}|\n"
            summ += f"| Log-likelihood      =             {llf:15} | AIC               =                        {aic:10} {add_sp}|\n"
            summ += f"| Fisher value        =             {fis:15} |                                                       {add_sp}|\n"
            summ += summary(summary_df)
            return(summ)

    def predict(self, new_data, conf_level=None):
        """Predicted :math:`\\hat{Y}` values for for a new dataset

        :param new_data: New data to evaluate with pandas data-frame format.
        :type new_data: :obj:`pandas.DataFrame`
        :param conf_level: Level of the confidence interval, defaults to None.
        :type conf_level: :obj:`float`

        :formulae: .. math:: \\hat{Y} = X \\hat{\\beta}

        The confidence interval is computed as:

            .. math:: \\left[ \\hat{Y} \\pm z_{1 - \\frac{\\alpha}{2}} \\dfrac{\\sigma}{\\sqrt{n - 1}} \\right]

        :return: Predictions :math:`\\hat{Y}`
        :rtype: :obj:`numpy.array`
        """
        df = parse_formula(data=new_data, formula=self.formula, check_values=True)
        X_array = df[self.X_col].to_numpy()
        if self.fit_intercept:
            new_X = np.hstack((np.ones((df.shape[0], 1), dtype=X_array.dtype), X_array))
        else:
            new_X = X_array

        y_pred = np.dot(new_X, self.get_betas())

        # No CI required
        if conf_level is None:
            pred = y_pred
        else:
            assert (conf_level > 0.) & (conf_level < 1.), "Your confidence level needs to be between 0 and 1."
            # User needs CI
            if self.std_err is None:
                _ = self.summary(return_df=True)

            quant_order = 1 - ((1 - conf_level) / 2)
            cv = scp.norm.ppf(quant_order)
            pred = pd.DataFrame({'Prediction': y_pred,
                                 'LowerBound': y_pred - cv * (self.std_err.sum() / np.sqrt(self.n - 1)),
                                 'UpperBound': y_pred + cv * (self.std_err.sum() / np.sqrt(self.n - 1))})

        return pred

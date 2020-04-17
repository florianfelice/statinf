import numpy as np
from scipy import stats
import pandas as pd


#TODO: Add Log-Likehood + AIC + BIC
#TODO: Add dask for GPU usage


class OLS:
    
    def __init__(self, formula, data, fit_intercept=True):
        """Ordinary Least Squares regression
        
        :param formula:  Regression formula to be run of the form :obj:`y ~ x1 + x2`.
        :type formula: :obj:`str`
        :param data: Input data with Pandas format.
        :type data: :obj:`pandas.DataFrame`
        :param fit_intercept: Used for adding intercept in the regression formula, defaults to True.
        :type fit_intercept: :obj:`bool`, optional
        """

        super(OLS, self).__init__()
        # Parse formula
        self.no_space_formula = formula.replace(' ', '')
        self.Y_col = self.no_space_formula.split('~')[0]
        self.X_col = self.no_space_formula.split('~')[1].split('+')
        # Subset X
        self.X = data[self.X_col].to_numpy()
        # Target variable
        self.Y = data[self.Y_col].to_numpy()
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

        :formula: .. math:: R^{2} = 1 - \dfrac{RSS}{TSS}

        :return: Goodness of fit.
        :rtype: :obj:`float`
        """

        return(1 - self.rss()/self.tss())

    def adjusted_r_squared(self):
        """Adjusted-:math:`R^{2}` -- Goodness of fit

        :formula: .. math:: R^{2}_{adj} = 1 - (1 - R^{2}) \dfrac{n - 1}{n - p - 1}

            where :math:`p` denotes the number of estimates (i.e. explanatory variables) and :math:`n` the sample size

        :references: Theil, Henri (1961). Economic Forecasts and Policy.

        :return: Adjusted goodness of fit.
        :rtype: :obj:`float`
        """

        adj_r_2 = 1 - (1 - self.r_squared()) * (self.n - 1) / (self.n - self.p - 1)
        return(adj_r_2)
    
    def _fisher(self):
        """Fisher test

        :formula: .. math:: \\mathcal{F} = \dfrac{TSS - RSS}{\\frac{RSS}{n - p}}

            where :math:`p` denotes the number of estimates (i.e. explanatory variables) and :math:`n` the sample size

        :references: Shen, Q., & Faraway, J. (2004). `An F test for linear models with functional responses <https://www.jstor.org/stable/24307230>`_. Statistica Sinica, 1239-1257.

        :return: Value of the :math:`\\mathcal{F}`-statistic.
        :rtype: :obj:`float`
        """
        MSE = (self.tss() - self.rss()) / (self.p - 1)
        MSR = self.rss() / self.dfe
        return(MSE/MSR)
    
    """
        Returns statistics summary for estimates
        
        Formula
        -------
        The p-values are computes as:
        p_value = 2 * (1 - T_n(t_value))
        
         * T denotes the Student Cumulative Distribution Function with n degrees of freedom
        
        The covariance matrix of beta is compute by:
        Var(b) = sigma_b**2 * (X'X)^-1
        
         * sigma_b denotes the standard deviation computed for a given estimate b
                
        Reference
        ---------
        
        """
    def summary(self, return_df=False):
        """Statistical summary for OLS
        
        :param return_df: Return the summary as a Pandas DataFrame, else print a string, defaults to False.
        :type return_df: :obj:`bool`

        :formulae: * Fisher test:

            .. math:: \\mathcal{F} = \dfrac{TSS - RSS}{\\frac{RSS}{n - p}}

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
        """
        # Initialize
        betas = self.get_betas()
        X = self._get_X()

        sigma_2 = (sum((self._get_error())**2))/(len(X) - len(X[0]))
        variance_beta = sigma_2 * (np.linalg.inv(np.dot(X.T, X)).diagonal())
        std_err_beta = np.sqrt(variance_beta)
        t_values = betas / std_err_beta
        
        p_values = [2 * (1 - stats.t.cdf(np.abs(i), (len(X)-1))) for i in t_values]
        
        summary_df = pd.DataFrame()
        summary_df["Variables"] = ['(Intercept)'] + self.X_col if self.fit_intercept else self.X_col
        summary_df["Coefficients"] = betas
        summary_df["Standard Errors"] = std_err_beta
        summary_df["t-values"] = t_values
        summary_df["Probabilities"] = p_values
        
        r2 = self.r_squared()
        adj_r2 = self.adjusted_r_squared()
        #
        fisher = self._fisher()
        #
        if return_df:
            return(summary_df)
        else:
            print('=========================================================================')
            print('                               OLS summary                               ')
            print('=========================================================================')
            print('| R² = {:.5f}                  | Adjusted-R² = {:.5f}'.format(r2, adj_r2))
            print('| n  = {:6}                   | p = {:5}'.format(self.n, self.p))
            print('| Fisher = {:.5f}                         '.format(fisher))
            print('=========================================================================')
            print(summary_df.to_string(index=False))

    def predict(self, new_data):
        """Predicted :math:`\\hat{Y}` values for for a new dataset
        
        :param new_data: New data to evaluate with pandas data-frame format.
        :type new_data: :obj:`pandas.DataFrame`
        
        :formula: .. math:: \\hat{Y} = X \\hat{\\beta}

        :return: Predictions
        :rtype: :obj:`numpy.array`
        """
        X_array = new_data[self.X_col].to_numpy()
        if self.fit_intercept:
            new_X = np.hstack((np.ones((new_data.shape[0], 1), dtype=X_array.dtype), X_array))
        else:
            new_X = X_array
        
        return np.dot(new_X, self.get_betas())

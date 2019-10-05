#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 17:13:38 2019

@author: Florian Felice
"""

import numpy as np
from scipy import stats
import pandas as pd


#TODO: qdd Fisher test
#TODO: Format Summary
#TODO: Add Log-Likehood (if relevant)
#TODO: Make model use pandas DataFrame as an option
#TODO: Add dask for GPU usage


def generate_dataset(coeffs, n, std_dev):
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
    y = np.matmul(x, coeff_mat).transpose() + e
    return x, y[0]


X, Y = generate_dataset([10, 5], 50, 100)


class OLS:
    """
    Class for Ordinary Least Squares model.
    Fits the data and returns coefficients and key metrics.
    
    Parameters
    ----------
    X : ndarray
        Explanatory variables.
    Y : ndarray
        Explained variables to be fitted.
    fit_intercept : bool
        Force the model to fit an intercept. Default is False.
    """
    X = []
    Y = []
    
    def __init__(self, X, Y, fit_intercept = False):
        super(OLS, self).__init__()
        self.X = X
        # Target variable
        self.Y = Y
        # Degrees of freedom of the population
        self.dft = X.shape[0] - 1
        # Degree of freedom of the residuals
        self.dfe = X.shape[0] - X.shape[1] - 1
        # Size of the population
        self.n = X.shape[0]
        # Number of explanatory variables / estimates
        self.p = X.shape[1]
        # Use intercept or only explanatory variables
        self.fit_intercept = fit_intercept
    
    def get_betas(self):
    	"""
        Computes the estimates for each explanatory variable
        
        Formula
        ----------
        b = (X'X)^-1 X'Y
        
         * X is a matrix for the explanatory variables
         * Y is a vector for the target variable
         * ' denotes the transpose operator
         * ^-1 denotes the inverse operator

        Returns
        ----------
        betas
        	The estimated coefficients.
        """
        XtX = self.X.T.dot(self.X)
        XtX_1 = np.linalg.inv(XtX)
        XtY = self.X.T.dot(self.Y)
        beta = XtX_1.dot(XtY)
        return(beta)
    
    def fitted_values(self):
    	"""
        Computes the estimated values of Y
        
        Formula
        ----------
        Y_hat = bX

        Returns
        ----------
        Y_hat
        	The fitted values of Y.
        """
        betas = self.get_betas()
        Y_hat = np.zeros(len(self.Y))
        for i in range(len(betas)):
            Y_hat += (betas[i] * self.X.T[i])
        return(Y_hat)
    
    def get_error(self):
    	"""
        Compute the error term/residuals
        
        Formula
        ----------
        res = Y - Y_hat

        Returns
        ----------
        res
        	The estimated residual term.
        """
        res = self.Y - self.fitted_values()
        return(res)
    
    def rss(self):
        """
        Computes Residual Sum of Squares
        
        Formula
        ----------
        RSS = Sum(y_i - y_hat_i)**2
        
         * y_i denotes the true/observed value of y for individual i
         * y_hat_i denotes the predicted value of y for individual i
        """
        return((self.get_error()**2).sum())
    
    def tss(self):
        """
        Computes Total Sum of Squares.
        
        Formula
        ----------
        TSS = Sum(Y_i - Y_bar)**2
        """
        y_bar = self.Y.mean()
        total_squared = (self.Y - y_bar) ** 2
        return(total_squared.sum())
    
    def r_squared(self):
        """
        Computes the standard R**2
        
        Formula
        ----------
        R**2 = 1 - RSS / TSS        
        """
        return(1 - self.rss()/self.tss())
    
    def adjusted_r_squared(self):
        """
        Computes Adjusted R**2.
        
        Formula
        ----------
        Adjusted R**2 = 1 - (1 - R**2) * (n - 1) / (n - p - 1)
        
         * p denotes the number of estimates (i.e. explanatory variables)
         * n denotes the sample size
        
        Reference
        ----------
        Theil, Henri (1961). Economic Forecasts and Policy
        """
        adj_r_2 = 1 - (1 - self.r_squared()) * (self.n - 1) / (self.n - self.p - 1)
        return(adj_r_2)
    
    def summary(self):
        """
        Returns statistics summary for estimates
        
        Formula
        ----------
        The p-values are computes as:
        p_value = 2 * (1 - T_n(sigma_b))
        
         * T denotes the Student Cumulative Distribution Function with n degrees of freedom
         * sigma_b denotes the standard deviation computed for a given estimate b
        
        The covariance matrix of beta is compute by:
        Var(b) = sigma_b**2 * (X'X)^-1
                
        Reference
        ----------
        Student. (1908). The probable error of a mean. Biometrika, 1-25.
        """
        # Initialize
        betas = self.get_betas()
        # Add intercept if fit is asked by user
        if self.fit_intercept:
            newX = np.append(np.ones((len(X),1)), X, axis=1)
        else:
            newX = self.X.copy()
        
        sigma_2 = (sum((self.get_error())**2))/(len(newX) - len(newX[0]))
        variance_beta = sigma_2 * (np.linalg.inv(np.dot(newX.T, newX)).diagonal())
        std_err_beta = np.sqrt(variance_beta)
        t_values = betas/ std_err_beta
        
        p_values =[2 * (1 - stats.t.cdf(np.abs(i), (len(newX)-1))) for i in t_values]
        
        summary_df = pd.DataFrame()
        summary_df["Coefficients"] = betas
        summary_df["Standard Errors"] = std_err_beta
        summary_df["t values"] = t_values
        summary_df["Probabilites"] = p_values
        
        return(summary_df)



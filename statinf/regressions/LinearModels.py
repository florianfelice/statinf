#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 17:13:38 2019

@author: Florian Felice
"""

import numpy as np
from scipy import stats
import pandas as pd

#TODO: Add Fisher test
#TODO: Add Log-Likehood + AIC + BIC
#TODO: Add dask for GPU usage


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
    
    def __init__(self, formula, data, fit_intercept=True):
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
    
    def get_X(self):
        if self.fit_intercept:
            return(np.hstack((np.ones((self.n, 1), dtype=self.X.dtype), self.X)))
        else:
            return(self.X)
    
    def get_betas(self):
        """
        Computes the estimates for each explanatory variable
        
        Formula
        -------
        b = (X'X)^-1 X'Y
        
         * X is a matrix for the explanatory variables
         * Y is a vector for the target variable
         * ' denotes the transpose operator
         * ^-1 denotes the inverse operator
        
        Returns
        -------
        betas
            The estimated coefficients.
        """
        XtX = self.get_X().T.dot(self.get_X())
        XtX_1 = np.linalg.inv(XtX)
        XtY = self.get_X().T.dot(self.Y)
        beta = XtX_1.dot(XtY)
        return(beta)
    
    def fitted_values(self):
        """
        Computes the estimated values of Y
        
        Formula
        -------
        Y_hat = bX
        
        Returns
        -------
        Y_hat
            The fitted values of Y.
        """
        betas = self.get_betas()
        Y_hat = np.zeros(len(self.Y))
        for i in range(len(betas)):
            Y_hat += (betas[i] * self.get_X().T[i])
        return(Y_hat)
    
    def get_error(self):
        """
        Compute the error term/residuals
        
        Formula
        -------
        res = Y - Y_hat
        
        Returns
        -------
        res
            The estimated residual term.
        """
        res = self.Y - self.fitted_values()
        return(res)
    
    def rss(self):
        """
        Computes Residual Sum of Squares
        
        Formula
        -------
        RSS = Sum(y_i - y_hat_i)**2
        
         * y_i denotes the true/observed value of y for individual i
         * y_hat_i denotes the predicted value of y for individual i
        """
        return((self.get_error()**2).sum())
    
    def tss(self):
        """
        Computes Total Sum of Squares.
        
        Formula
        -------
        TSS = Sum(Y_i - Y_bar)**2
        """
        y_bar = self.Y.mean()
        total_squared = (self.Y - y_bar) ** 2
        return(total_squared.sum())
    
    def r_squared(self):
        """
        Computes the standard R**2
        
        Formula
        -------
        R**2 = 1 - RSS / TSS        
        """
        return(1 - self.rss()/self.tss())
    
    def adjusted_r_squared(self):
        """
        Computes Adjusted R**2.
        
        Formula
        -------
        Adjusted R**2 = 1 - (1 - R**2) * (n - 1) / (n - p - 1)
        
         * p denotes the number of estimates (i.e. explanatory variables)
         * n denotes the sample size
        
        Reference
        ---------
        Theil, Henri (1961). Economic Forecasts and Policy
        """
        adj_r_2 = 1 - (1 - self.r_squared()) * (self.n - 1) / (self.n - self.p - 1)
        return(adj_r_2)
    
    def fisher(self):
        """
        """
        MSE = (self.tss() - self.rss()) / (self.p - 1)
        MSR = self.rss() / self.dfe
        return(MSE/MSR)
    
    def summary(self):
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
        Student. (1908). The probable error of a mean. Biometrika, 1-25.
        """
        # Initialize
        betas = self.get_betas()
        # Add intercept if fit is asked by user
        
        sigma_2 = (sum((self.get_error())**2))/(len(self.get_X()) - len(self.get_X()[0]))
        variance_beta = sigma_2 * (np.linalg.inv(np.dot(self.get_X().T, self.get_X())).diagonal())
        std_err_beta = np.sqrt(variance_beta)
        t_values = betas/ std_err_beta
        
        p_values =[2 * (1 - stats.t.cdf(np.abs(i), (len(self.get_X())-1))) for i in t_values]
        
        summary_df = pd.DataFrame()
        summary_df["Variables"] = ['(Intercept)'] + self.X_col if self.fit_intercept else self.X_col
        summary_df["Coefficients"] = betas
        summary_df["Standard Errors"] = std_err_beta
        summary_df["t values"] = t_values
        summary_df["Probabilites"] = p_values
        
        r2 = self.r_squared()
        adj_r2 = self.adjusted_r_squared()
        #
        fisher = self.fisher()
        #
        print('=========================================================================')
        print('                               OLS summary                               ')
        print('=========================================================================')
        print('| R² = {:.5f}                  | Adjusted-R² = {:.5f}'.format(r2, adj_r2))
        print('| n  = {:6}                   | p = {:5}'.format(self.n, self.p))
        print('| Fisher = {:.5f}                         '.format(fisher))
        print('=========================================================================')
        print(summary_df.to_string(index=False))
    
    def predict(self, new_data):
        """
        Returns predicted values Y_hat for for a new dataset
        
        Formula
        -------
        Y = X \beta
        
        """
        X_array = new_data[self.X_col].to_numpy()
        if self.fit_intercept:
            new_X = np.hstack((np.ones((new_data.shape[0], 1), dtype=X_array.dtype), X_array))
        else:
            new_X = X_array
        
        return np.dot(new_X, self.get_betas())



# Test:
"""
import statinf.GenerateData as gd

df = generate_dataset(coeffs=[1.2556, 3.465, 1.665414,9.5444], n=100, std_dev=50, intercept=3.6441)

formula = "Y ~ X1 + X2 + X3 + X0"
ols = OLS(formula, df, fit_intercept = True)

ols.summary()
"""
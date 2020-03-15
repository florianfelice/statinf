import numpy as np

from ..ml.initilizations import init_params
from ..ml.activations import sigmoid
# from .glm_test import GLM


class GLM:
    ''' 
    [C. Cameron & P.k. Trivedi] - Microeconometrics methods and applications - 2005
    '''
    def __init__(self, formula, data, fit_intercept=False, initializer='zeros'):
        self.data             = data # Pandas DataFrame
        # Parse the formula
        self.no_space_formula = formula.replace(' ', '')
        self.fit_intercept    = fit_intercept
        self.Y_col            = self.no_space_formula.split('~')[0]
        self.X_col            = self.no_space_formula.split('~')[1].split('+')
        self.X                = data[self.X_col].to_numpy()
        self.Y                = data[self.Y_col].to_numpy()
        self.log_likelihood   = []
        self.gradients        = []
        self.coefs_hist       = []
        self.hessian_hist     = []
        self.jacobian_hist    = []

        # Need to fit intercept?
        if self.fit_intercept:
            self.X.insert(1, 'intercept', 1)
        # Initialize parameters
        self.beta = init_params(rows=len(self.X_col), cols=1, method=initializer, isTheano=False)


    def _explain(self, beta):
        '''
        1rst step build the linear combination
        '''
        Xb   = self.X.dot(np.array(beta).T)
        return Xb

    def _prob(self, X):
        ''' 
        Prob
        '''
        prob = sigmoid(X.dot(self.beta))
        # prob = np.array([(1/(1+np.exp(-x))) for x in self._explain(beta)])
        return prob
        
    def _log_likelihood(self, X=None, Y=None):
        '''
        Compute Log-Likelihood 
        '''
        X_data = self.X if X is None else X
        Y_data = self.Y if Y is None else Y

        # Individual log-likelihood
        log_likelihood_id = np.array(Y_data * np.log(self._prob(X_data)) + (1 - Y_data) * np.log(1-self._prob(X_data)))
        # Overall log-likelihood
        log_likelihood = log_likelihood_id.sum()
        return log_likelihood


    def _jacobian(self):
        '''
        Gradient (Overall) not the score function 
        '''
        gradient = np.array(self.Y - self._prob(self.X)).dot(self.X) # gradient of the likelihood
        return gradient

    def _hessian(self):
        '''
        Hessian -- Fisher Information 
        '''
        hessian = np.array((self._prob(self.X).dot(1 - self._prob(self.X)).sum()) * self.X.T.dot(self.X)) #! Wrong formula
        return hessian

    def variance(self):
        '''
        Fisher Information
        '''
        variance = np.linalg.inv(self._hessian())
        return np.array(variance)

    def fit(self, max_it=10, improvement_threshold=0.995, keep_hist=True):
        it = 0
        diff = np.inf
        old_log_like = np.inf
    
        while (it < max_it) and (np.abs(diff) > 1-improvement_threshold):
            it           += 1
            log_like      = self._log_likelihood()
            gradient      = self._jacobian()
            variation     = self.variance().dot(gradient)
            self.beta    -= variation.T
            diff          = old_log_like - log_like
            old_log_like  = log_like

            if keep_hist:
                self.log_likelihood += [log_like]
                self.gradients      += [gradient]
            
        return print(it)

    def predict(self, max_it=10):
        beta         = self._fit(max_it)
        self.predict = [1 if x > 0.5 else 0 for x in self._prob(beta)]
        return print(f'''{self.predict}''')
    

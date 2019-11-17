import numpy as np


"""
Based on tutorial to be edited
- https://medium.com/@martinpella/logistic-regression-from-scratch-in-python-124c5636b8ac
- https://www.kaggle.com/jeppbautista/logistic-regression-from-scratch-python
"""


## TODO: check loss function
## TODO: check gradient descent methodology
## TODO: check stop iterations once convergence
## TODO: data generator
## TODO: summary
## TODO: add dask

class Logit:
    def __init__(self, learning_rate=0.01, max_iter=1000, fit_intercept=True, verbose=False):
        self.lr = lr
        self.num_iter = num_iter
        self.fit_intercept = fit_intercept
    
    def __add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)
    
    def __logistic(self, z):
        return 1 / (1 + np.exp(-z))

    def __log_likelihood(x, y, weights):
        z = np.dot(x, weights)
        ll = np.sum( y*z - np.log(1 + np.exp(z)) )
        return ll
    
    def fit(self, X, y):
        if self.fit_intercept:
            X = self.__add_intercept(X)
        
        # weights initialization
        self.theta = np.zeros(X.shape[1])
        
        for i in range(self.num_iter):
            h = self.__sigmoid(z)
            gradient = sum(y*log(h)+(1-y)*log(1-h))
            self.theta -= self.lr * gradient
            
            if(self.verbose == True and i % 10000 == 0):
                z = np.dot(X, self.theta)
                h = self.__sigmoid(z)
                print(f'loss: {self.__loss(h, y)} \t')

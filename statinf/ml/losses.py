import numpy as np
import theano.tensor as T


def log_stability(x):
    """
    """
    if x==0: 
        return 10e-9 
    elif x==1: 
        return 1.0-10e-9
    else:
        return x

def binary_cross_entropy(y_true, y_pred, verbose=False, tensor=False):
    """Computes the Mean Squared Error

    Args:
        y_true (list): Real values on which to compare.
        y_pred (list): Predicted values.
        root (bool): Return Root Mean Squared Error (RMSE) or simple MSE.

    Formula:
        $loss = \dfrac{1}{m} \times \sum_{i=1}^{m} (y_i - \hat{y}_i)^2$

    Returns:
        float: Binary Cross Entropy.
    
    References:
        * Friedman, J., Hastie, T. and Tibshirani, R., 2001. The elements of statistical learning. Ch. 4, pp. 120.
    """
    if tensor:
        loss = -y_true * T.log(y_pred) - (1-y_true) * T.log(1-y_pred)
    else:
        m = len(y_true) #.shape[1]
        pred_ls = [float(y) for y in y_pred]
        y_ls = [float(y) for y in y_true]
        pred = np.array([log_stability(a) for a in pred_ls]) #np.reshape(pred_ls, (np.product(pred_ls.shape),))])
        true = np.array([log_stability(a) for a in y_ls]) # np.reshape(y_ls, (np.product(y_ls.shape),))]) # np.array(np.reshape(y_true, (np.product(y_true.shape),)))
        # print(pred)
        # print(true)
        loss = (-1/m) * (np.dot(true.T, np.log(pred)) + np.dot(1-true.T, np.log(1-pred)))
    # print(loss)
    return loss



def mean_squared_error(y_true, y_pred, root=False, verbose=False):
    """Computes the Mean Squared Error

    Args:
        y_true (list): Real values on which to compare.
        y_pred (list): Predicted values.
        root (bool): Return Root Mean Squared Error (RMSE) or simple MSE.

    Formula:
        $loss = \dfrac{1}{m} \times \sum_{i=1}^{m} (y_i - \hat{y}_i)^2$

    Returns:
        float: MSE or RMSE.
    
    References:
        * Friedman, J., Hastie, T. and Tibshirani, R., 2001. The elements of statistical learning. Ch. 2, pp. 24.
    """
    m = len(y_true)
    loss = (1/m) * ((y_pred - y_true).sum())**2
    if root:
        return loss ** (1/2)
    else:
        return loss

def binary_accuracy(y_true, y_pred):
    """
    """
    return (y_true == y_pred).mean()
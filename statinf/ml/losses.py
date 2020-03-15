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
        $loss = - y_{i} \log \left[ G(\mathbf{x_i} \beta) \right] - (1 - y_{i}) \log \left[1 - G(\mathbf{x_i} \beta) \right]$

    Returns:
        float: Binary Cross Entropy.
    
    References:
        * Friedman, J., Hastie, T. and Tibshirani, R., 2001. The elements of statistical learning. Ch. 4, pp. 120.
    """
    if tensor:
        loss = -y_true * T.log(y_pred) - (1-y_true) * T.log(1-y_pred)
    else:
        loss = (-y_true * np.log(y_pred) - (1-y_true) * np.log(1-y_pred)).sum()
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
    loss = ((y_pred - y_true).sum())**2
    if root:
        return loss ** (1/2)
    else:
        return loss


def binary_accuracy(y_true, y_pred):
    """
    """
    return (y_true == y_pred).mean()


# MAPE formula
def mape(y_true, y_pred):
    """Computes the Mean Absolute Percentage Error.

    Args:
        y (list): Real values on which to compare.
        yhat (list): Predicted values.

    Returns:
        float: MAPE.
    """
    y = np.array(y_true)
    yhat = np.array(y_pred)
    m = len(y)
    mape = (100/m) * sum(np.abs(y-yhat))/sum(y)
    return mape
import numpy as np
import theano.tensor as T


def log_stability(x):
    """Log-stability for computing loss
    
    :param x: Input value
    :type x: float
    :return: Scaled value where :math:`\hat{x} \in (0, 1)`
    :rtype: float
    """
    if x==0: 
        return 10e-9 
    elif x==1: 
        return 1.0-10e-9
    else:
        return x


def binary_cross_entropy(y_true, y_pred, tensor=False):
    """Binary cross-entropy
    
    :param y_true: Real values on which to compare.
    :type y_true: numpy.array
    :param y_pred: Predicted values.
    :type y_pred: numpy.array
    :param tensor: Perform computation as tensor (theano type), defaults to False
    :type tensor: bool, optional
    :formula: :math:`loss = y_{i} \log \left[ \hat{y}_{i} \\right] + (1 - y_{i}) \log \left[1 - \hat{y}_{i} \\right]`
    :references: * Friedman, J., Hastie, T. and Tibshirani, R., 2001. `The elements of statistical learning <https://web.stanford.edu/~hastie/Papers/ESLII.pdf>`_. Ch. 2, pp. 24.
    :return: Binary cross-entropy
    :rtype: float
    """
    if tensor:
        loss = -y_true * T.log(y_pred) - (1-y_true) * T.log(1-y_pred)
    else:
        loss = (-y_true * np.log(y_pred) - (1-y_true) * np.log(1-y_pred)).sum()
    # print(loss)
    return loss




def mean_squared_error(y_true, y_pred, root=False):
    """Mean Squared Error
    
    :param y_true: Real values on which to compare.
    :type y_true: numpy.array
    :param y_pred: Predicted values.
    :type y_pred: numpy.array
    :param root: Return Root Mean Squared Error (RMSE), defaults to False.
    :type root: bool, optional

    :formula: :math:`loss = \\dfrac{1}{m} \\times \\sum_{i=1}^{m} (y_i - \\hat{y}_i)^2`

    :references: * Friedman, J., Hastie, T. and Tibshirani, R., 2001. `The elements of statistical learning <https://web.stanford.edu/~hastie/Papers/ESLII.pdf>`_. Ch. 2, pp. 24.

    :return: Mean Squared Error or its root.
    :rtype: float
    """
    loss = ((y_pred - y_true).sum())**2
    if root:
        return loss ** (1/2)
    else:
        return loss



def binary_accuracy(y_true, y_pred):
    """Accuracy for binary data
    
    :param y_true: Real values on which to compare.
    :type y_true: numpy.array
    :param y_pred: Predicted values
    :type y_pred: numpy.array
    
    :return: Binary accuracy (in percent)
    :rtype: float
    """
    return (y_true == y_pred).mean()


# MAPE formula
def mape(y_true, y_pred):
    """Computes the Mean Absolute Percentage Error
    
    :param y_true: Real values on which to compare.
    :type y_true: numpy.array
    :param y_pred: Predicted values.
    :type y_pred: numpy.array
    :formula: :math:`\dfrac{100}{n} \sum_{i=1}^{n} \dfrac{|y - \hat{y}|}{y}`
    
    :return: Mean Absolute Percentage Error as percentage.
    :rtype: float
    """
    y = np.array(y_true)
    yhat = np.array(y_pred)
    m = len(y)
    mape = (100/m) * sum(np.abs(y-yhat))/sum(y)
    return mape

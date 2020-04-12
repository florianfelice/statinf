import numpy as np
import theano.tensor as T

# Default activation functions
def sigmoid(x):
    """Sigmoid activation function
    
    :param x: Input value
    :type x: float or numpy.array
    
    :return: Sigmoid activated value: :math:`sigmoid(x) = \dfrac{1}{1 + e^{-x}}`
    :rtype: float
    """
    return 1 / (1 + np.exp(-x))


def tanh(x):
    """Hyperbolic tangent activation function
    
    :param x: Input value
    :type x: float or numpy.array
    
    :return: Activated value: :math:`tanh(x)`
    :rtype: float
    """
    return np.tanh(x)


def relu(x):
    """Rectified Linear Units activation function
    
    :param x: Input value
    :type x: float or numpy.array
    
    :return: Activated value: :math:`relu(x) = \max(0, x)`
    :rtype: float
    """
    return np.maximum(0, x)


def softplus(x):
    """Softplus activation function
    
    :param x: Input value
    :type x: float or numpy.array
    
    :return: Activated value: :math:`softplus(x) = \log(1 + e^{-x})`
    :rtype: float
    """
    return np.log(1 + np.exp(-x))

 

def logit(x, weights, bias, tensor=False):
    """Logistic function
    
    :param x: Input value
    :type x: numpy.array
    :param weights: Vector of weights :math:`\\beta`
    :type weights: numpy.array
    :param bias: Vector of bias :math:`\\epsilon`
    :type bias: numpy.array
    :param tensor: Perform computation as tensor (theano type), defaults to False
    :type tensor: bool, optional
    
    :return: Logistic transformation: :math:`logit(x, \\beta) = \dfrac{1}{1 + e^{-x \\beta}}`
    :rtype: float
    """

    if tensor:
        return 1 / (1 + T.exp(-T.dot(x, weights) + bias))
    else:
        return 1 / (1 + np.exp(-x.dot(weights) + bias))


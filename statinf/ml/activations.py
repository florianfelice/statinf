import numpy as np
import theano.tensor as T

# Default activation functions
def sigmoid(x):
    """
    Parameters
    -----------
    x: float
        Input value.
    
    Notes
    -----
    :math:`sigmoid(x) = \dfrac{1}{1 + e^{-x}}`
    
    Returns
    --------
    float: Sigmoid of x
    """
    return 1 / (1 + np.exp(-x))


def tanh(x):
    """
    Parameters
    -----------
    x: float
        Input value.
    
    Notes
    -----
    :math:`tanh(x)`
    
    Returns
    --------
    float: Hyperbloic tangent of x
    """
    return np.tanh(x)


def relu(x):
    """
    Parameters
    -----------
    x: float
        Input value.
    
    Notes
    -----
    :math:`relu(x) = \max(0, x)`
    
    Returns
    --------
    float: Hyperbloic tangent of x
    """
    return np.maximum(0, x)
    

def softplus(x):
    """
    Parameters
    -----------
    x: float
        Input value.
    
    Notes
    -----
    :math:`softplus(x) = \log(1 + e^{-x})`
    
    Returns
    --------
    float: Hyperbloic tangent of x
    """
    return np.log(1 + np.exp(-x))



def logit(x, weights, bias, tensor=False):
    """
    Parameters
    -----------
    x: float
        Input value.

    Notes
    -----
    :math:`sigmoid(x) = \dfrac{1}{1 + e^{-x}}`
    
    Returns
    --------
    float: Sigmoid of x
    """
    if tensor:
        return 1 / (1 + T.exp(-T.dot(x, weights) + bias))
    else:
        return 1 / (1 + np.exp(-x.dot(weights) + bias))
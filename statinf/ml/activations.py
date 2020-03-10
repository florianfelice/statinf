import numpy as np
import theano.tensor as T

# Default activation functions
def sigmoid(x):
    """
    Args:
        x (float): Input value.
    
    Formula:
        $\sigmoid(x) = \dfrac{1}{1 + e^{-x}} $
    
    Returns:
        float: Sigmoid of x
    """
    return 1 / (1 + np.exp(-x))


def tanh(x):
    """
    Args:
        x (float): Input value.
    
    Formula:
        $\tanh(x)$
    
    Returns:
        float: Hyperbloic tangent of x
    """
    return np.tanh(x)


def relu(x):
    """
    Args:
        x (float): Input value.
    
    Formula:
        $\relu(x) = \max(0, x)$
    
    Returns:
        float: Hyperbloic tangent of x
    """
    return np.maximum(0, x)
    

def softplus(x):
    """
    Args:
        x (float): Input value.
    
    Formula:
        $\softplus(x) = \log(1 + e^{-x})$
    
    Returns:
        float: Hyperbloic tangent of x
    """
    return np.log(1 + np.exp(-x))



def logit(x, weights, bias, tensor=False):
    """
    Args:
        x (float): Input value.
    
    Formula:
        $\sigmoid(x) = \dfrac{1}{1 + e^{-x}} $
    
    Returns:
        float: Sigmoid of x
    """
    if tensor:
        return 1 / (1 + T.exp(-T.dot(x, weights) + bias))
    else:
        return 1 / (1 + np.exp(-x.dot(weights) + bias))
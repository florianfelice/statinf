import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def tanh(x):
    return np.tanh(x)


def d_sigmoid(x, is_activated=False):
    """
    Args:
        x (float): Input value.
        is_activated (bool): Is the input value already an output of an activation function (defaults False).

    Formula:
        $\dfrac{\partial}{\partial x} \sigmoid(x) = \sigmoid(x) \times (1 - \sigmoid(x))$

    Returns:
        float: Derivative value of sigmoid.
    """
    if is_activated:
        d_f = x * (1 - x)
    else:
        d_f = sigmoid(x) * (1 - sigmoid(x))
    return d_f


def d_tanh(x, is_activated=False):
    """
    Args:
        x (float): Input value.
        is_activated (bool): Is the input value already an output of an activation function (defaults False).

    Formula:
        $\dfrac{\partial}{\partial x} \tanh(x) = 1 - \tanh^2(x)$

    Returns:
        float: Derivative value of tanh.
    """

    if is_activated:
        d_f = 1 - (x ** 2)
    else:
        d_f = 1 - (tanh(x) ** 2)
    return d_f
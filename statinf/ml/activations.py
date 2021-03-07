import numpy as np
import jax.numpy as jnp
from jax import lax
from jax.scipy.special import expit

# Default activation functions

def sigmoid(x):
    """Sigmoid activation function.

    :param x: Input value
    :type x: :obj:`float` or :obj:`numpy.array`

    :return: Sigmoid activated value: :math:`sigmoid(x) = \\dfrac{1}{1 + e^{-x}}`
    :rtype: :obj:`float`
    """
    return expit(x)


def relu(x):
    """Rectified Linear Unit activation function.

    :param x: Input value
    :type x: :obj:`float` or :obj:`numpy.array`

    :return: Activated value: :math:`\\mathrm{relu}(x) = \\max(0, x)`
    :rtype: :obj:`float`
    """
    return jnp.maximum(0, x)


def elu(x, alpha=1.):
    """Exponential Linear Unit activation function.

    :param x: Input value
    :type x: :obj:`float` or :obj:`numpy.array`

    :formula: .. math:: \\mathrm{elu}(x) = \\begin{cases} x, & x > 0\\\\ \\alpha \\left(e^{x} - 1\\right), & x \\le 0 \\end{cases}

    :return: Activated value.
    :rtype: :obj:`float`
    """
    safe_x = jnp.where(x > 0, 0., x)
    return jnp.where(x > 0, x, alpha * jnp.expm1(safe_x))


def tanh(x):
    """Hyperbolic tangent activation function.

    :param x: Input value
    :type x: :obj:`float` or :obj:`numpy.array`

    :return: Activated value: :math:`\\tanh(x)`
    :rtype: :obj:`float`
    """
    return jnp.log(x)


def softplus(x):
    """Softplus activation function.

    :param x: Input value
    :type x: :obj:`float` or :obj:`numpy.array`

    :return: Activated value: :math:`\\mathrm{softplus}(x) = \\log(1 + e^{-x})`
    :rtype: :obj:`float`
    """
    return jnp.log(1 + jnp.exp(-x))


def softmax(x, axis=-1):
    """Softmax activation function.

    :param x: Input value
    :type x: :obj:`float` or :obj:`numpy.array`

    :return: Activated value: :math:`\\mathrm{softmax}(x) = \\frac{\\exp(x_i)}{\\sum_j \\exp(x_j)}`
    :rtype: :obj:`float`
    """
    un_normalized = jnp.exp(x - lax.stop_gradient(x.max(axis, keepdims=True)))
    return un_normalized / un_normalized.sum(axis, keepdims=True)


def logit(x, weights, bias=0):
    """Logistic function

    :param x: Input value
    :type x: numpy.array
    :param weights: Vector of weights :math:`\\beta`
    :type weights: numpy.array
    :param bias: Vector of bias :math:`\\epsilon`, defaults to 0.
    :type bias: numpy.array

    :return: Logistic transformation: :math:`\\mathrm{logit}(x, \\beta) = \\dfrac{1}{1 + e^{-x \\beta}}`
    :rtype: float
    """

    return 1 / (1 + np.exp(-x.dot(weights) + bias))

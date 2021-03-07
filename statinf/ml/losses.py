import math
import numpy as np
import pandas as pd
import jax.numpy as jnp

from ..misc import format_object

def log_stability(x, delta=10e-5):
    """Log-stability for computing loss.

    :param x: Input value.
    :type x: :obj:`float`
    :param delta: Constant to move from 0 or 1, defaults to 10e-9.
    :type delta: :obj:`float`, optional

    :return: Stabilized value where :math:`\\hat{x} \\in (0, 1)`
    :rtype: :obj:`float`
    """
    if type(x) == float:
        if x == 0:
            return delta
        elif x == 1:
            return 1.0 - delta
        else:
            return x
    else:
        new_x = jnp.where(x == 0., delta, x)
        return jnp.where(new_x == 1., 1. - delta, new_x)

def binary_cross_entropy(y_true, y_pred):
    """Binary cross-entropy.

    :param y_true: Real values on which to compare.
    :type y_true: :obj:`numpy.array`
    :param y_pred: Predicted values.
    :type y_pred: :obj:`numpy.array`

    :formula: :math:`loss = y_{i} \\log \\left[ \\hat{y}_{i} \\right] + (1 - y_{i}) \\log \\left[1 - \\hat{y}_{i} \\right]`

    :references: * Friedman, J., Hastie, T. and Tibshirani, R., 2001. `The elements of statistical learning <https://web.stanford.edu/~hastie/Papers/ESLII.pdf>`_. Ch. 2, pp. 24.

    :return: Binary cross-entropy
    :rtype: :obj:`float`
    """
    # if tensor:
    #     loss = -y_true * T.log(y_pred) - (1-y_true) * T.log(1-y_pred)
    # else:
    # print(y_true)
    loss = jnp.sum(-y_true * jnp.log(log_stability(y_pred)) - (1 - y_true) * jnp.log(log_stability(1 - y_pred)))
    # print('Loss is')
    # print(loss)
    return loss


def mean_squared_error(y_true, y_pred, root=False):
    """Mean Squared Error.

    :param y_true: Real values on which to compare.
    :type y_true: :obj:`numpy.array`
    :param y_pred: Predicted values.
    :type y_pred: :obj:`numpy.array`
    :param root: Return Root Mean Squared Error (RMSE), defaults to False.
    :type root: :obj:`bool`, optional

    :formula: :math:`loss = \\dfrac{1}{m} \\times \\sum_{i=1}^{m} (y_i - \\hat{y}_i)^2`

    :references: * Friedman, J., Hastie, T. and Tibshirani, R., 2001. `The elements of statistical learning <https://web.stanford.edu/~hastie/Papers/ESLII.pdf>`_. Ch. 2, pp. 24.

    :return: Mean Squared Error or its root.
    :rtype: :obj:`float`
    """

    loss = jnp.square(y_pred - y_true)
    mse = jnp.mean(loss)
    if root:
        return jnp.sqrt(mse)
    else:
        return mse


def binary_accuracy(y_true, y_pred):
    """Accuracy for binary data.

    :param y_true: Real values on which to compare.
    :type y_true: :obj:`numpy.array`
    :param y_pred: Predicted values
    :type y_pred: numpy.array

    :return: Binary accuracy (in percent)
    :rtype: :obj:`float`
    """
    true = format_object(y_true, to_type='list', name='y_true')
    pred = format_object(y_pred, to_type='list', name='y_pred')
    # true = y_true if type(y_true) == list else [x[0] for x in np.asarray(y_true)]
    # pred = y_pred if type(y_pred) == list else [x[0] for x in np.asarray(y_pred)]

    perf = pd.DataFrame({'true': true, 'pred': pred})
    return (perf.true == perf.pred).mean()

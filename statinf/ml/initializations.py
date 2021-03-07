import jax.numpy as jnp
from jax import random as jrdm
import numpy as np

key, _ = jrdm.split(jrdm.PRNGKey(0))


def init_params(rows, cols, method='xavier', mean=0., std=1., key=key):
    """Initialize the weight and bias matrices based on probabilistic distribution.

    :param rows: Size of the input, number of rows to be generated.
    :type rows: int
    :param cols: Size of the output, number of columns to be generated.
    :type cols: int
    :param method: Distibution to use to generated the weights, defaults to 'xavier'.
    :type method: str, optional
    :param mean: Mean for the distribution to be generated, defaults to 0.
    :type mean: float, optional
    :param std: Standard deviation for the distribution to be generated, defaults to 1.
    :type std: float, optional
    :param tensor: Needs to return a theano friendly-format, defaults to True.
    :type tensor: bool, optional
    :param seed: Seed to be set for randomness, defaults to None.
    :type seed: int, optional

    :raises ValueError: `method` needs to be 'ones', 'zeros', 'uniform', 'xavier' or 'normal', see below for details.

    :method: * **Zeros**: :math:`W_j = \\vec{0}`
        * **Ones**: :math:`W_j = \\vec{1}`
        * **Uniform**: :math:`W_j \\sim \\mathcal{U} _{\\left[0, 1 \\right)}`
        * **Xavier**: :math:`W_j \\sim \\mathcal{U}\\left[ -\\frac{\\sqrt{6}}{\\sqrt{n_j + n_{j+1}}}, \\frac{\\sqrt{6}}{\\sqrt{n_j + n_{j+1}}} \\right]`
        * **Normal**: :math:`W_j \\sim \\mathcal{N}(0, 1)`

    :references: * Neuneier, Ralph, and Hans Georg Zimmermann. "`How to train neural networks <https://link.springer.com/chapter/10.1007/3-540-49430-8_18>`_" In Neural networks: tricks of the trade, pp. 373-423. Springer, Berlin, Heidelberg, 1998.
        * Glorot, Xavier, and Yoshua Bengio. "`Understanding the difficulty of training deep feedforward neural networks <http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf>`_" In Proceedings of the thirteenth international conference on artificial intelligence and statistics, pp. 249-256. 2010.
    :return: Weight or bias matrix for initiating ML optimization
    :rtype: numpy.array
    """

    # Get the weights
    if method.lower() == 'ones':
        # W = jnp.ones((rows, cols))
        W = np.ones((rows, cols))
    if method.lower() == 'zeros':
        W = jnp.zeros((rows, cols))
    elif method.lower() == 'uniform':
        W = jrdm.uniform(key, (rows, cols))
    elif method.lower() == 'xavier':
        W = jrdm.uniform(key, shape=(rows, cols),
                         minval=-jnp.sqrt(6. / (rows + cols)),
                         maxval=jnp.sqrt(6. / (rows + cols)))
    elif method.lower() == 'normal':
        W = jrdm.normal(key, (rows, cols))
    else:
        raise ValueError(f"Weight initialization method not valid. Muse be in 'ones', 'zeros', 'uniform', xavier', 'normal'. Got '{method}'.")

    return W

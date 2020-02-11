import numpy as np
import theano


def init_params(rows, cols, method='xavier', mean=0., std=1., isTheano=True, seed=None):
    """
    Initialize the weight matrix based on probabilitic distribution

    Args:
        rows (int): Size of the input, number of rows to be generated.
        cols (int): Size of the output, number of columns to be generated.
        method (str): Distibution to use to generated the weights (defaults 'xavier').
        mean (float): Mean for the distribution to be generated (defaults 0.0).
        std (float): Standard deviation for the distribution to be generated (defaults 1.0).
        isTheano (bool): Needs to return a theano friendly-format (defaults True).
        seed (int): Seed to be set for randomness (defaults None).

    Formula:
    * Zeros:
        $W_j = \vec{0}$
    * Ones:
        $W_j = \vec{1}$
    * Uniform:
        $W_j \sim \mathcal{U} _{\left[0, 1 \right)}$
    * Xavier:
        $W_j \sim \mathcal{U} _{\left[ -\frac{\sqrt{6}}{\sqrt{n_j + n_{j+1}}}, \frac{\sqrt{6}}{\sqrt{n_j + n_{j+1}}} \right]}$
    * Normal:
        $W_j \sim \mathcal{N}(0, 1)$

    References:
    * Neuneier, Ralph, and Hans Georg Zimmermann. "How to train neural networks." In Neural networks: tricks of the trade, pp. 373-423. Springer, Berlin, Heidelberg, 1998.
    * Glorot, Xavier, and Yoshua Bengio. "Understanding the difficulty of training deep feedforward neural networks." In Proceedings of the thirteenth international conference on artificial intelligence and statistics, pp. 249-256. 2010.

    Returns:
        np.array: Initialized weight matrix
    """
    # Set a seed or not
    rdm = np.random.RandomState(seed) if seed is not None else np.random

    # Get the weights
    if method.upper() == 'ONES':
        W = np.ones((rows, cols))
    if method.upper() == 'ZEROS':
        W = np.zeros((rows, cols))
    elif method.upper() == 'UNIFORM':
        W = rdm.rand(rows, cols)
    elif method.upper() == 'XAVIER':
        W = rdm.uniform(low = -np.sqrt(6. / (rows + cols)),
                    high = np.sqrt(6. / (rows + cols)),
                    size = (rows, cols))
    elif method.upper() == 'NORMAL':
        W = rdm.normal(mean, std, rows * cols).reshape((rows, cols))
    else:
        raise ValueError('Weight initialization method not valid.')

    if isTheano:
        return np.asarray(W, dtype=theano.config.floatX)
    else:
        return W


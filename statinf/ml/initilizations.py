import numpy as np


def init_weights(rows, cols, method='ones', mean=0., std=1.):
    """
    """
    if method.upper() == 'ONES':
        return np.ones((rows, cols))
    if method.upper() == 'ZEROS':
        return np.zeros((rows, cols))
    elif method.upper() == 'UNIFORM':
        return np.random.rand(rows, cols)
    elif method.upper() == 'NORMAL':
        return np.random.normal(mean, std, rows * cols).reshape((rows, cols))
    else:
        raise ValueError('Weight initialization method not valid.')


def init_bias(cols, rows=1, method='zeros', mean=0., std=1.):
    """
    """
    if method.upper() == 'ZEROS':
        return np.zeros((rows, cols))
    elif method.upper() == 'UNIFORM':
        return np.random.rand(rows, cols)
    elif method.upper() == 'NORMAL':
        return np.random.normal(mean, std, rows * cols).reshape((rows, cols))
    else:
        raise ValueError('Bias initialization method not valid.')
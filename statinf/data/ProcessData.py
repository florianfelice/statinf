import numpy as np

def rankdata(x):
    """Assigns rank to data.
    This is mainly used for analysis like Spearman's correlation.

    :param x: Input vector. Format can be :obj:`numpy.array`, :obj:`list` or :obj:`pandas.Series`.
    :type x: :obj:`numpy.array`

    :example:

    >>> rankdata([2., 5.44, 3.93, 3.3, 1.1])
    ... array([1, 4, 3, 2, 0])
    
    :return: Vector with ranked values.
    :rtype: :obj:`numpy.array`
    """
    x_arr = np.asarray(x)
    sorted_array = sorted(x_arr)
    rk = [sorted_array.index(i) for i in x_arr]
    return np.array(rk)

import jax.numpy as jnp

from collections import OrderedDict


class Optimizer:
    """Optimization updater

    :param object: Optimizer object
    :type object: class
    """
    def __init__(self, learning_rate=0.01):
        if learning_rate is None:
            raise NotImplementedError()
        self.learning_rate = learning_rate

    def updates(self, params=None, grads=None):
        if params is None:
            raise NotImplementedError()
        if grads is None:
            raise NotImplementedError()

        self.updates = OrderedDict()

class SGD(Optimizer):
    """Stochastic Gradient Descent optimizer.

    :param learning_rate: Step size, defaults to 0.01.
    :type learning_rate: :obj:`float`
    :param alpha: Momentum parameter, defaults to 0.0.
    :type alpha: :obj:`float`

    :formula: .. math:: \\theta_{t} = \\theta_{t-1} + \\alpha v - \\epsilon \\nabla f_{t}(\\theta_{t-1})

    :references: Goodfellow, I., Bengio, Y., & Courville, A. (2016). `Deep learning <https://www.deeplearningbook.org/>`_. MIT press.

    :example:

    >>> SGD(learning_rate=0.01, alpha=0.).updates(params, grads)
    """
    def __init__(self, learning_rate=0.01, alpha=0.):
        self.learning_rate = learning_rate
        self.alpha = alpha  # Momentum parameter
        self.vs = {'w': jnp.zeros(1), 'b': jnp.zeros(1)}

    def update(self, params=None, grads=None):
        """Update loss using the SGD formula.

        :param params: Dictionnary with parameters to be updated, defaults to None.
        :type params: :obj:`dict`, optional
        :param grads: Dictionnary with the computed gradients, defaults to None.
        :type grads: :obj:`dict`, optional

        :return: Updated parameters.
        :rtype: :obj:`dict`
        """

        # return updates
        updated = params

        for _l, _p in zip(params.keys(), ['w', 'b']):
            _v = self.vs[_p] * self.alpha - self.learning_rate * grads[_l][_p]
            updated[_l][_p] = params[_l][_p] + _v

            self.vs[_p] = _v

        return updated


class Adam(Optimizer):
    """Adaptive Moments optimizer.

    :param learning_rate: Step size, defaults to 0.001.
    :type learning_rate: :obj:`float`
    :param beta1: Exponential decay rate for first moment estimate, defaults to 0.9.
    :type beta1: :obj:`float`
    :param beta2: Exponential decay rate for scond moment estimate, defaults to 0.999.
    :type beta2: :obj:`float`
    :param delta: Constant for division stability, defaults to 10e-8.
    :type delta: :obj:`float`

    :formula: .. math:: m_{t} = \\beta_{1} m_{t-1} + (1 - \\beta_{1}) \\nabla_{\\theta} f_{t}(\\theta_{t-1})
            .. math:: v_{t} = \\beta_{2} v_{t-1} + (1 - \\beta_{2}) \\nabla_{\\theta}^{2} f_{t}(\\theta_{t-1})
            .. math:: \\hat{m}_{t} = \\dfrac{m_{t}}{1 - \\beta_{1}^{t}}
            .. math:: \\hat{v}_{t} = \\dfrac{v_{t}}{1 - \\beta_{2}^{t}}
            .. math:: \\theta_{t} = \\theta_{t-1} - \\alpha \\dfrac{\\hat{m}_{t}}{\\sqrt{\\hat{v}_{t}} + \\delta}


    :references: * Kingma, D. P., & Ba, J. (2014). `Adam: A method for stochastic optimization <https://arxiv.org/pdf/1412.6980.pdf>`_. arXiv preprint arXiv:1412.6980

        * Goodfellow, I., Bengio, Y., & Courville, A. (2016). `Deep learning <https://www.deeplearningbook.org/>`_. MIT press.

    :example:

    >>> Adam(learning_rate=0.001, beta1=0.9, beta2=0.999, delta=10e-8).updates(params, grads)
    """
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, delta=10e-8):
        self.alpha = learning_rate
        self.b1 = beta1
        self.b2 = beta2
        self.delta = delta
        self.t = 1.

        self.ms = {'w': jnp.zeros(1), 'b': jnp.zeros(1)}
        self.vs = {'w': jnp.zeros(1), 'b': jnp.zeros(1)}

    def update(self, params=None, grads=None):
        """Update loss using the Adam formula.

        :param params: Dictionnary with parameters to be updated, defaults to None.
        :type params: :obj:`dict`, optional
        :param grads: Dictionnary with the computed gradients, defaults to None.
        :type grads: :obj:`dict`, optional

        :return: Updated parameters.
        :rtype: :obj:`dict`
        """
        updated = params

        for _l, _p in zip(params.keys(), ['w', 'b']):
            m = self.ms[_p]
            v = self.vs[_p]
            _m = self.b1 * m + (1 - self.b1) * grads[_l][_p]
            _v = self.b2 * v + (1 - self.b2) * jnp.square(grads[_l][_p])

            m_hat = _m / (1 - jnp.asarray(self.b1, m.dtype) ** (self.t + 1))
            v_hat = _v / (1 - jnp.asarray(self.b2, m.dtype) ** (self.t + 1))

            updated[_l][_p] = params[_l][_p] - self.alpha * m_hat / (jnp.sqrt(v_hat) + self.delta)
            self.ms[_p] = _m
            self.vs[_p] = _v
        self.t += 1.

        return updated


class AdaMax(Optimizer):
    """AdaMax optimizer (Adam with infinite norm).

    :param learning_rate: Step size, defaults to 0.001.
    :type learning_rate: :obj:`float`
    :param beta1: Exponential decay rate for first moment estimate, defaults to 0.9.
    :type beta1: :obj:`float`
    :param beta2: Exponential decay rate for scond moment estimate, defaults to 0.999.
    :type beta2: :obj:`float`

    :formula: .. math:: m_{t} = \\beta_{1} m_{t-1} + (1 - \\beta_{1}) \\nabla_{\\theta} f_{t}(\\theta_{t-1})
            .. math:: u_{t} = \\max(\\beta_{2} \\cdot u_{t-1}, |\\nabla f_{t}(\\theta_{t-1})|)
            .. math:: \\theta_{t} = \\theta_{t-1} - \\alpha \\dfrac{\\hat{m}_{t}}{\\sqrt{\\hat{v}_{t}} + \\epsilon}


    :references: * Kingma, D. P., & Ba, J. (2014). `Adam: A method for stochastic optimization <https://arxiv.org/pdf/1412.6980.pdf>`_. arXiv preprint arXiv:1412.6980
        * Goodfellow, I., Bengio, Y., & Courville, A. (2016). `Deep learning <https://www.deeplearningbook.org/>`_. MIT press.

    :example:

    >>> AdaMax(learning_rate=0.001, beta1=0.9, beta2=0.999).updates(params, grads)
    """
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999):
        self.alpha = learning_rate
        self.b1 = beta1
        self.b2 = beta2
        self.t = 1.

        self.ms = {'w': jnp.zeros(1), 'b': jnp.zeros(1)}
        self.us = {'w': jnp.zeros(1), 'b': jnp.zeros(1)}

    def update(self, params=None, grads=None):
        """Update loss using the AdaMax formula.

        :param params: Dictionnary with parameters to be updated, defaults to None.
        :type params: :obj:`dict`, optional
        :param grads: Dictionnary with the computed gradients, defaults to None.
        :type grads: :obj:`dict`, optional

        :return: Updated parameters.
        :rtype: :obj:`dict`
        """
        updated = params

        for _l, _p in zip(params.keys(), ['w', 'b']):
            _m = self.b1 * self.ms[_p] + (1 - self.b1) * grads[_l][_p]
            _u = jnp.maximum(self.b2 * self.us[_p], jnp.abs(grads[_l][_p]))

            updated[_l][_p] = params[_l][_p] - (self.alpha / (1 - self.b1 ** self.t)) * (_m / _u)

            self.ms[_p] = _m
            self.us[_p] = _u

        self.t += 1

        return updated


class AdaGrad(Optimizer):
    """Adaptive Gradient optimizer.

    :param learning_rate: Step size, defaults to 0.001.
    :type learning_rate: :obj:`float`
    :param delta: Constant for division stability, defaults to 10e-7.
    :type delta: :obj:`float`

    :formula: .. math:: r = r + \\nabla f_{t}(\\theta_{t}) \\odot \\nabla f_{t}(\\theta_{t-1})
            .. math:: \\theta_{t} = \\theta_{t-1} - \\frac{\\epsilon}{\\sqrt{\\delta + r}} \\odot \\nabla f_{t}(\\theta_{t-1})


    :references: * Duchi, J., Hazan, E., & Singer, Y. (2011). `Adaptive subgradient methods for online learning and stochastic optimization <http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf>`_. Journal of machine learning research, 12 (Jul), 2121-2159.
        * Goodfellow, I., Bengio, Y., & Courville, A. (2016). `Deep learning <https://www.deeplearningbook.org/>`_. MIT press.

    :example:

    >>> AdaGrad(learning_rate=0.001, delta=10e-7).updates(params, grads)
    """
    def __init__(self, learning_rate=0.001, delta=10e-7):
        self.learning_rate = learning_rate
        self.delta = delta  # Small constant for numerical stability

        self.accugrads = {'w': jnp.zeros(1), 'b': jnp.zeros(1)}  # Gradient accumulation variable

    def update(self, params=None, grads=None):
        """Update loss using the AdaGrad formula.

        :param params: Dictionnary with parameters to be updated, defaults to None.
        :type params: :obj:`dict`, optional
        :param grads: Dictionnary with the computed gradients, defaults to None.
        :type grads: :obj:`dict`, optional

        :return: Updated parameters.
        :rtype: :obj:`dict`
        """
        updated = params

        for _l, _p in zip(params.keys(), ['w', 'b']):
            _r = self.accugrads[_p] + grads[_l][_p] * grads[_l][_p]
            updated[_l][_p] = params[_l][_p] - (self.learning_rate / jnp.sqrt(_r + self.delta)) * grads[_l][_p]

            self.accugrads[_p] = _r

        return updated


class RMSprop(Optimizer):
    """RMSprop optimizer.

    :param learning_rate: Step size, defaults to 0.001.
    :type learning_rate: :obj:`float`
    :param rho: Decay rate, defaults to 0.9.
    :type rho: :obj:`float`
    :param delta: Constant for division stability, defaults to 10e-6.
    :type delta: :obj:`float`

    :formula: .. math:: r = \\rho r + (1- \\rho) \\nabla f_{t}(\\theta_{t-1}) \\odot \\nabla f_{t}(\\theta_{t-1})

        .. math:: \\theta_{t} = \\theta_{t-1} - \\dfrac{\\epsilon}{\\sqrt{\\delta + r}} \\odot \\nabla f_{t}(\\theta_{t-1})


    :references: * Tieleman, T., & Hinton, G. (2012). `Lecture 6.5-rmsprop: Divide the gradient by a running average of its recent magnitude <https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf>`_. COURSERA: Neural networks for machine learning, 4(2), 26-31.

        * Goodfellow, I., Bengio, Y., & Courville, A. (2016). `Deep learning <https://www.deeplearningbook.org/>`_. MIT press.

    :example:

    >>> RMSprop(learning_rate=0.001, rho=0.9, delta=10e-6).updates(params, grads)
    """
    def __init__(self, learning_rate=0.001, rho=0.9, delta=10e-6):
        self.learning_rate = learning_rate
        self.rho = rho
        self.delta = delta

        self.rs = {'w': jnp.zeros(1), 'b': jnp.zeros(1)}

    def update(self, params=None, grads=None):
        """Update loss using the RMSprop formula.

        :param params: Dictionnary with parameters to be updated, defaults to None.
        :type params: :obj:`dict`, optional
        :param grads: Dictionnary with the computed gradients, defaults to None.
        :type grads: :obj:`dict`, optional

        :return: Updated parameters.
        :rtype: :obj:`dict`
        """
        updated = params

        for _l, _p in zip(params.keys(), ['w', 'b']):
            _r = self.rs[_p] * self.rho
            _r += (1 - self.rho) * grads[_l][_p] * grads[_l][_p]
            updated[_l][_p] = params[_l][_p] - self.learning_rate * grads[_l][_p] / jnp.sqrt(_r + self.delta)

            self.rs[_p] = _r

        return updated

import theano
import theano.tensor as T
import numpy as np
from collections import OrderedDict


"""
References
----------
https://github.com/satopirka/deep-learning-theano/blob/master/example/mlp.py
https://github.com/satopirka/deep-learning-theano/blob/master/dnn/optimizers.py
"""

def build_shared_zeros(shape, name='v'):
    """Builds a theano shared variable filled with a zeros numpy array
    
    :param shape: Shape of the vector to create.
    :type shape: tuple
    :param name: Name to give to the shared theano value, defaults to 'v'.
    :type name: str

    :example:

    >>> build_shared_zeros(shape=(5, 10), name='r')

    :return: Theano shared matrix
    :rtype: theano.shared
    """
    return theano.shared(
        value=np.zeros(shape, dtype=theano.config.floatX), 
        name=name, 
        borrow=True
    )


class Optimizer(object):
    """Optimization updater
    
    :param object: Optimizer object
    :type object: class
    """
    def __init__(self, params=None):
        if params is None:
            raise NotImplementedError()
        self.params = params

    def updates(self, loss=None):
        if loss is None:
            raise NotImplementedError()

        self.updates = OrderedDict()
        self.gparams = [T.grad(loss, param) for param in self.params]


class SGD(Optimizer):
    """Stochastic Gradient Descent optimizer
    
    :param params: List compiling the parameters to update.
    :type params: list
    :param learning_rate: Step size, defaults to 0.01.
    :type learning_rate: float

    :formula: .. math:: \\theta_{t} = \\theta_{t-1} - \\epsilon \\nabla f_{t}(\\theta_{t-1})

    :references: Goodfellow, I., Bengio, Y., & Courville, A. (2016). `Deep learning <https://www.deeplearningbook.org/>`_. MIT press.

    :example:
    
    >>> SGD(params=[W, b], learning_rate=0.01).updates(cost)
    """
    def __init__(self, learning_rate=0.01, params=None):
        super(SGD, self).__init__(params=params)
        self.learning_rate = learning_rate

    def updates(self, loss=None):
        """Update loss using the SGD formula
        
        :param loss: Vector of loss to be updated, defaults to None
        :type loss: tensor, optional

        :return: Updated parameters and gradients
        :rtype: tensor
        """
        super(SGD, self).updates(loss=loss)

        for param, gparam in zip(self.params, self.gparams):
            self.updates[param] = param - self.learning_rate * gparam

        return self.updates


class MomentumSGD(Optimizer):
    """Stochastic Gradient Descent with Momentum
    
    :param params: List compiling the parameters to update.
    :type params: list
    :param learning_rate: Step size, defaults to 0.01.
    :type learning_rate: float
    :param alpha: Momentum parameter, defaults to 0.9.
    :type alpha: float

    :formula: .. math:: \\theta_{t} = \\theta_{t-1} + \\alpha v - \\epsilon \\nabla f_{t}(\\theta_{t-1})

    :references: Goodfellow, I., Bengio, Y., & Courville, A. (2016). `Deep learning <https://www.deeplearningbook.org/>`_. MIT press.

    :example:

    >>> MomentumSGD(params=[W, b], learning_rate=0.01, alpha=0.9).updates(cost)
    """
    def __init__(self, params=None, learning_rate=0.01, alpha=0.9):
        super(MomentumSGD, self).__init__(params=params)
        self.learning_rate = learning_rate
        # Momentum value
        self.alpha = alpha
        self.vs = [build_shared_zeros(t.shape.eval(), 'v') for t in self.params]

    def updates(self, loss=None):
        """Update loss using the Momentum SGD formula
        
        :param loss: Vector of loss to be updated, defaults to None
        :type loss: tensor, optional

        :return: Updated parameters and gradients
        :rtype: tensor
        """
        super(MomentumSGD, self).updates(loss=loss)

        for v, param, gparam in zip(self.vs, self.params, self.gparams):
            _v = self.alpha * v - self.learning_rate * gparam
            self.updates[param] = param + _v
            self.updates[v] = _v

        return self.updates


class RMSprop(Optimizer):
    """RMSprop optimizer
    
    :param params: List compiling the parameters to update.
    :type params: list
    :param learning_rate: Step size, defaults to 0.001.
    :type learning_rate: float
    :param rho: Decay rate, defaults to 0.9.
    :type rho: float
    :param delta: Constant for division stability, defaults to 10e-6.
    :type delta: float

    :formula: .. math:: r = \\rho r + (1- \\rho) \\nabla f_{t}(\\theta_{t-1}) \\odot \\nabla f_{t}(\\theta_{t-1})
    
        .. math:: \\theta_{t} = \\theta_{t-1} - \\dfrac{\\epsilon}{\\sqrt{\\delta + r}} \\odot \\nabla f_{t}(\\theta_{t-1})


    :references: * Tieleman, T., & Hinton, G. (2012). `Lecture 6.5-rmsprop: Divide the gradient by a running average of its recent magnitude <https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf>`_. COURSERA: Neural networks for machine learning, 4(2), 26-31.
        
        * Goodfellow, I., Bengio, Y., & Courville, A. (2016). `Deep learning <https://www.deeplearningbook.org/>`_. MIT press.

    :example:

    >>> RMSprop(params=[W, b], learning_rate=0.01, rho=0.9, delta=10e-6).updates(cost)
    """
    def __init__(self, params=None, learning_rate=0.001, rho=0.9, delta=10e-6):
        super(RMSprop, self).__init__(params=params)

        self.learning_rate = learning_rate
        self.rho = rho
        self.delta = delta

        self.mss = [build_shared_zeros(t.shape.eval(), 'r') for t in self.params]

    def updates(self, loss=None):
        """Update loss using the RMSprop formula
        
        :param loss: Vector of loss to be updated, defaults to None
        :type loss: tensor, optional

        :return: Updated parameters and gradients
        :rtype: tensor
        """
        super(RMSprop, self).updates(loss=loss)

        for r, param, gparam in zip(self.mss, self.params, self.gparams):
            _r = r * self.rho
            _r += (1 - self.rho) * gparam * gparam
            self.updates[r] = _r
            self.updates[param] = param - self.learning_rate * gparam / T.sqrt(self.delta + _r)

        return self.updates


class Adam(Optimizer):
    """Adaptive Moments optimizer
    
    :param params: List compiling the parameters to update.
    :type params: list
    :param learning_rate: Step size, defaults to 0.001.
    :type learning_rate: float
    :param beta1: Exponential decay rate for first moment estimate, defaults to 0.9.
    :type beta1: float
    :param beta2: Exponential decay rate for scond moment estimate, defaults to 0.999.
    :type beta2: float
    :param delta: Constant for division stability, defaults to 10e-8.
    :type delta: float

    :formula: .. math:: m_{t} = \\beta_{1} m_{t-1} + (1 - \\beta_{1}) \\nabla_{\\theta} f_{t}(\\theta_{t-1})
            .. math:: v_{t} = \\beta_{2} v_{t-1} + (1 - \\beta_{2}) \\nabla_{\\theta}^{2} f_{t}(\\theta_{t-1})
            .. math:: \\hat{m}_{t} = \\dfrac{m_{t}}{1 - \\beta_{1}^{t}}
            .. math:: \\hat{v}_{t} = \\dfrac{v_{t}}{1 - \\beta_{2}^{t}}
            .. math:: \\theta_{t} = \\theta_{t-1} - \\alpha \\dfrac{\\hat{m}_{t}}{\\sqrt{\\hat{v}_{t}} + \\delta}


    :references: * Kingma, D. P., & Ba, J. (2014). `Adam: A method for stochastic optimization <https://arxiv.org/pdf/1412.6980.pdf>`_. arXiv preprint arXiv:1412.6980
        
        * Goodfellow, I., Bengio, Y., & Courville, A. (2016). `Deep learning <https://www.deeplearningbook.org/>`_. MIT press.

    :example:

    >>> ADAM(params=[W, b], learning_rate=0.001, beta1=0.9, beta2=0.999, delta=10e-8).updates(cost)
    """
    def __init__(self, params=None, learning_rate=0.001, beta1=0.9, beta2=0.999, delta=10e-8):
        super(Adam, self).__init__(params=params)

        self.alpha = learning_rate
        self.b1 = beta1
        self.b2 = beta2
        self.t = theano.shared(np.float32(1))
        self.delta = delta

        self.ms = [build_shared_zeros(t.shape.eval(), 'm') for t in self.params]
        self.vs = [build_shared_zeros(t.shape.eval(), 'v') for t in self.params]

    def updates(self, loss=None):
        """Update loss using the Adam formula
        
        :param loss: Vector of loss to be updated, defaults to None
        :type loss: tensor, optional

        :return: Updated parameters and gradients
        :rtype: tensor
        """
        super(Adam, self).updates(loss=loss)

        for m, v, param, gparam in zip(self.ms, self.vs, self.params, self.gparams):
            _m = self.b1 * m + (1 - self.b1) * gparam
            _v = self.b2 * v + (1 - self.b2) * gparam ** 2

            m_hat = _m / (1 - self.b1 ** self.t)
            v_hat = _v / (1 - self.b2 ** self.t)

            self.updates[param] = param - self.alpha * m_hat / (T.sqrt(v_hat) + self.delta)
            self.updates[m] = _m
            self.updates[v] = _v
        self.updates[self.t] = self.t + 1.0

        return self.updates


class AdaGrad(Optimizer):
    """Adaptive Gradient optimizer
    
    :param params: List compiling the parameters to update.
    :type params: list
    :param learning_rate: Step size, defaults to 0.01.
    :type learning_rate: float
    :param delta: Constant for division stability, defaults to 1e-7.
    :type delta: float

    :formula: .. math:: r = r + \\nabla f_{t}(\\theta_{t}) \\odot \\nabla f_{t}(\\theta_{t-1})
            .. math:: \\theta_{t} = \\theta_{t-1} - \\frac{\\epsilon}{\\sqrt{\\delta + r}} \\odot \\nabla f_{t}(\\theta_{t-1})


    :references: * Duchi, J., Hazan, E., & Singer, Y. (2011). `Adaptive subgradient methods for online learning and stochastic optimization <http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf>`_. Journal of machine learning research, 12 (Jul), 2121-2159.
        * Goodfellow, I., Bengio, Y., & Courville, A. (2016). `Deep learning <https://www.deeplearningbook.org/>`_. MIT press.

    :example:

    >>> AdaGrad(params=[W, b], learning_rate=0.01, delta=1e-7).updates(cost)
    """
    def __init__(self, params=None, learning_rate=0.01, delta=1e-7):
        super(AdaGrad, self).__init__(params=params)

        self.learning_rate = learning_rate
        # Small constant for numerical stability
        self.delta = delta
        # Gradient accumulation variable
        self.accugrads = [build_shared_zeros(t.shape.eval(), 'r') for t in self.params]

    def updates(self, loss=None):
        """Update loss using the AdaGrad formula
        
        :param loss: Vector of loss to be updated, defaults to None
        :type loss: tensor, optional

        :return: Updated parameters and gradients
        :rtype: tensor
        """
        super(AdaGrad, self).updates(loss=loss)

        for r, param, gparam in zip(self.accugrads, self.params, self.gparams):
            _r = r + gparam * gparam
            self.updates[param] = param - (self.learning_rate / T.sqrt(_r + self.delta)) * gparam
            self.updates[r] = _r

        return self.updates


class AdaMax(Optimizer):
    """AdaMax optimizer (Adam with infinite norm)
    
    :param params: List compiling the parameters to update.
    :type params: list
    :param learning_rate: Step size, defaults to 0.001.
    :type learning_rate: float
    :param beta1: Exponential decay rate for first moment estimate, defaults to 0.9.
    :type beta1: float
    :param beta2: Exponential decay rate for scond moment estimate, defaults to 0.999.
    :type beta2: float

    :formula: .. math:: m_{t} = \\beta_{1} m_{t-1} + (1 - \\beta_{1}) \\nabla_{\\theta} f_{t}(\\theta_{t-1})
            .. math:: u_{t} = \\max(\\beta_{2} \\cdot u_{t-1}, |\\nabla f_{t}(\\theta_{t-1})|)
            .. math:: \\theta_{t} = \\theta_{t-1} - \\alpha \\dfrac{\\hat{m}_{t}}{\\sqrt{\\hat{v}_{t}} + \\epsilon}


    :references: * Kingma, D. P., & Ba, J. (2014). `Adam: A method for stochastic optimization <https://arxiv.org/pdf/1412.6980.pdf>`_. arXiv preprint arXiv:1412.6980
        * Goodfellow, I., Bengio, Y., & Courville, A. (2016). `Deep learning <https://www.deeplearningbook.org/>`_. MIT press.

    :example:

    >>> AdaMax(params=[W, b], learning_rate=0.001, beta1=0.9, beta2=0.999).updates(cost)
    """
    
    def __init__(self, params=None, learning_rate=0.001, beta1=0.9, beta2=0.999):
        super(AdaMax, self).__init__(params=params)

        self.alpha = learning_rate
        self.b1 = beta1
        self.b2 = beta2
        self.t = theano.shared(np.float32(1))

        self.ms = [build_shared_zeros(t.shape.eval(), 'm') for t in self.params]
        self.us = [build_shared_zeros(t.shape.eval(), 'u') for t in self.params]

    def updates(self, loss=None):
        """Update loss using the AdaMax formula
        
        :param loss: Vector of loss to be updated, defaults to None
        :type loss: tensor, optional

        :return: Updated parameters and gradients
        :rtype: tensor
        """
        super(AdaMax, self).updates(loss=loss)

        for m, u, param, gparam in zip(self.ms, self.us, self.params, self.gparams):
            _m = self.b1 * m + (1 - self.b1) * gparam
            _u = T.maximum(self.b2 * u, T.abs_(gparam))

            self.updates[param] = param - (self.alpha / (1 - self.b1 ** self.t)) * (_m / _u)
            self.updates[m] = _m
            self.updates[u] = _u
        self.updates[self.t] = self.t + 1.0

        return self.updates

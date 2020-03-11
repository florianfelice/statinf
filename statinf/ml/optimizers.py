
import theano
import theano.tensor as T
import numpy as np
from collections import OrderedDict


"""
References: 
https://github.com/satopirka/deep-learning-theano/blob/master/example/mlp.py
https://github.com/satopirka/deep-learning-theano/blob/master/dnn/optimizers.py
"""



def build_shared_zeros(shape, name):
    """ Builds a theano shared variable filled with a zeros numpy array """
    return theano.shared(
        value=np.zeros(shape, dtype=theano.config.floatX), 
        name=name, 
        borrow=True
    )


class Optimizer(object):
    def __init__(self, params=None):
        if params is None:
            return NotImplementedError()
        self.params = params

    def updates(self, loss=None):
        if loss is None:
            return NotImplementedError()

        self.updates = OrderedDict()
        self.gparams = [T.grad(loss, param) for param in self.params]


class SGD(Optimizer):
    """
    Args:
        learning_rate (float): Step size (defaults 0.01).
        params (list): Parameters to update, containing gradients (defaults None).
    Formula:
        * $\theta_{t} = \theta_{t-1} - \epsilon \nabla f_{t}(\theta_{t})$

    References:
        * Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep learning. MIT press.

    Returns:
        list: Update parameters
    """
    def __init__(self, learning_rate=0.01, params=None):
        super(SGD, self).__init__(params=params)
        self.learning_rate = learning_rate

    def updates(self, loss=None):
        super(SGD, self).updates(loss=loss)

        for param, gparam in zip(self.params, self.gparams):
            self.updates[param] = param - self.learning_rate * gparam

        return self.updates


class MomentumSGD(Optimizer):
    """
    Args:
        learning_rate (float): Step size (defaults 0.01).
        alpha (float): Momentum parameter (defaults 0.9).
        params (list): Parameters to update, containing gradients (defaults None).
    Formula:
        * $\theta_{t} = \theta_{t-1} + \alpha v - \epsilon \nabla f_{t}(\theta_{t})$

    References:
        * Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep learning. MIT press.

    Returns:
        list: Update parameters
    """
    def __init__(self, learning_rate=0.01, alpha=0.9, params=None):
        super(MomentumSGD, self).__init__(params=params)
        self.learning_rate = learning_rate
        # Momentum value
        self.alpha = alpha
        self.vs = [build_shared_zeros(t.shape.eval(), 'v') for t in self.params]

    def updates(self, loss=None):
        super(MomentumSGD, self).updates(loss=loss)

        for v, param, gparam in zip(self.vs, self.params, self.gparams):
            _v = self.alpha * v - self.learning_rate * gparam
            self.updates[param] = param + _v
            self.updates[v] = _v

        return self.updates


class RMSprop(Optimizer):
    """
    Args:
        learning_rate (float): Step size (defaults 0.001).
        rho (float): Decay rate (defaults 0.9).
        delta (float): Constant for division stability (defaults 10e-6).
        params (list): Parameters to update, containing gradients (defaults None).
    Formula:
        * $r = \rho r + (1- \rho) \nabla f_{t}(\theta_{t}) \odot \nabla f_{t}(\theta_{t})$
        * $\theta_{t} = \theta_{t-1} - \dfrac{\epsilon}{\sqrt{\delta + r}} \odot \nabla f_{t}(\theta_{t})$

    References:
        * Tieleman, T., & Hinton, G. (2012). Lecture 6.5-rmsprop: Divide the gradient by a running average of its recent magnitude. COURSERA: Neural networks for machine learning, 4(2), 26-31.
        * Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep learning. MIT press.

    Returns:
        list: Update parameters
    """
    def __init__(self, learning_rate=0.001, rho=0.9, delta=10e-6, params=None):
        super(RMSprop, self).__init__(params=params)

        self.learning_rate = learning_rate
        self.rho = rho
        self.delta = delta

        self.mss = [build_shared_zeros(t.shape.eval(), 'r') for t in self.params]

    def updates(self, loss=None):
        super(RMSprop, self).updates(loss=loss)

        for r, param, gparam in zip(self.mss, self.params, self.gparams):
            _r = r * self.rho
            _r += (1 - self.rho) * gparam * gparam
            self.updates[r] = _r
            self.updates[param] = param - self.learning_rate * gparam / T.sqrt(self.delta + _r)

        return self.updates


class Adam(Optimizer):
    """
    Args:
        learning_rate (float): Step size (defaults 0.001).
        beta1 (float): Exponential decay rate for first moment estimate (defaults 0.9).
        beta2 (float): Exponential decay rate for second moment estimate (defaults 0.999).
        delta (float): Small constant for numerical stability (defaults 10e-8)
        params (list): Parameters to update, containing gradients (defaults None).
    Formula:
        * $m_{t} = \beta_{1} m_{t-1} + (1 - \beta_{1}) \nabla_{\theta} f_{t}(\theta_{t-1})$
        * $v_{t} = \beta_{2} v_{t-1} + (1 - \beta_{2}) \nabla_{\theta}^{2} f_{t}(\theta_{t-1})$
        * $\hat{m}_{t} = \dfrac{m_{t}}{1 - \beta_{1}^{t}}$
        * $\hat{v}_{t} = \dfrac{v_{t}}{1 - \beta_{2}^{t}}$
        * $\theta_{t} = \theta_{t-1} - \alpha \dfrac{\hat{m}_{t}}{\sqrt{\hat{v}_{t}} + \delta}$

    References:
        * Kingma, D. P., & Ba, J. (2014). Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980 (https://arxiv.org/pdf/1412.6980.pdf)
        * Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep learning. MIT press.

    Returns:
        list: Update parameters
    """
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, delta=10e-8, params=None):
        super(Adam, self).__init__(params=params)

        self.alpha = learning_rate
        self.b1 = beta1
        self.b2 = beta2
        self.t = theano.shared(np.float32(1))
        self.delta = delta

        self.ms = [build_shared_zeros(t.shape.eval(), 'm') for t in self.params]
        self.vs = [build_shared_zeros(t.shape.eval(), 'v') for t in self.params]

    def updates(self, loss=None):
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
    """
    Args:
        learning_rate (float): Step size (defaults 0.001).
        delta (float): Constant for division stability (defaults 10e-6).
        params (list): Parameters to update, containing gradients (defaults None).
    
    Formula:
        * $r = r + \nabla f_{t}(\theta_{t}) \odot \nabla f_{t}(\theta_{t})$
        * $\theta_{t} = \theta_{t-1} - \dfrac{\epsilon}{\sqrt{\delta + r}} \odot \nabla f_{t}(\theta_{t})$

    References:
        * Duchi, J., Hazan, E., & Singer, Y. (2011). Adaptive subgradient methods for online learning and stochastic optimization. Journal of machine learning research, 12(Jul), 2121-2159. (http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf)
        * Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep learning. MIT press.

    Returns:
        list: Update parameters
    """
    def __init__(self, learning_rate=0.01, delta=1e-7, params=None):
        super(AdaGrad, self).__init__(params=params)

        self.learning_rate = learning_rate
        # Small constant for numerical stability
        self.delta = delta
        # Gradient accumulation variable
        self.accugrads = [build_shared_zeros(t.shape.eval(), 'r') for t in self.params]

    def updates(self, loss=None):
        super(AdaGrad, self).updates(loss=loss)

        for r, param, gparam in zip(self.accugrads, self.params, self.gparams):
            _r = r + gparam * gparam
            self.updates[param] = param - (self.learning_rate / T.sqrt(_r + self.delta)) * gparam
            self.updates[r] = _r

        return self.updates


class AdaMax(Optimizer):
    """
    Args:
        learning_rate (float): Step size (defaults 0.001).
        beta1 (float): Exponential decay rate for first moment estimate (defaults 0.9).
        beta2 (float): Exponential decay rate for second moment estimate (defaults 0.999).
        params (list): Parameters to update, containing gradients (defaults None).
    Formula:
        * $m_{t} = \beta_{1} m_{t-1} + (1 - \beta_{1}) \nabla_{\theta} f_{t}(\theta_{t-1})$
        * $u_{t} = \max(\beta_{2} \cdot u_{t-1}, |\nabla f_{t}(\theta_{t-1})|)$
        * $\theta_{t} = \theta_{t-1} - \alpha \dfrac{\hat{m}_{t}}{\sqrt{\hat{v}_{t}} + \epsilon}$

    References:
        * Kingma, D. P., & Ba, J. (2014). Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980 (https://arxiv.org/pdf/1412.6980.pdf)
        * Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep learning. MIT press.

    Returns:
        list: Update parameters
    """
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, params=None):
        super(AdaMax, self).__init__(params=params)

        self.alpha = learning_rate
        self.b1 = beta1
        self.b2 = beta2
        self.t = theano.shared(np.float32(1))

        self.ms = [build_shared_zeros(t.shape.eval(), 'm') for t in self.params]
        self.us = [build_shared_zeros(t.shape.eval(), 'u') for t in self.params]

    def updates(self, loss=None):
        super(AdaMax, self).updates(loss=loss)

        for m, u, param, gparam in zip(self.ms, self.us, self.params, self.gparams):
            _m = self.b1 * m + (1 - self.b1) * gparam
            _u = T.maximum(self.b2 * u, T.abs_(gparam))

            self.updates[param] = param - (self.alpha / (1 - self.b1 ** self.t)) * (_m / _u)
            self.updates[m] = _m
            self.updates[u] = _u
        self.updates[self.t] = self.t + 1.0

        return self.updates

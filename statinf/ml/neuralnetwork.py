import jax.numpy as jnp
from jax import random as jrdm
from jax import jit, vmap, grad, value_and_grad, lax
import numpy as np
import random
import matplotlib.pyplot as plt

import time
import pycof as pc

from .initializations import init_params
from .losses import mean_squared_error, binary_cross_entropy
from .activations import tanh, sigmoid, elu, relu, softplus, softmax
from .optimizers import Adam, AdaGrad, AdaMax
from .optimizers import SGD
from .optimizers import RMSprop

key = jrdm.PRNGKey(0)


all_activations = {None: None,
                   'linear': None,
                   'tanh': tanh,
                   'sigmoid': sigmoid,
                   'elu': elu,
                   'relu': relu,
                   'softplus': softplus,
                   'softmax': softmax,
                   }

all_optimizers = {'sgd': SGD,
                  'stochastic_gradient_descent': SGD,
                  'adam': Adam,
                  'adagrad': AdaGrad,
                  'adamax': AdaMax,
                  'rmsprop': RMSprop,
                  }

losses = {'mse': mean_squared_error,
          'mean_squared_error': mean_squared_error,
          'bce': binary_cross_entropy,
          'binary_cross_entropy': binary_cross_entropy,
          }


class Layer:
    def __init__(self, n_in, n_out, activation='linear', W=None, b=None, init_weights='xavier', init_bias='zeros', seed=key):

        self.type = 'Dense'
        self.activation = activation.lower()
        # self.f = T_activations[activation]
        self.activate = all_activations[activation.lower()]
        self.n_in = n_in
        self.n_out = n_out
        self.id = 1

        w_key, b_key = jrdm.split(seed)
        if W is None:
            # If weights matrix is not provided (default), initialize from a distribution
            W_values = init_params(rows=n_in, cols=n_out, method=init_weights, key=w_key)
            # Multiply by 4 the weights for sigmoid activation. See Y. Bengio's website
            if activation == "sigmoid":
                W_values *= 4
        elif W.shape == (n_in, n_out):
            raise ValueError(f'Weights dimension does not match. Dimension for W should be {(n_in, n_out)} and got {W.shape}.')

        if b is None:
            # If bias matrix is not provided (default), initialize from a distribution
            b_values = init_params(rows=1, cols=n_out, method=init_bias, key=b_key)
            # b_values = scale * jrdm.normal(b_key, (1, n_out))
            # # b_values = jnp.zeros((1, n_out))
        elif b.shape == (n_in, n_out):
            raise ValueError(f'Weights dimension does not match. Dimension for b should be {(1, n_out)} and got {b.shape}.')

        # Parameters of the model
        self.params = {'w': W_values, 'b': b_values}
        # Store parameters in the class
        self.optimal_w = W_values
        self.optimal_b = b_values

        # Save number of parameters
        self.nb_params = (W_values.shape[0] * W_values.shape[1])  # Weights
        self.nb_params += (b_values.shape[0] * b_values.shape[1])  # Bias

    def _feed_forward(self, input_x, params=None):
        if params is None:
            _p = self.params
        else:
            _p = params

        Xb_e = jnp.dot(input_x, _p['w']) + _p['b']
        self.output = Xb_e if self.activation == 'linear' else self.activate(Xb_e)
        return self.output


class MLP:
    """Multi-Layer Perceptron

    :references: Rosenblatt, F. (1958). `The perceptron: a probabilistic model for information storage and organization in the brain <http://web.engr.oregonstate.edu/~huanlian/teaching/machine-learning/2017fall/extra/rosenblatt-1958.pdf>`_. Psychological review, 65(6), 386.

    """
    def __init__(self, loss='MSE', random=None):
        self.loss = loss
        self._layers = []
        self.params = []
        self.opt_update = None
        self.get_params = None
        self.L1 = 0.
        self.L2 = 0.
        self.L1_reg = 0.
        self.L2_reg = 0.
        self._cost_value = []
        self._layer_ids = 0

    def add(self, layer):
        """Stack layer to Neural Network

        :param layer: Layer to be stacked
        :type layer: :func:`~Layer`
        """
        self._layers.append(layer)
        self._layer_ids += 1
        self._layers[-1].id = str(self._layer_ids)

    def _params(self):
        _p = {}
        for layer in self._layers:
            _p.update({layer.id: layer.params})
        return _p

    def get_weights(self, layer='all', param='all'):
        """Fetches the parameters from the network.

        :param layer: Layer id from which to fetch the parameters, defaults to 'all'
        :type layer: :obj:`int`, optional
        :param param: What parameter we need to fetch (can either be 'weight', 'bias' or 'all'), defaults to 'all'
        :type param: :obj:`str`, optional

        :return: Weights and Bias used in the Neural Network.
        :rtype: :obj:`dict`
        """
        # Check what parameter the user was to get
        if param.lower() in ['weights', 'w', 'weight']:
            par = 'w'
        elif param.lower() in ['biases', 'b', 'bias']:
            par = 'b'
        elif param.lower() == 'all':
            par = 'all'
        else:
            raise ValueError('Value for "param" is not value. Please chose between "weights", "bias" or "all".')

        weights = self._params()

        if layer == 'all':
            # If user wants to see all layers, we create a dictionnary
            if par != 'all':
                weights = {k: v[par] for k, v in list(weights.items())}
        elif layer in range(1, len(self._layers) + 1):
            # If user wants only 1 specific layer,
            weights = weights[str(layer)]
            if par != 'all':
                weights = weights[par]
        else:
            raise ValueError(f'Layer is incorrect. Please chose either "all" or layer <= {len(self._layers)}. Got layer = {layer}.')

        return weights

    def _forward_prop(self, x, params=None):
        output = x
        if params is None:
            for layer in self._layers:
                output = layer._feed_forward(output)
        else:
            for layer in self._layers:
                output = layer._feed_forward(output, params[layer.id])
        return output

    def _L1(self):
        """Compute L1 regularization.

        :formula: .. math:: L_{1} = \\sum_{i=1}^{L} |\\beta|

        :return: L1 penalty
        :rtype: :obj:`float`
        """
        if self.L1_reg != 0.:
            for layer in self._layers:
                # L1 norm ; one regularization option is to enforce L1 norm to
                # be small
                self.L1 += jnp.sum(jnp.absolute(layer.params['w']))
        return self.L1

    def _L2(self):
        """Compute L2 regularization.

        :formula: .. math:: L_{2} = \\sum_{i=1}^{L} \\beta^{2}

        :return: L2 penalty
        :rtype: :obj:`float`
        """
        if self.L2_reg != 0.:
            for layer in self._layers:
                # square of L2 norm ; one regularization option is to enforce
                # square of L2 norm to be small
                self.L2 += jnp.sum(jnp.square(layer.params['w']))
        return self.L2

    def _cost(self, params, x, y):
        """ Compute the multi-class cross-entropy loss """
        output = self._forward_prop(x, params)
        return losses[self.loss.lower()](y_true=y, y_pred=output) + (self.L1_reg * self._L1()) + (self.L2_reg * self._L2())
        # return jnp.mean(jnp.square(output - y)) + (self.L1_reg * self._L1()) + (self.L2_reg * self._L2())

    def _update(self, params, x, y):
        # return [(w - 0.05 * dw, b - 0.05 * db) for (w, b), (dw, db) in zip(params, grads)]

        grads = grad(self._cost)(params, x, y)

        self.params = self.opt.update(params=params, grads=grads)

        return self.params

    def fit(self, data, X, Y='Y', epochs=100, optimizer='SGD', batch_size=1, training_size=0.8, test_set=None, learning_rate=0.05,
            L1_reg=0., L2_reg=0., early_stop=True, patience=10, improvement_threshold=0.995, restore_weights=True, verbose=True, verbose_all=False,
            plot=False, *args):
        """Train the Neural Network.

        :param data: Layer id from which to fetch the parameters.
        :type data: :obj:`pandas.DataFrame`
        :param X: List of X input variables.
        :type X: :obj:`list`
        :param Y: Variable to predict, defaults to 'Y'.
        :type Y: :obj:`str`, optional
        :param epochs: Number of epochs to train the network, defaults to 100.
        :type epochs: :obj:`int`, optional
        :param optimizer: Algortihm to use to minimize the loss function, defaults to 'SGD' (see `optimizers <optimizers.html>`_).
        :type optimizer: :obj:`str`, optional
        :param batch_size: Size of each batch to be trained, defaults to 1.
        :type batch_size: :obj:`int`, optional
        :param training_size: Ratio of the data to be used for training set (:math:`\\in (0, 1)`) the remainder is used for test set, defaults to 0.8.
        :type training_size: :obj:`float`, optional
        :param test_set: Data frame to use as test set (overrides **training_size** if provided), defaults to None.
        :type test_set: :obj:`pandas.DataFrame`, optional
        :param learning_rate: Learning rate (step size) for gradient descent, defaults to 0.01.
        :type learning_rate: :obj:`float`, optional
        :param L1_reg: Coefficient :math:`\\lambda_{1} \\in (0,1)` for the L1 penalty, defaults to 0.
        :type L1_reg: :obj:`float`, optional
        :param L2_reg: Coefficient :math:`\\lambda_{2} \\in (0,1)` for the L2 penalty, defaults to 0.
        :type L2_reg: :obj:`float`, optional
        :param early_stop: Apply early-stopping method to avoid over-fitting. Stop the training sequence after the patience (see below) with no improvement, defaults to True.
        :type early_stop: :obj:`bool`, optional
        :param patience:  Number of iterations to wait after a minimum has been found. If no improvement, training sequence is stopped, defaults to 10.
        :type patience: :obj:`int`, optional
        :param improvement_threshold: Change level of the loss from which we consider it is an improvement, defaults to 0.995.
        :type improvement_threshold: :obj:`float`, optional
        :param restore_weights: If training is early-stopped, restore the weights of the best iteration, defaults to True.
        :type restore_weights: :obj:`bool`, optional
        :param verbose: Verbosity level (True or 1 for minimal display, 2 for details), defaults to True.
        :type verbose: :obj:`int`, optional
        :param plot: Plot the evolution of the training and test loss for each epoch, defaults to False.
        :type plot: :obj:`bool`, optional
        :param \\*args: Arguments to be passed to the optimizer.
        :type \\*args: :obj:`str`

        :references: * Friedman, J., Hastie, T., & Tibshirani, R. (2001). `The elements of statistical learning <https://web.stanford.edu/~hastie/Papers/ESLII.pdf>`_ (Vol. 1, No. 10). New York: Springer series in statistics.
            * Goodfellow, I., Bengio, Y., & Courville, A. (2016). `Deep learning <https://www.deeplearningbook.org/>`_. MIT press.
        """

        # Parse arguments to pass to self
        self.learning_rate = learning_rate
        self.L1_reg = L1_reg
        self.L2_reg = L2_reg
        self.X_col = X
        self.Y_col = Y
        self.optimizer = str(optimizer).lower()

        verbose = True if verbose in [True, 1, 2] else False
        verbose_all = True if verbose in [2] else False

        # Get indexes
        self.data_size = data.shape[0]

        if test_set is None:
            data_idx = list(data.index)
            train_index = random.sample(data_idx, int(self.data_size * training_size))
            test_index = [i for i in data_idx if i not in train_index]
            # Train set
            train_set_x = data.loc[train_index, self.X_col].values
            n_train_set = train_set_x.shape[0]
            train_set_y = jnp.reshape(data.loc[train_index, self.Y_col].values, n_train_set)
            # Test set
            test_set_x = data.loc[test_index, self.X_col].values
            n_test_set = test_set_x.shape[0]
            test_set_y = jnp.reshape(data.loc[test_index, self.Y_col].values, n_test_set)
        else:
            # Train set
            train_set_x = data[self.X_col].values
            n_train_set = train_set_x.shape[0]
            train_set_y = jnp.reshape(data[self.Y_col].values, n_train_set)
            # Test set
            test_set_x = test_set[self.X_col].values
            n_test_set = test_set_x.shape[0]
            test_set_y = jnp.reshape(test_set[self.Y_col].values, n_test_set)

        n_train_batches = n_train_set // batch_size

        params = self._params()

        # Initialize values
        self.train_losses = []   # Training loss vector to plot
        self.test_losses = []    # Test loss vector to plot
        self.best_loss = np.inf  # Initial best loss value
        self.best_iter = 1
        verif_i = epochs         # First, verify epoch at then end (init only)
        i = 1

        # self.opt = SGD(params=params, learning_rate=self.learning_rate)
        self.opt = all_optimizers[self.optimizer](learning_rate=self.learning_rate)

        while i <= epochs:
            start_time = time.time()
            for b in range(n_train_batches):
                x_batch = train_set_x[b * batch_size:b * batch_size + batch_size, :]
                y_batch = train_set_y[b * batch_size:b * batch_size + batch_size]
                # Update parameters (feed forward + back-propagation)
                params = self._update(params, x_batch, y_batch)
            epoch_time = time.time() - start_time

            # COmpute loss on global train set
            train_loss = self._cost(params, train_set_x, train_set_y)
            # Evaluate model iteration on test set
            test_loss = self._cost(params, test_set_x, test_set_y)

            # Save iteration losses (for both train and test)
            self.train_losses += [train_loss]
            self.test_losses += [test_loss]

            if (test_loss < self.best_loss * improvement_threshold) & (i <= verif_i) & early_stop:
                # If a new minimum has been found
                pc.verbose_display(f'New min found for iter {i} -> train loss = {train_loss} - test loss {test_loss}', verbose=verbose_all)
                verif_i = i + patience
                self.best_loss = test_loss
                self.best_iter = i
                for layer in self._layers:
                    layer.optimal_w = layer.params['w']
                    layer.optimal_b = layer.params['b']
            elif (i < verif_i):
                # If no improvement
                pc.verbose_display(f'No improvement at iter {i}', verbose=verbose)
                pass
            else:
                # If no improvement found after patience, stop training
                pc.verbose_display(f'Stop training. Minimum found after {self.best_iter} iterations', verbose=verbose)
                if restore_weights:
                    pc.verbose_display(f'Restoring weights from epoch {self.best_iter}', verbose=verbose)
                    for layer in self._layers:
                        layer.params['w'] = layer.optimal_w
                        layer.params['b'] = layer.optimal_b
                break

            i += 1

        if plot:
            # Plot the training and test losses
            plt.title('MLP train vs. test error through epochs', loc='center')
            plt.plot(self.train_losses, label='Training loss')
            plt.plot(self.test_losses, label='Test loss')
            plt.legend()
            plt.show()

    def predict(self, new_data, binary=False, threshold=0.5):
        """Generates output prediction after feedforward pass.

        :param new_data: Input data.
        :type new_data: :obj:`pandas.DataFrame`
        :param binary: Boolean for returning a binary output (not probability), defaults to False.
        :type binary: :obj:`bool`, optional
        :param threshold: Probability threshold for binary response, defaults to 0.5.
        :type threshold: :obj:`float`, optional

        :return: Predicted values.
        :rtype: :obj:`list`
        """
        # Transform input data to be transformed into numpy array format
        x = new_data[self.X_col].values
        # Apply feedforward step to the data to get the prediction (apply weights and activations)
        output = self._forward_prop(x)
        if binary:
            return [1. if y[0] > threshold else 0. for y in output.reshape((new_data.shape[0], 1))]
        else:
            return [y[0] for y in output.reshape((new_data.shape[0], 1))]

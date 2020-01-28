import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

import numpy as np
from .losses import *

np.random.seed(100)


class NeuralNetwork:
    """
    Represents a neural network.
    """
    
    def __init__(self, n_input, n_neurons, activation, weights=None, bias=None, weights_init=None, bias_init=None):
        """
        :param int n_input: The input size (coming from the input layer or a previous hidden layer)
        :param int n_neurons: The number of neurons in this layer.
        :param str activation: The activation function to use (if any).
        :param weights: The layer's weights.
        :param bias: The layer's bias.
        """
        
        self._layers = []
        self.input_dim = n_input
        self.output_dim = n_neurons
        self.w_method = weights_init.lower() if weights_init is not None else 'ones'
        self.b_method = bias_init.lower() if bias_init is not None else 'zeros'
        self.bias = bias if bias is not None else self.init_params(param='bias')
        self.weights = weights if weights is not None else self.init_params(param='weights')
        self.activation = activation
        self.last_activation = None
        self.error = None
        self.delta = None

    def init_params(self, param='weights'):
        """
        """
        if param == 'weights':
            # Initialize the weights
            if self.w_method == 'random':
                w = np.random.randn(self.output_dim, self.input_dim)
            elif self.w_method == 'uniform':
                w = np.random.uniform(-1, 1, (self.output_dim, self.input_dim))
            elif self.w_method == 'ones':
                w = np.ones((self.output_dim, self.input_dim))
            else:
                raise ValueError(f"{self.w_method} is not supported for weights initializer.")
        elif param == 'bias':
            # Initialize the bias
            if self.b_method == 'random':
                w = np.random.randn(self.output_dim, 1)
            elif self.b_method == 'zeros':
                w = np.zeros((self.output_dim, 1))
            else:
                raise ValueError(f"{self.b_method} is not supported for bias initializer.")
        else:
            raise ValueError(f"{param} is not an accepted parameter. Please chose either 'weights' or 'bias'.")
        return w


    def activate(self, x):
        """
        Calculates the dot product of this layer.
        :param x: The input.
        :return: The result.
        """

        r = np.dot(x.T, self.weights) + self.bias
        self.last_activation = self._apply_activation(r)
        return self.last_activation

    def _apply_activation(self, r):
        """
        Applies the chosen activation function (if any).
        :param r: The normal value.
        :return: The "activated" value.
        """

        # In case no activation function was chosen
        if self.activation == 'linear':
            # Linear
            return r
        elif self.activation == 'tanh':
            # tanh
            return np.tanh(r)
        elif self.activation == 'sigmoid':
            # sigmoid
            return 1 / (1 + np.exp(-r))
        else:
            return r

    def apply_activation_derivative(self, r):
        """
        Applies the derivative of the activation function (if any).
        :param r: The normal value.
        :return: The "derived" value.
        """

        # We use 'r' directly here because its already activated, the only values that
        # are used in this function are the last activations that were saved.

        if self.activation == "linear":
            return 1
        elif self.activation == 'tanh':
            return 1 - r ** 2
        elif self.activation == 'sigmoid':
            return r * (1 - r)
        else:
            raise ValueError(f'{self.activation} is not a valid activation function')


    def add_layer(self, layer):
        """
        Adds a layer to the neural network.
        :param Layer layer: The layer to add.
        """

        self._layers.append(layer)

    def feed_forward(self, X):
        """
        Feed forward the input through the layers.
        :param X: The input values.
        :return: The result.
        """

        for layer in self._layers:
            X = layer.activate(X)

        return X

    def predict(self, X):
        """
        Predicts a class (or classes).
        :param X: The input values.
        :return: The predictions.
        """

        ff = self.feed_forward(X)

        # One row
        if ff.ndim == 1:
            return np.argmax(ff)

        # Multiple rows
        return np.argmax(ff, axis=1)

    def backpropagation(self, X, y, output):
        """
        Performs the backward propagation algorithm and updates the layers weights.
        :param X: The input values.
        :param y: The target values.
        """
        

        # Loop over the layers backward
        for i in reversed(range(len(self._layers))):
            layer = self._layers[i]

            # If this is the output layer
            if layer == self._layers[-1]:
                layer.error = y - output
                # The output = layer.last_activation in this case
                layer.delta = layer.error * layer.apply_activation_derivative(output)
            else:
                next_layer = self._layers[i + 1]
                layer.error = np.dot(next_layer.weights, next_layer.delta)
                layer.delta = layer.error * layer.apply_activation_derivative(layer.last_activation)

        # Update the weights
        for i in range(len(self._layers)):
            layer = self._layers[i]
            # The input is either the previous layers output or X itself (for the first hidden layer)
            input_to_use = np.atleast_2d(X if i == 0 else self._layers[i - 1].last_activation)
            layer.weights += layer.delta * input_to_use.T * self.learning_rate

    def train(self, data, y_col, learning_rate, max_epochs, loss=mean_squared_error):
        """
        Trains the neural network using backpropagation.
        :param X: The input values.
        :param y: The target values.
        :param float learning_rate: The learning rate (between 0 and 1).
        :param int max_epochs: The maximum number of epochs (cycles).
        :return: The list of calculated MSE errors.
        """

        self.X_col = [c for c in data.columns if c != y_col]

        self.X = data[self.X_col].to_numpy().T
        self.y = data[y_col].to_numpy().reshape((data[y_col].to_numpy().shape[0], -1)).T

        self.learning_rate = learning_rate

        self.loss_history = []

        for i in range(max_epochs):
            #for j in range(len(self.X)):
            # Feed forward for the output
            output = self.feed_forward(self.X)
            # Back propagate
            self.backpropagation(self.X, self.y, output)
            if i % 10 == 0:
                loss_val = loss(y_true=self.y, y_pred=nn.feed_forward(self.X))
                #np.mean(np.square(self.y - nn.feed_forward(self.X)))
                self.loss_history = self.loss_history.append(loss_val)
                print('Epoch: #%s, Loss: %f' % (i, float(loss_val)))

        return self.loss_history



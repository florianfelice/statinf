import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

import numpy as np
from .losses import *
from .activations import tanh, sigmoid
from .activations import d_tanh, d_sigmoid
from .initilizations import init_weights, init_bias



"""
# Credits:
Original code from: https://blog.zhaytam.com/2018/08/15/implement-neural-network-backpropagation/
"""

## TODO: Use class for non binary output
## TODO: Merge Layer and NeuralNetwork class for add_layer function?
## TODO: Make function use pandas

np.random.seed(100)


class Layer:
    """
    Represents a layer (hidden or output) in our neural network.
    """

    def __init__(self, n_input, n_neurons, activation='linear', weights='ones', bias='zeros'):
        """
        :param int n_input: The input size (coming from the input layer or a previous hidden layer)
        :param int n_neurons: The number of neurons in this layer.
        :param str activation: The activation function to use (if any).
        :param weights: The layer's weights.
        :param bias: The layer's bias.
        """
        
        self.weights = weights if type(weights) != str else init_weights(rows=n_input, cols=n_neurons, method=weights)
        self.activation = activation
        self.bias = bias if type(bias) != str else init_bias(cols=n_neurons, method=bias)
        self.last_activation = None
        self.error = None
        self.delta = None

    def activate(self, x, verbose=False):
        """
        Calculates the dot product of this layer.
        :param x: The input.
        :return: The result.
        """
        
        self.last_activation = self._apply_activation(x)
        # print(self.last_activation)
        return self.last_activation

    def _apply_activation(self, r):
        """
        Applies the chosen activation function (if any).
        :param r: The normal value.
        :return: The "activated" value.
        """

        # linear
        if self.activation == 'linear':
            return r
        # tanh
        if self.activation == 'tanh':
            return tanh(r)
        # sigmoid
        if self.activation == 'sigmoid':
            return sigmoid(r)

    def apply_activation_derivative(self, r):
        """
        Applies the derivative of the activation function (if any).
        :param r: The normal value.
        :return: The "derived" value.
        """

        # We use 'r' directly here because its already activated, the only values that
        # are used in this function are the last activations that were saved.

        if self.activation == 'linear':
            return [1.]*len(r)

        if self.activation == 'tanh':
            return 1 - r ** 2

        if self.activation == 'sigmoid':
            return r * (1 - r)


class NeuralNetwork:
    """
    Represents a neural network.
    """

    def __init__(self, loss='MSE'):
        self._layers = []
        self.loss = loss

    def add_layer(self, layer):
        """
        Adds a layer to the neural network.
        :param Layer layer: The layer to add.
        """

        self._layers.append(layer)

    def feed_forward(self, X, verbose=False):
        """
        Feed forward the input through the layers.
        :param X: The input values.
        :return: The result.
        """

        for layer in self._layers:
            # print(layer)
            Xb = X.dot(layer.weights)
            if verbose:
                print(layer.weights)
            Xb_e = Xb + layer.bias
            X = layer.activate(Xb_e, verbose=verbose)

        return X

    def predict(self, X, binary_threshold=None):
        """
        Predicts a class (or classes).
        :param X: The input values.
        :return: The predictions.
        """

        ff = self.feed_forward(X)

        # One row
        # if ff.ndim == 1:
        #     return np.argmax(ff)

        # # Multiple rows
        # return np.argmax(ff, axis=1)
        if binary_threshold is None:
            return ff
        else:
            return np.array([0 if x < binary_threshold else 1 for x in ff])

    def backpropagation(self, X, y, learning_rate):
        """
        Performs the backward propagation algorithm and updates the layers weights.
        :param X: The input values.
        :param y: The target values.
        :param float learning_rate: The learning rate (between 0 and 1).
        """
        # print('# Feedforward')
        # Feed forward for the output
        output = self.feed_forward(X, verbose=False)
        # print(output)
        # Store evolution of the last predicted value
        self.last_predicted = output

        # print('# Feedforward')
        # Loop over the layers backward
        for i in reversed(range(len(self._layers))):
            layer = self._layers[i]

            # If this is the output layer
            if layer == self._layers[-1]:
                # print('- Last layer')
                layer.error = y - output
                # print(output)
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
            layer.weights += layer.delta * input_to_use.T * learning_rate

    def compute_loss(self, y_true, y_pred, loss=None, verbose=False):
        """
        """
        ls = loss if loss is not None else self.loss
        if ls.upper() in ['MSE', 'MEAN_SQUARED_ERROR']:
            return mean_squared_error(y_true=y_true, y_pred=y_pred, verbose=verbose)
        elif ls.upper() in ['BINARY_CROSS_ENTROPY']:
            return binary_cross_entropy(y_true=y_true, y_pred=y_pred, verbose=verbose)
        elif ls.upper() in ['BINARY_LOSS']:
            return (y_pred == y_true).mean()
        else:
            raise ValueError('Loss function is not valid')
    
    def train(self, X, y, max_epochs, learning_rate=0.1, verbose_every=50, plot=False):
        """
        Trains the neural network using backpropagation.
        :param X: The input values.
        :param y: The target values.
        :param float learning_rate: The learning rate (between 0 and 1).
        :param int max_epochs: The maximum number of epochs (cycles).
        :return: The list of calculated MSE errors.
        """

        losses = []

        for i in range(max_epochs):
            for j in range(len(X)):
                # print(f'X = {X[j]}, Y = {y[j]}')
                # print(j)
                self.backpropagation(X[j], y[j], learning_rate)
            # print(self.last_predicted)
            ## Compute accuracy
            #mse = np.mean(np.square(y - self.feed_forward(X)))
            ## Feedforward with updated weights
            ff_update = self.feed_forward(X)
            acc = self.compute_loss(y_true=y, y_pred=ff_update, loss=self.loss)
            losses.append(acc)
            
            # Display progress
            if i % verbose_every == 0:
                print('Epoch: #%s, MSE: %f' % (i, float(acc)))
        #return losses

        if plot:
            plt.plot(losses)
            plt.title('Changes in loss function')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            return plt.show()
        else:
            return losses
    

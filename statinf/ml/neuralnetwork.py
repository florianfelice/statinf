import theano
import theano.tensor as T
import numpy as np
import random
import matplotlib.pyplot as plt

import pycof as pc

from .optimizers import Adam, AdaGrad, AdaMax
from .optimizers import SGD, MomentumSGD
from .optimizers import RMSprop
from .initilizations import init_params
from .activations import tanh, sigmoid, relu, softplus
from .losses import mean_squared_error, binary_cross_entropy

"""
DeepLearning website - Yoshua Bengio:
- http://deeplearning.net/tutorial/mlp.html
"""

## TODO: add batch
## TODO: comment out

T_activations = {None: None,
                'linear': None,
                'tanh': T.tanh,
                'sigmoid': T.nnet.sigmoid,
                'relu': T.nnet.relu,
                'softplus': T.nnet.softplus,
                }

activations = {None: None,
                'linear': None,
                'tanh': tanh,
                'sigmoid': sigmoid,
                'relu': relu,
                'softplus': softplus,
                }

optimizers = {'sgd': SGD,
                'stochastic_gradient_descent': SGD,
                'momentum': MomentumSGD,
                'momentumsgd': MomentumSGD,
                'adam': Adam,
                'adagrad': AdaGrad,
                'adamax': AdaMax,
                'rmsprop': RMSprop,
                }


class Layer(object):
    def __init__(self, n_in, n_out, W=None, b=None, activation=None, seed=None, init_weights='xavier', init_bias='zeros'):
      
        self.input = input
        self.activation = activation
        self.f = T_activations[activation]
        self.activate = activations[activation]

        if W is None:
            # If weights matrix is not provided (default), initialize from a distribution
            W_values = init_params(rows=n_in, cols=n_out, method=init_weights, seed=seed)
            # Multiply by 4 the weights for sigmoid activation. See Y. Bengio's website
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4
        elif W.shape == (n_in, n_out):
            raise ValueError(f'Weights dimension does not match. Dimension for W should be {(n_in, n_out)} and got {W.shape}.')

        if b is None:
            # If bias matrix is not provided (default), initialize from a distribution
            b_values = init_params(rows=1, cols=n_out, method=init_bias, seed=seed)
        elif b.shape == (n_in, n_out):
            raise ValueError(f'Weights dimension does not match. Dimension for b should be {(1, n_out)} and got {b.shape}.')
        
        # Store parameters in the class as shared parameters with theano
        self.W = theano.shared(value=W_values, name='W', borrow=True)
        self.b = theano.shared(value=b_values, name='b', borrow=True)
        self.optimal_w = W_values
        self.optimal_b = b_values
        
        # Parameters of the model
        self.params = [self.W, self.b]

    def feed_forward(self, input, tensor=True):
        if tensor:
            Xb_e = T.dot(input, self.W) + self.b
            self.output = (Xb_e if self.activation in [None, 'linear'] else self.f(Xb_e))
        else:
            # Used as feed forward for prediction (avoid using Theano)
            Xb_e = input.dot(self.W.get_value()) + self.b.get_value()
            self.output = (Xb_e if self.activation in [None, 'linear'] else self.activate(Xb_e))
        return self.output

class MLP:
    def __init__(self, loss='MSE', random=None):
        self.loss = loss
        self._layers = []
        self.params = []
        self.L1 = 0.
        self.L2 = 0.
        self._cost = []

    def _L1(self):
        
        for layer in self._layers:
            # L1 norm ; one regularization option is to enforce L1 norm to
            # be small
            self.L1 += abs(layer.W).sum()
        return self.L1
    
    def _L2(self):
        for layer in self._layers:
            # square of L2 norm ; one regularization option is to enforce
            # square of L2 norm to be small
            self.L2 += (layer.W ** 2).sum()
        return self.L2
    
    def _params(self):
        # the parameters of the model are the parameters of the two layer it is
        # made out of
        _params = []
        for layer in self._layers:
            _params += list(layer.params)
        
        self.params = list(_params)
    
    def add(self, layer):
        
        self._layers.append(layer)
    
    def forward_prop(self, x, tensor=True):
        output = x
        for layer in self._layers:
            output = layer.feed_forward(output, tensor)
        return output

    def cost(self, x, y):
        output = self.forward_prop(x)
        # Compute loss
        if self.loss.lower() in ['mse', 'mean_squared_error']:
            _cost = mean_squared_error(y_true=y, y_pred=output) # ((output - y) ** 2).sum()
        elif self.loss.lower() in ['binary_cross_entropy', 'bce']:
            _cost = binary_cross_entropy(y_true=y, y_pred=output)
        else:
            raise ValueError('Loss function is not valid.')
        self._cost += [_cost]
        return _cost
    
    def get_weights(self, layer='all', param='all'):
        """
        Fetches the parameters from the network.
        
        Parameters
        ----------
        layer: int
            Layer id from which to fetch the parameters (defaults 'all').
        param: str
            What parameter we need to fetch (defaults 'all').
        
        Returns
        -------
        weights: dict
            The parameters in the network.
        """
        # Check what parameter the user was to get
        if param.lower() in ['weights', 'w', 'weight']:
            par = 0
        elif param.lower() in ['biases', 'b', 'bias']:
            par = 1
        elif param.lower() == 'all':
            par = 2
        else:
            raise ValueError('Value for "param" is not value. Please chose between "weights", "bias" or "all".')
        
        if layer == 'all':
            # If user wants to see all layers, we create a dictionnary
            weights = {}
            for i in range(len(self._layers)):
                if par == 0:
                    weights.update({f'weights {i}': self._layers[i].W.get_value()})
                elif par == 1:
                    weights.update({f'bias {i}': self._layers[i].b.get_value()})
                else:
                    weights.update({f'weights {i}': self._layers[i].W.get_value(), f'bias {i}': self._layers[i].b.get_value()})
        elif layer in range(len(self._layers)):
            # If user wants only 1 specific layer,
            if par == 2:
                # we return a dict for all params
                weights = {'weights': self._layers[layer].Z.get_value(), 'bias': self._layers[layer].b.get_value()}
            else:
                # or an array for 1 single param
                weights = self._layers[layer].Z.get_value()
        else:
            raise ValueError(f'Layer is incorrect. Please chose either "all" or layer <= {len(self._layers) - 1}. Got layer = {layer}')
        
        return weights
 
    def train(self, data, X, Y='Y', epochs=100, optimizer='SGD', batch_size=1, training_size=0.8, test_set=None, learning_rate=0.01, L1_reg=0., L2_reg=0., early_stop=True, patience=10, improvement_threshold=0.995, restore_weights=True, verbose=True, verbose_all=False, plot=False):
        """
        Train the Neural Network.
        
        Parameters
        ----------
        data: pd.DataFrame
            Layer id from which to fetch the parameters.
        X: list
            List of X input variables.
        Y: str
            Variable to predict (defaults 'Y').
        epochs: int
            Number of epochs to train (defaults 100).
        optimizer: str
            Algortihm to use to optimize the loss function (defaults 'SGD')
        batch_size: int
            Size of each batch to be trained (defaults 1).
        learning_rate: float
            Learning rate for gradient descent (defaults 0.01).
        patience: int
            Look as this many examples regardless (defaults 100).
        improvement_threshold: float
            Relative improvement to be considered as significant (defaults 0.995).
        L1_reg: float
            L1 regularization rate (defaults 0.0)
        L2_reg: float
            L2 regularization rate (defaults 0.0)
        
        Returns
        -------
        weights: dict
            The parameters in the network.
        """
        # Parse arguments to pass to self
        self.learning_rate = learning_rate
        self.L1_reg = L1_reg
        self.L2_reg = L2_reg
        self.X_col = X
        self.Y_col = Y
        self.optimizer = str(optimizer).lower()

        # Get indexes
        self.data_size = data.shape[0]

        if test_set is None:
            data_idx = list(data.index)
            train_index = random.sample(data_idx, int(self.data_size * training_size))
            test_index = [i for i in data_idx if i not in train_index]
            # Train set
            train_set_x = data.loc[train_index, self.X_col]
            train_set_y = data.loc[train_index, self.Y_col]
            # Test set
            test_set_x = data.loc[test_index, self.X_col]
            test_set_y = data.loc[test_index, self.Y_col]
        else:
            # Train set
            train_set_x = data[self.X_col]
            train_set_y = data[self.Y_col]
            # Test set
            test_set_x = test_set[self.X_col]
            test_set_y = test_set[self.Y_col]
        
        # Set data to theano
        # Train set
        train_set_x = theano.shared(np.array(train_set_x, dtype=theano.config.floatX))
        train_set_y = theano.shared(np.array(train_set_y, dtype=theano.config.floatX))
        # Test set
        test_set_x = theano.shared(np.array(test_set_x, dtype=theano.config.floatX))
        test_set_y = theano.shared(np.array(test_set_y, dtype=theano.config.floatX))
        self.test_set = test_set_x

        index = T.lscalar()  # index to a [mini]batch
        x = T.vector('x')
        y = T.scalar('y')

        # Get initial parameters
        self._params()

        # Set cost
        cost = self.cost(x, y) + (self.L1_reg * self._L1()) + (self.L2_reg * self._L2())
        
        ## Define update with optimizer
        try:
            self.updates = optimizers[self.optimizer](params=self.params, learning_rate=self.learning_rate).updates(cost)
        except BaseException:
            posible_opt = "', '".join([i for i in optimizers.keys()])
            raise ValueError(f"Optimizer not value. Please make sure you choose in '{posible_opt}'. Got '{self.optimizer}'.")
 
        # Define Theano function for training step
        train_model = theano.function(
            inputs=[index],
            outputs=cost,
            updates=self.updates,
            givens={x: train_set_x[index], y: train_set_y[index]}
        )
        
        # Define Theano function for validation step
        validate_model = theano.function(
            inputs=[index],
            outputs=cost,
            givens={x: test_set_x[index], y: test_set_y[index]}
        )
        # Define length of test
        n_train_set = train_set_x.get_value(borrow=True).shape[0]
        n_test_set = test_set_x.get_value(borrow=True).shape[0]
        
        # Initialize values
        self.train_losses = []       # Training loss vector to plot
        self.test_losses = []        # Test loss vector to plot
        self.best_loss = np.inf # Initial best loss value
        self.best_iter = 0
        verif_i = epochs        # First, verify epoch at then end (init only)
        i = 0

        # Start training
        while i <= epochs:
            i += 1
            # Measure accuracy on train set
            epoch_loss = np.mean([train_model(i) for i in range(n_train_set)])
            # Evaluate on test set
            epoch_loss_test = np.mean([validate_model(i).tolist() for i in range(n_test_set)])
            # Save to historical performance
            self.train_losses += [epoch_loss]
            self.test_losses += [epoch_loss_test]
            # If we got the best score until now, and patience is not reached
            if (epoch_loss_test < self.best_loss * improvement_threshold) & (i <= verif_i) & early_stop:
                pc.verbose_display(f'New min found for iter {i}', verbose=verbose_all)
                verif_i = i + patience
                self.best_loss = epoch_loss_test
                self.best_iter = i
                # self.optimal_w = self.w.get_value()
                # self.optimal_b = self.b.get_value()
                for layer in self._layers:
                    layer.optimal_w = layer.W.get_value()
                    layer.optimal_b = layer.b.get_value()
            # if no improvement
            elif (i < verif_i):
                # pc.verbose_display(f'No improvement at iter {i}', verbose=verbose_all)
                pass
            else:
                pc.verbose_display(f'Stop training. Minimum found after {self.best_iter} iterations', verbose=verbose)
                if restore_weights:
                    for layer in self._layers:
                        layer.W = theano.shared(value=layer.optimal_w, name='W', borrow=True)
                        layer.b = theano.shared(value=layer.optimal_b, name='b', borrow=True)
                        layer.params = [layer.W, layer.b]
                break

        if plot:
            # Plot the training and test loss
            plt.title('MLP train vs. test error through epochs', loc='center')
            plt.plot(self.train_losses, label='Training loss')
            plt.plot(self.test_losses, label='Test loss')
            plt.legend()
            plt.show()
    

    def predict(self, new_data, binary=False, threshold=0.5):
        """
        """
        # Transform input data to be transformed into numpy array format
        x = new_data[self.X_col].values
        # Apply feedforward step to the data to get the prediction (apply weights and activations)
        output = self.forward_prop(x, tensor=False)
        if binary:
            return [1. if y[0] > threshold else 0. for y in output.reshape((new_data.shape[0], 1))]
        else:
            return [y[0] for y in output.reshape((new_data.shape[0], 1))]

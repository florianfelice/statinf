import theano
import theano.tensor as T
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt


"""
DeepLearning website - Yoshua Bengio:
- http://deeplearning.net/tutorial/mlp.html
"""

## TODO: add batch
## TODO: comment out
        
class Layer(object):
    def __init__(self, n_in, n_out, W=None, b=None, activation=None, seed=None):
      
        self.input = input

        if W is None:
            rdm = np.random.RandomState(seed) if seed is not None else np.random
            W_values = np.asarray(rdm.normal(1., 0., n_in * n_out).reshape((n_in, n_out)), dtype=theano.config.floatX)
            W_values = np.asarray(rdm.uniform(
                    low = -np.sqrt(6. / (n_in + n_out)),
                    high = np.sqrt(6. / (n_in + n_out)),
                    size = (n_in, n_out)), dtype=theano.config.floatX)
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b
        # parameters of the model
        self.params = [self.W, self.b]

    def feed_forward(self, input, activation=None):
        lin_output = T.dot(input, self.W) + self.b
        self.output = (lin_output if activation in [None, 'linear'] else activation(lin_output))
        return self.output

class MLP:
    def __init__(self, loss='MSE', optimizer='sgd', random=None):
        self.loss = loss
        self.optimizer = optimizer
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
    
    def cost(self, x, y):
        output = x
        for layer in self._layers:
            output = layer.feed_forward(output)
        if self.loss.lower() in ['mse', 'mean_squared_error']:
            _cost = ((output - y) ** 2).sum()
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
                    weights.update({f'weights{i}': self._layers[i].params[0].get_value()})
                elif par == 1:
                    weights.update({f'bias {i}': self._layers[i].params[1].get_value()})
                else:
                    weights.update({f'weights {i}': self._layers[i].params[0].get_value(), f'bias {i}': self._layers[i].params[1].get_value()})
        elif layer in range(len(self._layers)):
            # If user wants only 1 specific layer,
            if par == 2:
                # we return a dict for all params
                weights = {'weights': self._layers[layer].params[0].get_value(), 'bias': self._layers[layer].params[1].get_value()}
            else:
                # or an array for 1 single param
                weights = self._layers[layer].params[0].get_value()
        else:
            raise ValueError(f'Layer is incorrect. Please chose either "all" or layer <= {len(self._layers) - 1}. Got layer = {layer}')
        
        return weights
 
    def train(self, data, X, Y='Y', epochs=100, batch_size=1, training_size=0.8, learning_rate=0.01, L1_reg=0., L2_reg=0.):
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
        batch_size: int
            Size of each batch to be trained (defaults 1).
        learning_rate: float
            Learning rate for gradient descent (defaults 0.01).
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

        # Prepare K-fold
        # Get indexes
        self.data_size = data.shape[0]
        data_idx = list(data.index)
        train_index = random.sample(data_idx, int(self.data_size * training_size))
        test_index = [i for i in data_idx if i not in train_index]

        # Train set
        valid_set_x = data.loc[train_index, X]
        valid_set_y = data.loc[train_index, Y]
        # Set data to theano
        valid_set_x = theano.shared(np.array(valid_set_x, dtype=theano.config.floatX))
        valid_set_y = theano.shared(np.array(valid_set_y, dtype=theano.config.floatX))
        
        # Test set
        test_set_x = data.loc[test_index, X]
        test_set_y = data.loc[test_index, Y]
        # Set data to theano
        test_set_x = theano.shared(np.array(test_set_x, dtype=theano.config.floatX))
        test_set_y = theano.shared(np.array(test_set_y, dtype=theano.config.floatX))

        index = T.lscalar()  # index to a [mini]batch
        x = T.vector('x')
        y = T.scalar('y')

        # Get initial parameters
        self._params()

        # Set cost
        cost = self.cost(x, y) + (self.L1_reg * self._L1()) + (self.L2_reg * self._L2())

        gparams = [T.grad(cost, param) for param in self.params]

        if self.optimizer.lower() in ['sgd', 'stochastic_gradient_descent']:
            updates = [(param, param - self.learning_rate * gparam) for param, gparam in zip(self.params, gparams)]
        else:
            raise ValueError('Optimizer not valid. Please use "sdg"')
        
        # Initialize variables for training
        current_epoch = 0
        done_looping = False
        best_validation_loss = np.inf
        best_iter = 0
        test_score = 0.
        
        train_model = theano.function(
            inputs=[index],
            outputs=cost,
            updates=updates,
            givens={x: valid_set_x[index], y: valid_set_y[index]}
        )

        # theano.function(inputs=[x,y], outputs=[prediction, xent], updates=[[w, w-0.01*gw], [b, b-0.01*gb]], name = "train")
        
        validate_model = theano.function(
            inputs=[index],
            outputs=cost,
            givens={x: valid_set_x[index], y: valid_set_y[index]}
        )

        # early-stopping parameters
        patience = 10000  # look as this many examples regardless
        patience_increase = 2  # wait this much longer when a new best is found
        improvement_threshold = 0.995  # a relative improvement of this much is considered significant
        validation_frequency = 10
        
        n_valid_set = valid_set_x.get_value(borrow=True).shape[0]
        n_test_set = test_set_x.get_value(borrow=True).shape[0] 

        while (current_epoch < epochs+1) & (not done_looping):
            current_epoch += 1
            minibatch_avg_cost = [train_model(i) for i in range(n_valid_set)]
            minibatch_avg_cost = np.mean(minibatch_avg_cost)
            if current_epoch % validation_frequency == 0:
                validation_losses = [validate_model(i).tolist() for i in range(n_test_set)]
                validation_losses = np.mean(validation_losses)
                if validation_losses < best_validation_loss:
                    if (validation_losses < best_validation_loss * improvement_threshold):
                        patience = max(patience, current_epoch * patience_increase)
                    best_validation_loss = validation_losses
                    best_iter = current_epoch
                    print('Epoch %i, validation error %f' %(current_epoch, best_validation_loss))
            
            # Check early stopping
            if patience <= current_epoch:
                done_looping = True
                break

import numpy as np
import random
import theano
import theano.tensor as T
import matplotlib.pyplot as plt

# import pycof as pc

import sys
sys.path.append("/Users/flofelic/Documents/statinf/")

from statinf.ml.losses import binary_cross_entropy
from statinf.ml.activations import logit
from statinf.data.GenerateData import generate_dataset



class GLM:
    def __init__(self, formula, data, fit_intercept=False, test_set=None, training_size=0.2, seed=None):
        super(GLM, self).__init__()
        # Parse formula
        self.no_space_formula = formula.replace(' ', '')
        self.Y_col = self.no_space_formula.split('~')[0]
        self.X_col = self.no_space_formula.split('~')[1].split('+')
        # Data information
        self.data = data
        self.N = data.shape[0]
        self.p = len(self.X_col)
        self.test_set = test_set
        # Set randomness if asked
        if seed is not None:
            self.rdm = np.random.RandomState(seed)
            random.seed(seed)
        else:
            self.rdm = np.random

        # Set test data
        if test_set is None:
            # Getting the data indexes
            data_idx = data.index.tolist()
            # Get random subsample from index for training and test sets
            train_index = random.sample(data_idx, int(self.N * training_size))
            test_index = [i for i in data_idx if i not in train_index]
            # Training set
            self.Y_train = [float(y) for y in self.data.loc[train_index, self.Y_col].values]
            self.X_train = self.data.loc[train_index, self.X_col]
            # Test set with randomly selected indexes
            self.Y_test = [float(y) for y in self.data.loc[test_index, self.Y_col].values]
            self.X_test = self.data.loc[test_index, self.X_col]
        else:
            # Training set from initial data provided
            self.Y_train = [float(y) for y in self.test_set[self.Y_col].values]
            self.X_train = self.data[self.X_col]
            # Test set from provided data
            self.Y_test = [float(y) for y in self.test_set[self.Y_col].values]
            self.X_test = self.test_set[self.X_col]
        
        # Theano data format
        ## Train set
        self.Y_train_theano = np.array(self.Y_train).astype(theano.config.floatX)
        self.X_train_theano = self.X_train.values.astype(theano.config.floatX)
        ## Test set
        self.Y_test_theano = np.array(self.Y_test).astype(theano.config.floatX)
        self.X_test_theano = self.X_test.values.astype(theano.config.floatX)

    
    def fit(self, epochs=1000, patience=0.1, improvement_threshold=0.995, learning_rate=0.01, restore_weights=True, early_stop=True, plot=False, verbose=False, verbose_all=False):
        
        verbose = True if verbose_all else verbose
        # Parse arguments
        self.learning_rate = learning_rate
        if type(patience) == float:
            # User can provide either test data size or ratio of total data
            patience = epochs * patience
        
        # Initialize values
        train_losses = []       # Training loss vector to plot
        test_losses = []        # Test loss vector to plot
        self.best_loss = np.inf # Initial best loss value
        verif_i = epochs        # First, verify epoch at then end (init only)
        i = 0

        # Declare Theano symbolic variables
        self.x = T.matrix("x")
        self.y = T.vector("y")
        self.w = theano.shared(self.rdm.randn(self.p).astype(theano.config.floatX), name="w")
        self.b = theano.shared(np.asarray(0., dtype=theano.config.floatX), name="b")
        # Test tag values from Theano objects
        self.x.tag.test_value = self.X_train_theano
        self.y.tag.test_value = self.Y_train_theano
        
        # Construct Theano expression graph
        self.p_1 = logit(x=self.x, weights=self.w, bias=self.b, tensor=True)    # Probability of having a one
        self.prediction = self.p_1 > 0.5                              # The prediction that is done: 0 or 1

        # Compute gradients
        self.xent = binary_cross_entropy(self.y, self.p_1, tensor=True)
        self.cost = self.xent.mean() #+ 0.01*(w**2).sum() # The cost to optimize
        self.gw, self.gb = T.grad(self.cost, [self.w, self.b])

        # Training and prediction functions
        self._train = theano.function(inputs=[self.x, self.y], outputs=[self.prediction, self.xent], updates=[[self.w, self.w-self.learning_rate*self.gw], [self.b, self.b-self.learning_rate*self.gb]], name = "train")
        self._predict = theano.function(inputs=[self.x], outputs=self.prediction, name="predict")

        # Start training
        while i <= epochs:
            i += 1
            # Train
            output = self._train(self.X_train_theano, self.Y_train_theano)
            # Evaluate on test set
            test_out = self._predict(self.X_test_theano)
            # Measure accuracy on train and test sets
            epoch_loss = binary_cross_entropy(self.Y_train_theano, output[0])
            epoch_loss_test = binary_cross_entropy(self.Y_test_theano, test_out)
            # Save to historical performance
            train_losses += [epoch_loss]
            test_losses += [epoch_loss_test]
            # If we got the best score until now, and patience is not reached
            if (epoch_loss_test < self.best_loss * improvement_threshold) & (i <= verif_i) & early_stop:
                # pc.verbose_display(f'New min found for iter {i}', verbose=verbose_all)
                verif_i = i + patience
                self.best_loss = epoch_loss_test
                self.best_iter = i
                self.optimal_w = self.w.get_value()
                self.optimal_b = self.b.get_value()
            # if no improvement
            elif (i <= verif_i):
                # pc.verbose_display(f'No improvement at iter {i}', verbose=verbose_all)
                pass
            else:
                # pc.verbose_display(f'Stop training. Minimum found after {self.best_iter} iterations', verbose)
                break

        if restore_weights:
            # Keep the parameters of the best run and not of the last one
            self.w = theano.shared(self.optimal_w.astype(theano.config.floatX), name="w")
            self.b = theano.shared(self.optimal_b.astype(theano.config.floatX), name="b")
        
        if plot:
            # Plot the training and test loss
            plt.title('Logit train vs. test error through epochs', loc='center')
            plt.plot(list(range(i)), train_losses, label='Training loss')
            plt.plot(list(range(i)), test_losses, label='Test loss')
            plt.legend()
            plt.show()

    def predict(self, new_data):
        data_to_pred = new_data[self.X_col].values.astype(theano.config.floatX)
        new_out = self._predict(data_to_pred)
        return [float(y) for y in new_out]

    def get_weights(self):
        self.parameters = {'weights': self.w.get_value(), 'bias': self.b.get_value()}
        return self.parameters




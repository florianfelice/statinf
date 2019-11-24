import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm


class MLP():
    """
        Multi Layer Perceptron network.
        Needs architecture + initializer
        - architecture: layers' dimension for in and out + activation function
        - intializers: bias + weights matrices initializers' method
    """

    def __init__(self, data, y, architecture, initializers):
        self.X_col = [c for c in data.columns if c != y]

        self.X = data[self.X_col].to_numpy().T
        self.y = data[y].to_numpy().reshape((data[y].to_numpy().shape[0], -1)).T

        self.architecture = architecture
        self.initializers = initializers
        self.params = self._initialize_params()

    def _initialize_params(self):
        """
            Initialize weight and bias matrices:
            - weights: random or uniform init
            - bias: init with zeros or random
        """

        params = {}
        bias_init = self.initializers['bias']
        weights_init = self.initializers['weights']

        for id_, layer in enumerate(self.architecture):
            layer_id = id_ + 1
            
            if layer_id == 1:
                input_dim = self.X.shape[0]
            else:
                input_dim = self.architecture[layer_id - 2]['layer_size']
            
            output_dim = layer['layer_size']

            # Set allowed argument for bias/weight initializers
            w_allow = ['random', 'uniform']
            b_allow = ['random', 'zeros']
            
            # Initialize weights matrix
            if weights_init == 'random':
                params['W'+str(layer_id)] = np.random.randn(output_dim, input_dim)
            elif weights_init == 'uniform':
                params['W'+str(layer_id)] = np.random.uniform(-1, 1, (output_dim, input_dim))
            else:
                raise Exception(f"{bias_init} is not supported for weights initializer, only {', '.join(w_allow)} are accepted")

            # Initialize bias parameters
            if bias_init == 'random':
                params['b'+str(layer_id)] = np.random.randn(output_dim, 1)
            elif bias_init == 'zeros':
                params['b'+str(layer_id)] = np.zeros((output_dim, 1))
            else:
                raise Exception(f"{bias_init} is not supported for bias initializer, only {', '.join(b_allow)} and  can be chosen")

        return params

    # Define activation functions and their derivatives
    # Sigmoid function
    def sigmoid(self, Z):
        return 1/(1+np.exp(-Z))

    # Derivative of sigmoid for BPTT
    def sigmoid_backward(self, dA, z_curr):
        sig = self.sigmoid(z_curr)
        return sig*(1-sig)*dA

    # ReLU: Retified Linear Units
    def relu(self, Z):
        return np.maximum(0, Z)

    # Derivative of ReLU for BPTT
    def relu_backward(self, dA, z_curr):
        dz = np.array(dA, copy=True)
        dz[z_curr <= 0] = 0
        return dz
    
    # Linear function
    def linear(self, Z):
        return(Z)
    
    # Derivative of Linear for BPTT
    def linear_backward(self, Z):
        return(1)
    
    # Filter on the selected activation function
    def get_activation(self, activation, ls=False):
        all_activations = {'relu': self.relu,
                        'sigmoid': self.sigmoid,
                        'linear': self.linear}
        if ls:
            return [k for k in all_activations.keys()]
        else:
            return all_activations[activation]
    
    # Filter on the derivative of the activation function selected
    def get_derivative(self, activation, ls=False):
        all_derivatives = {'relu': self.relu_backward,
                        'sigmoid': self.sigmoid_backward,
                        'linear': self.linear_backward}
        if ls:
            return [k for k in all_derivatives.keys()]
        else:
            return all_derivatives[activation]


    # Define feedforward propagation
    def _forward_prop_this_layer(self, A_prev, W_curr, b_curr, activation_function):
        """
            In feedforward propagation, we compute:
            g(z)
            where,
            z = W.X + b
                with W being the weight matrix, X the input in the layer and b the bias
            and g the activation function
        """

        # Compute current value of z = W.X + b
        z_curr = np.dot(W_curr, A_prev) + b_curr

        # Set allowed activation functions
        act_allow = ['relu', 'sigmoid', 'linear']

        # We define the shape of g(.)
        try:
            activation = self.get_activation(activation_function)
        except:
            ls_activations = ', '.join(self.get_activation("", ls=True))
            raise Exception(f"{activation_function} activations is not supported, only {ls_activations} are supported")

        # We return the transformed inputs and the actual inputs
        return activation(z_curr), z_curr

    # Forward pass
    def _forward(self, X):
        cache = {}
        A_current = X
        for layer_id_prev, layer in enumerate(self.architecture):
            current_layer_id = layer_id_prev+1

            A_previous = A_current
            activation = layer['activation']

            W_curr = self.params['W'+str(current_layer_id)]
            b_curr = self.params['b'+str(current_layer_id)]

            A_current, Z_curr = self._forward_prop_this_layer(A_previous, W_curr,
                                                        b_curr, activation)

            cache['A'+str(layer_id_prev)] = A_previous
            cache['Z'+str(current_layer_id)] = Z_curr
        return A_current, cache

    def log_stability(self, x):
        x = x[0]
        x[x == 1.0] = 1 - 10e-15
        x[x == 0.0] = 10e-15
        return(x)

    def _criterion(self, yhat):
        m = yhat.shape[1]
        if (self.y.max() == 1.0) & (self.y.min() == 0.0):
            y_transform = self.log_stability(yhat)
        else:
            y_transform = yhat

        cost = -1/m * (np.dot(self.y, np.log(y_transform).T) + np.dot(1-self.y, np.log(1-y_transform).T))
        return np.squeeze(cost)
  
    def _backprop_this_layer(self, da_curr, z_curr, W_curr, b_curr, A_prev, activation_function):
        try:
            activation_back = self.get_derivative(activation_function)
        except:
            ls_derivatives = ', '.join(self.get_derivative("", ls=True))
            raise Exception(f"{activation_function} activations is not supported, only {ls_derivatives} are supported")
        
        m = A_prev.shape[1]

        dz_curr = activation_back(da_curr, z_curr)
        dw_curr = np.dot(dz_curr, A_prev.T)/m
        db_curr = np.sum(dz_curr, axis=1, keepdims=True)/m
        da_prev = np.dot(W_curr.T, dz_curr)

        return da_prev, dw_curr, db_curr
  
    def _backward(self, ytrue, ypred, cache):
        grads = {}
        m = ytrue.shape[1]
        da_prev = np.divide(1-ytrue, 1-ypred) - np.divide(ytrue, ypred)
    
        for prev_layer_id, layer in reversed(list(enumerate(self.architecture))):
            layer_id = prev_layer_id + 1
            activation = layer['activation']

            da_curr = da_prev

            A_prev = cache['A'+str(prev_layer_id)]
            Z_curr = cache['Z'+str(layer_id)]

            W_curr = self.params['W'+str(layer_id)]
            b_curr = self.params['b'+str(layer_id)]

            self.weights = W_curr
            self.bias = b_curr

            da_prev, dw_curr, db_curr = self._backprop_this_layer(
                da_curr, Z_curr, W_curr, b_curr, A_prev, activation)

            grads["dw"+str(layer_id)] = dw_curr
            grads['db'+str(layer_id)] = db_curr
        return grads
  
    def update(self, grads, learning_rate):
        for layer_id, layer in enumerate(self.architecture, 1):
            self.params['W'+str(layer_id)] -= learning_rate * grads['dw'+str(layer_id)]
            self.params['b'+str(layer_id)] -= learning_rate * grads['db'+str(layer_id)]
  
    def fit(self, epochs, learning_rate=0.01, verbose=True, show_loss=True, plot=False):
        self.loss_history, self.accuracy_history = [], []
        
        for epoch in tqdm(range(epochs), total=epochs, unit='epoch'):
            yhat, cache = self._forward(self.X)
            loss = self._criterion(yhat)
            self.loss_history.append(loss)

            yacc = yhat.copy()
            yacc[yacc>0.5] = 1
            yacc[yacc<=0.5] = 0

            accuracy = np.sum(self.y[0]==yacc[0])/(yacc.shape[1])
            self.accuracy_history.append(accuracy)

            grads_values = self._backward(self.y, yhat, cache)

            self.update(grads_values, learning_rate)
            if(epoch % 500 == 0):
                if(verbose):
                    print("Epoch: {:05} - cost: {:.5f} - accuracy: {:.5f}".format(epoch, loss, accuracy))
        self.yhat = yhat

        if plot:
            fig = plt.figure(figsize=(12,10))
            plt.plot(range(epochs), self.loss_history, 'r-')
            plt.plot(range(epochs), self.accuracy_history, 'b--')
            plt.legend(['Training_loss', 'Training_Accuracy'])
            plt.xlabel('Epochs')
            plt.ylabel('Loss/Accuracy')
            plt.show()
    
    def predict(self, new_data):
        X = new_data.to_numpy().T
        yhat, _ = self._forward(X)
        activ = self.get_activation(self.architecture[len(self.architecture)-1]['activation'])
        out = activ(yhat)
        return np.squeeze(out)





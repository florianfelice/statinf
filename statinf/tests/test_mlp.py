
import sys, os
import pycof
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath('/Users/flofelic/Documents/statinf/'))

from statinf.regressions.LinearModels import OLS

from statinf.data.GenerateData import generate_dataset
from statinf.ml.neuralnetwork import NeuralNetwork, Layer
from statinf.ml.losses import mean_squared_error


data = generate_dataset(coeffs=[0.342, 0.3423, 0.465546, 0.54642], n=100, std_dev=[.0754, 0.042, .035, .0342])

data_X = data[['X0', 'X1', 'X2', 'X3']]


X = data_X.values

y_data = data['Y'].values

nn = NeuralNetwork(loss='MSE') # loss='binary_cross_entropy'
nn.add_layer(Layer(4, 1, 'linear'))
# nn.add_layer(Layer(5, 4, 'linear'))
# nn.add_layer(Layer(4, 1, 'linear'))


# Train the neural network
nn.train(X, y_data, max_epochs=1, plot=True, verbose_every=50)
pred = nn.predict(X)
# print(pred)
# print(y)
print('Accuracy: %.2f%%' % (nn.compute_loss(y_pred=pred.flatten(), y_true=y_data.flatten(), verbose=True) * 100))

print(nn._layers[0].weights)


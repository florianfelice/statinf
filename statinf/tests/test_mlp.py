
import sys, os
import pycof
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath('/Users/flofelic/Documents/statinf/'))

from statinf.regressions.LinearModels import OLS

from statinf.data.GenerateData import generate_dataset
from statinf.ml.neuralnetwork import NeuralNetwork, Layer
from statinf.ml.losses import mean_squared_error


data = generate_dataset(coeffs=[1.342, 6.3423, .465546, 6.54642], n=100, std_dev=[.454,1.342, 4.0035, 3.342])

data_X = data[['X0', 'X1', 'X2', 'X3']]

print()


X = data_X.values

y = data['Y'].values

nn = NeuralNetwork(loss='MSE') # loss='binary_cross_entropy'
nn.add_layer(Layer(4, 4, 'tanh'))
# nn.add_layer(Layer(4, 2, 'sigmoid'))
nn.add_layer(Layer(4, 1, 'linear'))


# X = np.array([[.76545, 45], [3234.3423, 13.3534], [.3543, 34.453432], [2123.4534235, 20.46534]])

# y = np.array([[3.3423], [45.35324], [24.34567], [32.2354664]])

# Define dataset
# X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

# y = np.array([[0], [0], [0], [1]])

# y_pred = np.array([[0.043598], [0.09845], [0.0940385], [0.94576]])

# binary_cross_entropy(y, y_pred)[0][0]


# Train the neural network
nn.train(X, y, learning_rate=0.3, max_epochs=2, plot=False, verbose_every=1)
pred = nn.predict(X, binary_threshold=0.5)
print(pred)
print(y)
print('Accuracy: %.2f%%' % (100 - nn.cost(y_pred=pred.flatten(), y_true=y.flatten(), loss='MSE', verbose=True) * 100))

# # Plot changes in mse
# plt.plot(errors)
# plt.title('Changes in MSE')
# plt.xlabel('Epoch (every 10th)')
# plt.ylabel('MSE')
# plt.show()
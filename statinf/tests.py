import numpy as np

from statinf.regressions.LinearModels import OLS

from statinf.data.GenerateData import generate_dataset
from statinf.ml.neuralnetwork import MLP, Layer
from statinf.ml.losses import mean_squared_error


print('=== GENERATE SYNTHETIC DATA ===\n')

data = generate_dataset(coeffs=[1.2556, -6.465, 1.665414, 1.5444], n=1000, std_dev=1.6, intercept=.0)

X = ['X0', 'X1', 'X2', 'X3']


##########################################

print('\n=== TEST OLS ===\n')

formula = "Y ~ X0 + X1 + X2 + X3"
ols = OLS(formula, data, fit_intercept=False)

print(ols.summary())



##########################################

print('\n=== TEST MLP ===\n')

nn = MLP()
nn.add(Layer(4, 1))

# Train the neural network
nn.train(data=data, X=X, Y='Y', epochs=1, learning_rate=0.001)

print(nn.get_weights())

import numpy as np
import pandas as pd
import pycof as pc

import os, sys
import getpass

sys.path.append(f"/Users/{getpass.getuser()}/Documents/statinf/")

from statinf.regressions.LinearModels import OLS
from statinf.data.GenerateData import generate_dataset
from statinf.ml.neuralnetwork import MLP, Layer
from statinf.ml.losses import mean_squared_error


# General parameters
n = 2500


print('##########################################')
print('############## LINEAR DATA ###############')
print('##########################################')

print('=== GENERATE SYNTHETIC DATA ===\n')

coeffs_lin = [1.2556, -0.465, 1.665414, 2.5444, -7.56445]
# Find the list of variables to train on
X_lin = [f'X{i}' for i in range(len(coeffs_lin))]
data = generate_dataset(coeffs=coeffs_lin, n=n, std_dev=2.6)

train = data.iloc[1:1000]
test = data.iloc[1001:2000]
appli = data.iloc[2001:n]


##########################################


print('\n=== TEST OLS ===\n')

formula = "Y ~ " + ' + '.join(X_lin)
ols = OLS(formula, train, fit_intercept=False)

pred_ols = ols.predict(appli)

print(f"RMSE for OLS is: {mean_squared_error(y_true=appli['Y'], y_pred=pred_ols, root=True)}")

print('Model summary')
print(ols.summary())



##########################################

print('\n=== TEST MLP ===\n')

nn = MLP(loss='mse', optimizer='sgd')
nn.add(Layer(len(X_lin), 1))

# Train the neural network
nn.train(data=train, X=X_lin, Y='Y', epochs=1, learning_rate=0.001)
pred = nn.predict(appli)

print(f"RMSE is: {mean_squared_error(y_true=appli['Y'], y_pred=pred, root=True)}")


print('Model parameters')
print(nn.get_weights())

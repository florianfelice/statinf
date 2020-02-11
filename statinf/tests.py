import numpy as np


import os, sys

sys.path.append("/Users/flofelic/Documents/statinf/")

from statinf.regressions.LinearModels import OLS
from statinf.regressions.glm import GLM
from statinf.data.GenerateData import generate_dataset
from statinf.ml.neuralnetwork import MLP, Layer
from statinf.ml.losses import mean_squared_error, binary_accuracy


# General parameters
n      = 2500
coeffs = [1.2556, -6.465, 1.665414, 1.5444]


# Find the list of variables to train on
X       = [f'X{i}' for i in range(len(coeffs))]


# print('##########################################')
# print('############## LINEAR DATA ###############')
# print('##########################################')

# print('=== GENERATE SYNTHETIC DATA ===\n')

# data = generate_dataset(coeffs=coeffs, n=n, std_dev=1.6)

# train = data.iloc[1:1000]
# test = data.iloc[1001:2000]
# appli = data.iloc[2001:n]


# ##########################################


# print('\n=== TEST OLS ===\n')

# formula = "Y ~ X0 + X1 + X2 + X3"
# ols = OLS(formula, train, fit_intercept=False)

# pred_ols = ols.predict(appli)

# print(f"RMSE for OLS is: {mean_squared_error(y_true=appli['Y'], y_pred=pred_ols, root=True)}")

# print('Model summary')
# print(ols.summary())



# ##########################################

# print('\n=== TEST MLP ===\n')

# nn = MLP()
# nn.add(Layer(4, 1))

# # Train the neural network
# nn.train(data=train, X=X, Y='Y', epochs=1, learning_rate=0.001)
# pred = nn.predict(appli)

# print(f"RMSE is: {mean_squared_error(y_true=appli['Y'], y_pred=pred, root=True)}")


# print('Model parameters')
# print(nn.get_weights())



print('##########################################')
print('############## BINARY DATA ###############')
print('##########################################')


print('=== GENERATE SYNTHETIC DATA ===\n')

bin_data = generate_dataset(coeffs=[1.2556, -6.465, 1.665414, -1.5444], n=n, std_dev=10.5, binary=True, seed=564)

train_bin = bin_data.iloc[1:1000]
test_bin = bin_data.iloc[1001:2000]
appli_bin = bin_data.iloc[2001:n]


data_balance = train_bin['Y'].sum() / train_bin['Y'].shape[0]
print(f'The models are expected to find {round(data_balance, 3)}% of 1s\n')


##########################################

print('\n=== TEST LOGIT ===\n')

formula = "Y ~ X0 + X1 + X2 + X3"
logit = GLM(formula, train_bin, test_set=test_bin)

# Fit the model
logit.fit(plot=True, epochs=1000, verbose=True)

pred_logit = logit.predict(new_data=appli_bin)

logit_acc = binary_accuracy(y_true=appli_bin['Y'], y_pred=pred_logit) * 100
data_balance_logit = np.sum(pred_logit) / len(pred_logit)

print(f"Model accuracy is {round(logit_acc, 3)}%")
print(f'Model found {round(data_balance_logit, 3)}% of 1s')

print('Model parameters')
print(logit.get_weights())


##########################################

print('\n=== TEST MLP ===\n')

nn = MLP()
nn.add(Layer(4, 1, activation='sigmoid'))

# Train the neural network
nn.train(data=train_bin, X=X, Y='Y', test_set=test_bin, epochs=100, learning_rate=0.001, plot=True)
pred_mlp_bin = nn.predict(appli_bin, binary=True)

bin_mlp_acc = binary_accuracy(y_true=appli_bin['Y'], y_pred=pred_mlp_bin) * 100
data_balance_bin_mlp = np.sum(pred_mlp_bin) / len(pred_mlp_bin)

print(f"Model accuracy is {round(bin_mlp_acc, 3)}%")
print(f'Model found {round(data_balance_bin_mlp, 3)}% of 1s')

print('Model parameters')
print(nn.get_weights())

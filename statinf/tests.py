import numpy as np
import pandas as pd

import os, sys
import getpass

sys.path.append(f"/Users/{getpass.getuser()}/Documents/statinf/")

from statinf.regressions.LinearModels import OLS
from statinf.regressions.glm import glm as GLM
from statinf.data.GenerateData import generate_dataset
from statinf.ml.neuralnetwork import MLP, Layer
from statinf.ml.losses import mean_squared_error, binary_accuracy

# to me removed
import statsmodels.api as sm

# General parameters
n = 2500


# print('##########################################')
# print('############## LINEAR DATA ###############')
# print('##########################################')

# print('=== GENERATE SYNTHETIC DATA ===\n')

coeffs_lin = [1.2556, -0.465, 1.665414, 2.5444, -7.56445]
# Find the list of variables to train on
X_lin = [f'X{i}' for i in range(len(coeffs_lin))]
# data = generate_dataset(coeffs=coeffs_lin, n=n, std_dev=2.6)

# train = data.iloc[1:1000]
# test = data.iloc[1001:2000]
# appli = data.iloc[2001:n]


# ##########################################


# print('\n=== TEST OLS ===\n')

# formula = "Y ~ " + ' + '.join(X_lin)
# ols = OLS(formula, train, fit_intercept=False)

# pred_ols = ols.predict(appli)

# print(f"RMSE for OLS is: {mean_squared_error(y_true=appli['Y'], y_pred=pred_ols, root=True)}")

# print('Model summary')
# print(ols.summary())



# ##########################################

# print('\n=== TEST MLP ===\n')

# nn = MLP(loss='mse')
# nn.add(Layer(len(X_lin), 1))

# # Train the neural network
# nn.train(data=train, X=X_lin, Y='Y', epochs=1, learning_rate=0.001)
# pred = nn.predict(appli)

# print(f"RMSE is: {mean_squared_error(y_true=appli['Y'], y_pred=pred, root=True)}")


# print('Model parameters')
# print(nn.get_weights())



print('##########################################')
print('############## BINARY DATA ###############')
print('##########################################')


print('=== GENERATE SYNTHETIC DATA ===\n')

coeffs_bin = [-1.2556, -1.465, -2.665414, -8.5444, 1.56445, 1.564684651, 4.314433]
# Find the list of variables to train on
X_bin = [f'X{i}' for i in range(len(coeffs_bin))]

bin_data = generate_dataset(coeffs=coeffs_bin, n=n, std_dev=1.6, loc=0.25, scale=1.5)
bin_data['X4'] = np.random.choice([0, 1], size=(n,), p=[2./3, 1./3])
e = np.random.normal(loc=0.5, scale=2, size=n)
bin_data['Y'] = [1. if y > 0. else 0. for y in bin_data[X_bin].dot(coeffs_bin) + e]



train_bin = bin_data.iloc[1:1000]
test_bin = bin_data.iloc[1001:2000]
appli_bin = bin_data.iloc[2001:n]


data_balance = train_bin['Y'].sum() / train_bin['Y'].shape[0]
print(f'The models are expected to find {round(data_balance*100, 3)}% of 1s\n')


##########################################

# print('\n=== TEST LOGIT ===\n')

# formula = "Y ~ " + ' + '.join(X_bin)
# logit = GLM(formula, train_bin) #test_set=test_bin)
# logit.fit()
# print(logit._log_likelihood())

# # Fit the model
# logit.fit(plot=True, epochs=1000, verbose=True)

# pred_logit = logit.predict(new_data=appli_bin)

# logit_acc = binary_accuracy(y_true=appli_bin['Y'], y_pred=pred_logit) * 100
# data_balance_logit = np.sum(pred_logit) / len(pred_logit)

# print(f"Model accuracy is {round(logit_acc, 3)}%")
# print(f'Model found {round(data_balance_logit, 3)}% of 1s')

# print('Model parameters')
# print(logit.get_weights())

print('\n=== TEST STATMODELS LOGIT ===\n')
logit = sm.Logit(train_bin['Y'], train_bin[X_bin]).fit()
# res = logit.fit(maxiter=100)

logit.summary()


pred_logit = [1. if x > 0.5 else 0. for x in logit.predict(appli_bin[X_bin])]


logit_acc = binary_accuracy(y_true=appli_bin['Y'], y_pred=pred_logit) * 100
data_balance_logit = np.sum(pred_logit) / len(pred_logit)

print(f"Model accuracy is {round(logit_acc, 3)}%")
print(f'Model found {round(data_balance_logit*100, 3)}% of 1s')
print('And true coefficients are:')
print(coeffs_bin)

# print('Model parameters')
print(logit.params.tolist())

odds_l = np.exp(logit.params[0])

# Prepare data for odd ratios
for_odd_0 = appli_bin.loc[2001, X_bin].copy()
for_odd_0['X4'] = 0.

for_odd_1 = appli_bin.loc[2001, X_bin].copy()
for_odd_1['X4'] = 1.

print('SM OR')
print(logit.predict(for_odd_1.to_numpy()) / logit.predict(for_odd_0.to_numpy()))


##########################################

print('\n=== TEST PERCEPTRON ===\n')

pt = MLP(loss='bce')
pt.add(Layer(len(X_bin), 1, activation='sigmoid'))

# Train the neural network
pt.train(data=train_bin, X=X_bin, Y='Y', test_set=test_bin, epochs=100, learning_rate=0.01, plot=False)
pred_pt_bin = pt.predict(appli_bin, binary=True)

bin_pt_acc = binary_accuracy(y_true=appli_bin['Y'], y_pred=pred_pt_bin) * 100
data_balance_bin_pt = np.sum(pred_pt_bin) / len(pred_pt_bin)

print(f"Model accuracy is {round(bin_pt_acc, 3)}%")
print(f'Model found {round(data_balance_bin_pt*100, 3)}% of 1s')

print('Model parameters')
pt_coeffs = [x[0] for x in pt.get_weights(param='weights')['weights 0']]
print(pt_coeffs)

print('Perceptron OR')
print(pt.predict(pd.DataFrame(for_odd_1).transpose())[0] / pt.predict(pd.DataFrame(for_odd_0).transpose())[0])



##########################################

print('\n=== TEST MLP ===\n')

nn = MLP(loss='bce')
nn.add(Layer(len(X_bin), 4, activation='linear'))
nn.add(Layer(4, 2, activation='linear'))
nn.add(Layer(2, 1, activation='sigmoid', init_weights='zeros'))

# Train the neural network
nn.train(data=train_bin, X=X_bin, Y='Y', test_set=test_bin, epochs=100, learning_rate=0.01, plot=False)
pred_mlp_bin = nn.predict(appli_bin, binary=True)

bin_mlp_acc = binary_accuracy(y_true=appli_bin['Y'], y_pred=pred_mlp_bin) * 100
data_balance_bin_mlp = np.sum(pred_mlp_bin) / len(pred_mlp_bin)

print(f"Model accuracy is {round(bin_mlp_acc, 3)}%")
print(f'Model found {round(data_balance_bin_mlp*100, 3)}% of 1s')

print('Model parameters')
mlp_coeffs = [x[0] for x in nn.get_weights(param='weights')['weights 0']]
print(mlp_coeffs)

print('MLP OR')
print(nn.predict(pd.DataFrame(for_odd_1).transpose())[0] / nn.predict(pd.DataFrame(for_odd_0).transpose())[0])

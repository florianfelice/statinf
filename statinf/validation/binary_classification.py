import numpy as np
import pandas as pd

import os
import sys
import getpass

sys.path.append(f"/Users/{getpass.getuser()}/Documents/statinf/")
sys.path.append(f"/Users/{getpass.getuser()}/Documents/PYCOF/")

import pycof as pc

from statinf.regressions.LinearModels import OLS
from statinf.regressions.glm import GLM
from statinf.data.GenerateData import generate_dataset
from statinf.ml.neuralnetwork import MLP, Layer
from statinf.ml.losses import mean_squared_error
from statinf.ml.performance import BinaryPerformance

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--iterations", help="Number of iterations", type=int, default=100)
parser.add_argument("-n", "--n_obervations", help="Number of observations", type=int, default=2500)
parser.add_argument("-c", "--n_cols", help="Number of observations", type=int, default=20)
parser.add_argument("-v", "--verbose", help="Display details", action='store_true')
args = parser.parse_args()


# to me removed
import statsmodels.api as sm


if sys.platform in ['darwin']:
    output_path = f'/Users/{getpass.getuser()}/Downloads/binary_models_OR_comparison.csv'
elif sys.platform in ['linux']:
    output_path = f'/home/{getpass.getuser()}/binary_models_OR_comparison.csv'
elif sys.platform in ['win32', 'win64']:
    output_path = f'C:/Users/{getpass.getuser()}/Downloads/binary_models_OR_comparison.csv'
else:
    raise OSError('Could not identify your OS')


data_balance = 0.

pc.verbose_display('=== GENERATE SYNTHETIC DATA ===\n', args.verbose)

while (data_balance < 0.02) or (data_balance > 0.98):
    coeffs_bin = np.random.uniform(low=-10, high=10, size=args.n_cols)
    # Find the list of variables to train on
    X_bin = [f'X{i}' for i in range(len(coeffs_bin))]

    bin_var = 'X4'

    loc_param = np.random.uniform(low=-5, high=5, size=1)

    bin_data = generate_dataset(coeffs=coeffs_bin, n=args.n_obervations, std_dev=1.6, loc=loc_param, scale=1.5)
    bin_data[bin_var] = np.random.choice([0, 1], size=(args.n_obervations,), p=[2./3, 1./3])

    quad_add = np.random.choice(range(1, len(coeffs_bin)), size=1)[0]

    hidden_vars = []
    hidden_coeffs = []

    if quad_add > 0:
        pc.verbose_display(f'We create {quad_add} tranformation variables', args.verbose)
        transfrom_var = [x for x in list(np.random.choice(X_bin, size=quad_add)) if x != bin_var]
        for x in transfrom_var:
            bin_data[x + x] = bin_data[x] ** 2
            hidden_vars += [x + x]

    hidden_coeffs = np.random.uniform(low=-5, high=5, size=len(hidden_vars))

    # Noise
    e = np.random.normal(loc=loc_param, scale=np.sqrt(loc_param), size=args.n_obervations)

    bin_data['Y'] = [1. if y > 0. else 0. for y in bin_data[X_bin + hidden_vars].dot(np.append(coeffs_bin, hidden_coeffs)) + e]


    # Compute real Odds Ratio
    p_1 = bin_data.Y[bin_data[bin_var] == 1.].mean()
    p_0 = bin_data.Y[bin_data[bin_var] == 0.].mean()

    odds_1 = p_1 / (1 - p_1)
    odds_0 = p_0 / (1 - p_0)
    real_OR = odds_1 / odds_0

    data_to_split = bin_data[X_bin + ['Y']].copy()

    # Split into train / test / appli sets
    train_bin = data_to_split.iloc[1:1000]
    test_bin = data_to_split.iloc[1001:2000]
    appli_bin = data_to_split.iloc[2001:args.n_obervations]

    data_balance = train_bin['Y'].sum() / train_bin['Y'].shape[0]


pc.verbose_display(f'The models are expected to find {round(data_balance*100, 3)}% of 1s\n', args.verbose)


# Prepare data for odd ratios for models
for_odd_0 = appli_bin.loc[2001, X_bin].copy()
for_odd_0['X4'] = 0.

for_odd_1 = appli_bin.loc[2001, X_bin].copy()
for_odd_1['X4'] = 1.

##########################################

print('\n=== TEST LOGIT ===\n')

formula = "Y ~ " + ' + '.join(X_bin)
logit = GLM(formula, train_bin, fit_intercept=False, family='binomial')
logit.fit(cov_type='nonrobust', plot=True)
# print(logit._log_likelihood())

print('Estimated coeffs')
print(logit.summary())

print('Real coeffs')
print(coeffs_bin)

# # Fit the model
# logit.fit(plot=True, epochs=1000, verbose=True)

pred_logit = logit.predict(new_data=appli_bin)
print(logit.partial_effects(['X1', 'X4'], average=True))
logit_acc = BinaryPerformance(y_true=appli_bin['Y'].values, y_pred=pred_logit).accuracy() * 100
# data_balance_logit = np.sum(pred_logit) / len(pred_logit)

print(f"Model accuracy is {round(logit_acc, 3)}%")
# print(f'Model found {round(data_balance_logit, 3)}% of 1s')

# print('Model parameters')
# print(logit.get_weights())

print('\n=== TEST STATMODELS LOGIT ===\n')
# logit_sm = sm.Logit(endog=train_bin['Y'], exog=train_bin[X_bin])
logit_sm = sm.Logit(endog=train_bin['Y'], exog=train_bin[X_bin])
logit_res = logit_sm.fit()

print(logit_res.summary())


# pred_logit = [1. if x > 0.5 else 0. for x in logit.predict(appli_bin[X_bin])]


# logit_acc = binary_accuracy(y_true=appli_bin['Y'], y_pred=pred_logit) * 100
# data_balance_logit = np.sum(pred_logit) / len(pred_logit)

# pc.verbose_display(f"Model accuracy is {round(logit_acc, 3)}%", args.verbose)
# pc.verbose_display(f'Model found {round(data_balance_logit*100, 3)}% of 1s', args.verbose)
# pc.verbose_display('And true coefficients are:', args.verbose)
# pc.verbose_display(coeffs_bin, args.verbose)

# # pc.verbose_display('Model parameters', args.verbose)
# pc.verbose_display(logit.params.tolist(), args.verbose)

# odds_l = np.exp(logit.params[0])


# ##########################################

# pc.verbose_display('\n=== TEST PERCEPTRON ===\n', args.verbose)

# pt = MLP(loss='bce')
# pt.add(Layer(len(X_bin), 1, activation='sigmoid'))

# # Train the neural network
# pt.train(data=train_bin, X=X_bin, Y='Y', test_set=test_bin, epochs=100, learning_rate=0.01, plot=False, verbose=args.verbose)
# pred_pt_bin = pt.predict(appli_bin, binary=True)

# bin_pt_acc = binary_accuracy(y_true=appli_bin['Y'], y_pred=pred_pt_bin) * 100
# data_balance_bin_pt = np.sum(pred_pt_bin) / len(pred_pt_bin)

# pc.verbose_display(f"Model accuracy is {round(bin_pt_acc, 3)}%", args.verbose)
# pc.verbose_display(f'Model found {round(data_balance_bin_pt*100, 3)}% of 1s', args.verbose)

# pc.verbose_display('Model parameters', args.verbose)
# pt_coeffs = [x[0] for x in pt.get_weights(param='weights')['weights 0']]
# pc.verbose_display(pt_coeffs, args.verbose)

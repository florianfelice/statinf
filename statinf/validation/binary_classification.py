import numpy as np
import pandas as pd
import pycof as pc

import os, sys
import getpass

sys.path.append(f"/Users/{getpass.getuser()}/Documents/statinf/")

from statinf.regressions.LinearModels import OLS
from statinf.regressions.glm import glm as GLM
from statinf.data.GenerateData import generate_dataset
from statinf.ml.neuralnetwork import MLP, Layer
from statinf.ml.losses import mean_squared_error, binary_accuracy


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


pc.verbose_display('################################################################################')
pc.verbose_display('############## Monte Carlo method for compare binary classifiers ###############')
pc.verbose_display('################################################################################')


pc.write('i,quad_vars,data_balance,SM_accuracy,Perceptron_accuracy,MLP_accuracy,MLP2_accuracy,MLP3_accuracy,MLP4_accuracy,DMLP_accuracy,DMLP2_accuracy,Real_OR,SM_OR,Perceptron_OR,MLP_OR,MLP2_OR,MLP3_OR,MLP4_OR,DMLP_OR,DMLP2_OR', output_path, perm='w')


for i in pc.verbose_display(range(args.iterations)):
# try:
    data_balance = 0.

    pc.verbose_display('=== GENERATE SYNTHETIC DATA ===\n', args.verbose)
    while (data_balance < 0.02) or (data_balance > 0.98):
        coeffs_bin = np.random.uniform(low=-10, high=10, size = args.n_cols)
        # Find the list of variables to train on
        X_bin = [f'X{i}' for i in range(len(coeffs_bin))]

        bin_var = 'X4'

        loc_param = np.random.uniform(low=-5, high=5, size=1)

        bin_data = generate_dataset(coeffs=coeffs_bin, n=args.n_obervations, std_dev=1.6, loc=loc_param, scale=1.5)
        bin_data[bin_var] = np.random.choice([0, 1], size=(args.n_obervations,), p=[2./3, 1./3])
        
        quad_add = np.random.choice(range(1, len(coeffs_bin)), size=1)[0]
        
        hidden_vars = []
        hidden_coeffs = []

        if quad_add>0:
            pc.verbose_display(f'We create {quad_add} tranformation variables', args.verbose)
            transfrom_var = [x for x in list(np.random.choice(X_bin, size=quad_add)) if x != bin_var]
            for x in transfrom_var:
                bin_data[x + x] = bin_data[x] **2
                hidden_vars += [x+x]

        hidden_coeffs = np.random.uniform(low=-5, high=5, size=len(hidden_vars))
        
        # Noise
        e = np.random.normal(loc=loc_param, scale=np.sqrt(loc_param), size=args.n_obervations)

        bin_data['Y'] = [1. if y > 0. else 0. for y in bin_data[X_bin + hidden_vars].dot(np.append(coeffs_bin, hidden_coeffs)) + e]


        ## Compute real Odds Ratio
        p_1 = bin_data.Y[bin_data[bin_var] == 1.].mean()
        p_0 = bin_data.Y[bin_data[bin_var] == 0.].mean()

        odds_1 = p_1/(1 - p_1)
        odds_0 = p_0/(1 - p_0)
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

    pc.verbose_display('\n=== TEST STATMODELS LOGIT ===\n', args.verbose)
    logit = sm.Logit(train_bin['Y'], train_bin[X_bin]).fit(disp=0)
    # res = logit.fit(maxiter=100)

    logit.summary()


    pred_logit = [1. if x > 0.5 else 0. for x in logit.predict(appli_bin[X_bin])]


    logit_acc = binary_accuracy(y_true=appli_bin['Y'], y_pred=pred_logit) * 100
    data_balance_logit = np.sum(pred_logit) / len(pred_logit)

    pc.verbose_display(f"Model accuracy is {round(logit_acc, 3)}%", args.verbose)
    pc.verbose_display(f'Model found {round(data_balance_logit*100, 3)}% of 1s', args.verbose)
    pc.verbose_display('And true coefficients are:', args.verbose)
    pc.verbose_display(coeffs_bin, args.verbose)

    # pc.verbose_display('Model parameters', args.verbose)
    pc.verbose_display(logit.params.tolist(), args.verbose)

    odds_l = np.exp(logit.params[0])


    ##########################################

    pc.verbose_display('\n=== TEST PERCEPTRON ===\n', args.verbose)

    pt = MLP(loss='bce')
    pt.add(Layer(len(X_bin), 1, activation='sigmoid'))

    # Train the neural network
    pt.train(data=train_bin, X=X_bin, Y='Y', test_set=test_bin, epochs=100, learning_rate=0.01, plot=False, verbose=args.verbose)
    pred_pt_bin = pt.predict(appli_bin, binary=True)

    bin_pt_acc = binary_accuracy(y_true=appli_bin['Y'], y_pred=pred_pt_bin) * 100
    data_balance_bin_pt = np.sum(pred_pt_bin) / len(pred_pt_bin)

    pc.verbose_display(f"Model accuracy is {round(bin_pt_acc, 3)}%", args.verbose)
    pc.verbose_display(f'Model found {round(data_balance_bin_pt*100, 3)}% of 1s', args.verbose)

    pc.verbose_display('Model parameters', args.verbose)
    pt_coeffs = [x[0] for x in pt.get_weights(param='weights')['weights 0']]
    pc.verbose_display(pt_coeffs, args.verbose)



    ##########################################

    pc.verbose_display('\n=== TEST MLP ===\n', args.verbose)

    nn = MLP(loss='bce', optimizer='adamax')
    # nn.add(Layer(len(X_bin), 10, activation='linear'))
    # nn.add(Layer(10, 10, activation='relu'))
    # nn.add(Layer(10, 5, activation='relu'))
    nn.add(Layer(len(X_bin), 1, activation='sigmoid'))

    # Train the neural network
    nn.train(data=train_bin, X=X_bin, Y='Y', test_set=test_bin, epochs=100, learning_rate=0.01, plot=False, verbose=args.verbose)
    pred_mlp_bin = nn.predict(appli_bin, binary=True)

    bin_mlp_acc = binary_accuracy(y_true=appli_bin['Y'], y_pred=pred_mlp_bin) * 100
    data_balance_bin_mlp = np.sum(pred_mlp_bin) / len(pred_mlp_bin)

    pc.verbose_display(f"Model accuracy is {round(bin_mlp_acc, 3)}%", args.verbose)
    pc.verbose_display(f'Model found {round(data_balance_bin_mlp*100, 3)}% of 1s', args.verbose)

    pc.verbose_display('Model parameters', args.verbose)
    mlp_coeffs = [x[0] for x in nn.get_weights(param='weights')['weights 0']]
    pc.verbose_display(mlp_coeffs, args.verbose)


    ##########################################

    pc.verbose_display('\n=== TEST MLP 2 ===\n', args.verbose)

    nn2 = MLP(loss='bce', optimizer='rmsprop')
    # nn2.add(Layer(len(X_bin), 10, activation='linear'))
    nn2.add(Layer(len(X_bin), 1, activation='sigmoid'))

    # Train the neural network
    nn2.train(data=train_bin, X=X_bin, Y='Y', test_set=test_bin, epochs=100, learning_rate=0.01, plot=False, verbose=args.verbose)
    pred_mlp2_bin = nn2.predict(appli_bin, binary=True)

    bin_mlp2_acc = binary_accuracy(y_true=appli_bin['Y'], y_pred=pred_mlp2_bin) * 100
    data_balance_bin_mlp2 = np.sum(pred_mlp2_bin) / len(pred_mlp2_bin)

    pc.verbose_display(f"Model accuracy is {round(bin_mlp2_acc, 3)}%", args.verbose)
    pc.verbose_display(f'Model found {round(data_balance_bin_mlp2*100, 3)}% of 1s', args.verbose)

    pc.verbose_display('Model parameters', args.verbose)
    mlp2_coeffs = [x[0] for x in nn2.get_weights(param='weights')['weights 0']]
    pc.verbose_display(mlp2_coeffs, args.verbose)

    ##########################################

    pc.verbose_display('\n=== TEST MLP 3 ===\n', args.verbose)

    nn3 = MLP(loss='bce', optimizer='adam')
    nn3.add(Layer(len(X_bin), len(X_bin), activation='linear'))
    nn3.add(Layer(len(X_bin), 1, activation='sigmoid'))

    # Train the neural network
    nn3.train(data=train_bin, X=X_bin, Y='Y', test_set=test_bin, epochs=100, learning_rate=0.01, plot=False, verbose=args.verbose)
    pred_mlp3_bin = nn3.predict(appli_bin, binary=True)

    bin_mlp3_acc = binary_accuracy(y_true=appli_bin['Y'], y_pred=pred_mlp3_bin) * 100
    data_balance_bin_mlp3 = np.sum(pred_mlp3_bin) / len(pred_mlp3_bin)

    pc.verbose_display(f"Model accuracy is {round(bin_mlp3_acc, 3)}%", args.verbose)
    pc.verbose_display(f'Model found {round(data_balance_bin_mlp3*100, 3)}% of 1s', args.verbose)

    pc.verbose_display('Model parameters', args.verbose)
    mlp3_coeffs = [x[0] for x in nn3.get_weights(param='weights')['weights 0']]
    pc.verbose_display(mlp3_coeffs, args.verbose)

    ##########################################

    pc.verbose_display('\n=== TEST MLP 4 ===\n', args.verbose)

    nn4 = MLP(loss='bce', optimizer='rmsprop')
    nn4.add(Layer(len(X_bin), len(X_bin), activation='linear'))
    nn4.add(Layer(len(X_bin), 1, activation='sigmoid'))

    # Train the neural network
    nn4.train(data=train_bin, X=X_bin, Y='Y', test_set=test_bin, epochs=100, learning_rate=0.01, plot=False, verbose=args.verbose)
    pred_mlp4_bin = nn4.predict(appli_bin, binary=True)

    bin_mlp4_acc = binary_accuracy(y_true=appli_bin['Y'], y_pred=pred_mlp4_bin) * 100
    data_balance_bin_mlp4 = np.sum(pred_mlp4_bin) / len(pred_mlp4_bin)

    pc.verbose_display(f"Model accuracy is {round(bin_mlp4_acc, 3)}%", args.verbose)
    pc.verbose_display(f'Model found {round(data_balance_bin_mlp4*100, 3)}% of 1s', args.verbose)

    pc.verbose_display('Model parameters', args.verbose)
    mlp4_coeffs = [x[0] for x in nn4.get_weights(param='weights')['weights 0']]
    pc.verbose_display(mlp4_coeffs, args.verbose)


    ##########################################

    pc.verbose_display('\n=== TEST DEEP MLP ===\n', args.verbose)

    dnn = MLP(loss='bce', optimizer='adam')
    dnn.add(Layer(len(X_bin), 10, activation='relu'))
    dnn.add(Layer(10, 5, activation='relu'))
    dnn.add(Layer(5, 3, activation='relu'))
    dnn.add(Layer(3, 1, activation='sigmoid'))

    # Train the neural network
    dnn.train(data=train_bin, X=X_bin, Y='Y', test_set=test_bin, epochs=100, learning_rate=0.01, plot=False, verbose=args.verbose)
    pred_dmlp_bin = dnn.predict(appli_bin, binary=True)

    bin_dmlp_acc = binary_accuracy(y_true=appli_bin['Y'], y_pred=pred_dmlp_bin) * 100
    data_balance_bin_dmlp = np.sum(pred_dmlp_bin) / len(pred_dmlp_bin)

    pc.verbose_display(f"Model accuracy is {round(bin_dmlp_acc, 3)}%", args.verbose)
    pc.verbose_display(f'Model found {round(data_balance_bin_dmlp*100, 3)}% of 1s', args.verbose)

    pc.verbose_display('Model parameters', args.verbose)
    dmlp_coeffs = [x[0] for x in dnn.get_weights(param='weights')['weights 0']]
    pc.verbose_display(dmlp_coeffs, args.verbose)

    ##########################################

    pc.verbose_display('\n=== TEST DEEP MLP 2 ===\n', args.verbose)

    dnn2 = MLP(loss='bce', optimizer='rmsprop')
    dnn2.add(Layer(len(X_bin), 10, activation='relu'))
    dnn2.add(Layer(10, 5, activation='relu'))
    dnn2.add(Layer(5, 3, activation='relu'))
    dnn2.add(Layer(3, 1, activation='sigmoid'))

    # Train the neural network
    dnn2.train(data=train_bin, X=X_bin, Y='Y', test_set=test_bin, epochs=100, learning_rate=0.01, plot=False, verbose=args.verbose)
    pred_dmlp2_bin = dnn2.predict(appli_bin, binary=True)

    bin_dmlp2_acc = binary_accuracy(y_true=appli_bin['Y'], y_pred=pred_dmlp2_bin) * 100
    data_balance_bin_dmlp2 = np.sum(pred_dmlp2_bin) / len(pred_dmlp2_bin)

    pc.verbose_display(f"Model accuracy is {round(bin_dmlp2_acc, 3)}%", args.verbose)
    pc.verbose_display(f'Model found {round(data_balance_bin_dmlp2*100, 3)}% of 1s', args.verbose)

    pc.verbose_display('Model parameters', args.verbose)
    dmlp2_coeffs = [x[0] for x in dnn2.get_weights(param='weights')['weights 0']]
    pc.verbose_display(dmlp2_coeffs, args.verbose)


    ##########################################

    pc.verbose_display('\n=== COMPARE OR ===\n', args.verbose)


    pc.verbose_display('Real OR', args.verbose)
    pc.verbose_display(real_OR, args.verbose)

    pc.verbose_display('SM OR', args.verbose)
    sm_or = (logit.predict(for_odd_1.to_numpy()) / logit.predict(for_odd_0.to_numpy()))[0]
    pc.verbose_display(sm_or, args.verbose)

    pc.verbose_display('Perceptron OR', args.verbose)
    pt_or = pt.predict(pd.DataFrame(for_odd_1).transpose())[0] / pt.predict(pd.DataFrame(for_odd_0).transpose())[0]
    pc.verbose_display(pt_or, args.verbose)

    pc.verbose_display('MLP OR', args.verbose)
    mlp_or = nn.predict(pd.DataFrame(for_odd_1).transpose())[0] / nn.predict(pd.DataFrame(for_odd_0).transpose())[0]
    pc.verbose_display(mlp_or, args.verbose)

    pc.verbose_display('MLP2 OR', args.verbose)
    mlp2_or = nn2.predict(pd.DataFrame(for_odd_1).transpose())[0] / nn2.predict(pd.DataFrame(for_odd_0).transpose())[0]
    pc.verbose_display(mlp2_or, args.verbose)

    pc.verbose_display('MLP3 OR', args.verbose)
    mlp3_or = nn3.predict(pd.DataFrame(for_odd_1).transpose())[0] / nn3.predict(pd.DataFrame(for_odd_0).transpose())[0]
    pc.verbose_display(mlp3_or, args.verbose)

    pc.verbose_display('MLP4 OR', args.verbose)
    mlp4_or = nn4.predict(pd.DataFrame(for_odd_1).transpose())[0] / nn4.predict(pd.DataFrame(for_odd_0).transpose())[0]
    pc.verbose_display(mlp4_or, args.verbose)

    pc.verbose_display('DNN OR', args.verbose)
    dmlp_or = dnn.predict(pd.DataFrame(for_odd_1).transpose())[0] / dnn.predict(pd.DataFrame(for_odd_0).transpose())[0]
    pc.verbose_display(dmlp_or, args.verbose)

    pc.verbose_display('DNN2 OR', args.verbose)
    dmlp2_or = dnn2.predict(pd.DataFrame(for_odd_1).transpose())[0] / dnn2.predict(pd.DataFrame(for_odd_0).transpose())[0]
    pc.verbose_display(dmlp2_or, args.verbose)
    
    pc.write(f'{i},{quad_add},{data_balance},{logit_acc},{bin_pt_acc},{bin_mlp_acc},{bin_mlp2_acc},{bin_mlp3_acc},{bin_mlp4_acc},{bin_dmlp_acc},{bin_dmlp2_acc},{real_OR},{sm_or},{pt_or},{mlp_or},{mlp2_or},{mlp3_or},{mlp4_or},{dmlp_or},{dmlp2_or}', output_path)
# except:
#     pc.verbose_display('Error', verbose=args.verbose)
#     pass

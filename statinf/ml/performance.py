import warnings
import pandas as pd
import numpy as np


class BinaryPerformance:
    def __init__(self, y_true, y_pred):
        warnings.filterwarnings('ignore')
        # Put data to a DF
        for_conf = pd.DataFrame({'true': y_true,
            'pred': y_pred,
            'perf': ''})

        # Compute True and False positives/negatives
        for_conf['perf'][(for_conf.true == 1) & (for_conf.pred == 1)] = 'true_positive'
        for_conf['perf'][(for_conf.true == 0) & (for_conf.pred == 0)] = 'true_negative'
        for_conf['perf'][(for_conf.true == 1) & (for_conf.pred == 0)] = 'false_negative'
        for_conf['perf'][(for_conf.true == 0) & (for_conf.pred == 1)] = 'false_positive'
        # Number of True and False positives/negatives
        self.tp = len(for_conf[(for_conf.perf == 'true_positive')])
        self.tn = len(for_conf[(for_conf.perf == 'true_negative')])
        self.fn = len(for_conf[(for_conf.perf == 'false_negative')])
        self.fp = len(for_conf[(for_conf.perf == 'false_positive')])
        self.m = len(y_pred)
        # Confusion matrix
        self.conf = pd.DataFrame({'True 0': [0., 0.], 'True 1': [0., 0.]}, index=['Predicted 0', 'Predicted 1'])
        # Fill confusion matrix
        self.conf['True 0']['Predicted 0'] = self.tn/self.m
        self.conf['True 0']['Predicted 1'] = self.fp/self.m
        self.conf['True 1']['Predicted 0'] = self.fn/self.m
        self.conf['True 1']['Predicted 1'] = self.tp/self.m
        warnings.filterwarnings('default')

    def accuracy(self):
        return((self.tp + self.tn)/self.m)

    def confusion(self):
        return(self.conf * 100)

    def precision(self):
        return(self.tp / (self.tp + self.fp))

    def recall(self):
        return(self.tp / (self.tp + self.fn))

    def F1_score(self):
        return(2 * (self.precision() * self.recall()) / (self.precision() + self.recall()))

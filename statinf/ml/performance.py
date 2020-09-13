import warnings
import pandas as pd
import numpy as np
import math


class BinaryPerformance:
    def __init__(self, y_true, y_pred):
        """Gives detailed perfomance metrics for binary calssification models.

        :param y_true: Array of true targets.
        :type y_true: :obj:`numpy.ndarray`
        :param y_pred: Array of predicted targets.
        :type y_pred: :obj:`numpy.ndarray`
        """
        warnings.filterwarnings('ignore')
        # Format y_true
        if type(y_true) in [pd.Series, pd.DataFrame]:
            true = list(y_true.values)
        elif type(y_true) == list:
            true = y_true
        elif type(y_true) == np.ndarray:
            if y_true.shape == (len(y_true), 1):
                true = [x[0] for x in np.asarray(y_true)]
            elif y_true.shape == (len(y_true),):
                true = list(y_true)
            else:
                raise TypeError('Cannot properly read shape for y_true.')
        else:
            raise TypeError('Type for y_true is not valid.')
        # Format y_pred
        if type(y_pred) in [pd.Series, pd.DataFrame]:
            pred = list(y_pred.values)
        elif type(y_pred) == list:
            pred = y_pred
        elif type(y_pred) == np.ndarray:
            if y_pred.shape == (len(y_pred), 1):
                pred = [x[0] for x in np.asarray(y_pred)]
            elif y_pred.shape == (len(y_pred),):
                pred = list(y_pred)
            else:
                raise TypeError('Cannot properly read shape for y_pred.')
        else:
            raise TypeError('Type for y_pred is not valid.')

        # Put data to a DF
        for_conf = pd.DataFrame({'true': true,
            'pred': pred,
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
        """Binary accuracy of the model. Percentage of equal values between :obj:`y_true` and :obj:`y_pred`.

        :formula: .. math:: accuracy = \\dfrac{TP + TN}{n}

        :return: Accuracy
        :rtype: :obj:`float`
        """
        return((self.tp + self.tn)/self.m)

    def confusion(self):
        """Confusion matrix

        :return: Confusion matrix

            +-----------------+------------+------------+
            |                 | **True 0** | **True 0** |
            +=================+============+============+
            | **Predicted 0** | :math:`TN` | :math:`TN` |
            +-----------------+------------+------------+
            | **Predicted 1** | :math:`FP` | :math:`TP` |
            +-----------------+------------+------------+

        :rtype: :obj:`pandas.DataFrame`
        """
        return(self.conf * 100)

    def precision(self):
        """Precision metric, proportion of actual 1 values amongst the ones predicted.

        :formula: .. math:: precision = \\dfrac{TP}{TP + FP}

        :return: Precision
        :rtype: :obj:`float`
        """
        return(self.tp / (self.tp + self.fp))

    def recall(self):
        """  Recall metric, proportion of values we predicted as one from the actuals ones.

        :formula: .. math:: recall = \\dfrac{TP}{TP + FN}

        :return: Recall
        :rtype: :obj:`float`
        """
        return(self.tp / (self.tp + self.fn))

    def F1_score(self):
        """F1-score

        :formula: .. math:: F_{1} = 2 \\cdot \\dfrac{precision \\times recall}{precision + recall}

        :return: F1-score
        :rtype: :obj:`float`
        """
        return(2 * (self.precision() * self.recall()) / (self.precision() + self.recall()))


def mean_squared_error(y_true, y_pred, root=False):
    """Mean Squared Error

    :param y_true: Real values on which to compare.
    :type y_true: :obj:`numpy.ndarray`
    :param y_pred: Predicted values.
    :type y_pred: :obj:`numpy.ndarray`
    :param root: Return Root Mean Squared Error (RMSE), defaults to False.
    :type root: :obj:`bool`, optional

    :formula: :math:`loss = \\dfrac{1}{m} \\times \\sum_{i=1}^{m} (y_i - \\hat{y}_i)^2`

    :references: * Friedman, J., Hastie, T. and Tibshirani, R., 2001. `The elements of statistical learning <https://web.stanford.edu/~hastie/Papers/ESLII.pdf>`_. Ch. 2, pp. 24.

    :return: Mean Squared Error or its root.
    :rtype: float
    """

    warnings.filterwarnings('ignore')

    # Format y_true
    if type(y_true) in [pd.Series, pd.DataFrame]:
        true = np.array(y_true.values)
    elif type(y_true) in [list]:
        true = np.array(y_true)
    elif type(y_true) == np.ndarray:
        true = y_true
    else:
        raise TypeError('Type for y_true is not valid.')

    # Format y_pred
    if type(y_pred) in [pd.Series, pd.DataFrame]:
        pred = np.array(y_pred.values)
    elif type(y_pred) in [list]:
        pred = np.array(y_pred)
    elif type(y_pred) == np.ndarray:
        pred = y_pred
    else:
        raise TypeError('Type for y_pred is not valid.')

    warnings.filterwarnings('default')

    # Compute the square of the difference
    loss = (pred - true)**2

    if root:
        return math.sqrt(loss.mean())
    else:
        return loss.mean()


# MAPE formula
def mape(y_true, y_pred, weights=False):
    """Computes the Mean Absolute Percentage Error (MAPE) or Weighted Mean Absolute Percentage Error (WMAPE).

    :param y_true: Real values on which to compare.
    :type y_true: :obj:`numpy.array`
    :param y_pred: Predicted values.
    :type y_pred: :obj:`numpy.array`
    :param weights: Compute WMAPE.
    :type weights: :obj:`bool`

    :formula:

    * :math:`MAPE(y, \\hat{y}) = \\dfrac{100}{n} \\sum_{i=1}^{n} \\dfrac{|y - \\hat{y}|}{y}`
    * :math:`WMAPE(y, \\hat{y}) = 100 \\dfrac{\\sum_{i=1}^{n} {\\dfrac{|y - \\hat{y}|}{y}} \\times y}{\\sum_{i=1}^{n} y}`

    :return: Mean Absolute Percentage Error as percentage.
    :rtype: float
    """

    warnings.filterwarnings('ignore')
    # Format y_true
    if type(y_true) in [pd.Series, pd.DataFrame]:
        _true = list(y_true.values)
    elif type(y_true) == list:
        _true = y_true
    elif type(y_true) == np.ndarray:
        if y_true.shape == (len(y_true), 1):
            _true = [x[0] for x in np.asarray(y_true)]
        elif y_true.shape == (len(y_true),):
            _true = list(y_true)
        else:
            raise TypeError('Cannot properly read shape for y_true.')
    else:
        raise TypeError('Type for y_true is not valid.')
    # Format y_pred
    if type(y_pred) in [pd.Series, pd.DataFrame]:
        _pred = list(y_pred.values)
    elif type(y_pred) == list:
        _pred = y_pred
    elif type(y_pred) == np.ndarray:
        if y_pred.shape == (len(y_pred), 1):
            _pred = [x[0] for x in np.asarray(y_pred)]
        elif y_pred.shape == (len(y_pred),):
            _pred = list(y_pred)
        else:
            raise TypeError('Cannot properly read shape for y_pred.')
    else:
        raise TypeError('Type for y_pred is not valid.')

    preds = pd.DataFrame({'true': _true, 'pred': _pred})
    preds['abs_err'] = np.abs(preds['true'] - preds['pred'])
    preds['err_contrib'] = preds['abs_err']/preds['true']

    if weights:
        preds['err_contrib'] = preds['err_contrib'] * preds['true'] * 100
        out = preds['err_contrib'].sum() / preds.true.sum()
    else:
        out = 100 * preds['err_contrib'].sum() / preds['true'].sum()

    return out

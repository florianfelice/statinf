import numpy as np
import pandas as pd

from scipy.stats import norm

from ..misc import test_summary, format_object
from ..regressions.LinearModels import OLS


# This is being validated with statsmodels implementation:
# https://github.com/statsmodels/statsmodels/blob/master/statsmodels/tsa/stattools.py#L159
# As well as checked with the litterature cited below
"""
Aligned with the mackinnonp function from adfvalues in statsmodels
"""
_stats_value_max = {
    "none": [-1.04, -1.53, -2.68, -3.09, -3.07, -3.77],
    "c": [-1.61, -2.62, -3.13, -3.47, -3.78, -3.93],
    "ct": [-2.89, -3.19, -3.50, -3.65, -3.80, -4.36],
    # "ctt": [-3.21, -3.51, -3.81, -3.83, -4.12, -4.63],
}

_dist_quant_small = {
    "none": [0.6344, 1.2378, 3.2496 * 1e-2],
    "c": [2.1659, 1.4412, 3.8269 * 1e-2],
    "ct": [3.2512, 1.6047, 4.9588 * 1e-2],
    "ctt": [4.0003, 1.658, 4.8288 * 1e-2],
}

_dist_quant_large = {
    "none": [0.4797, 9.3557, -0.6999, 3.3066],
    "c": [1.7339, 9.3202, -1.2745, -1.0368],
    "ct": [2.5261, 6.1654, -3.7956, -6.0285],
    "ctt": [3.0778, 4.9529, -4.1477, -5.9359],
}

_crit_values = {
    'none': [-2.56574, -1.941, -1.61682],
    'c': [-3.43035, -2.86154, -2.56677],
    'ct': [-3.95877, -3.41049, -3.12705],
    'ctt': [-4.37113, -3.83239, -3.55326],
}

# MacKinnon p-values
def _MacKinnon_pvalues(statvalue, trend, k=1):
    if statvalue <= _stats_value_max[trend][k - 1]:
        values = _dist_quant_small[trend]
    else:
        values = np.array(_dist_quant_large[trend]) * np.array([1, 1e-1, 1e-1, 1e-2])

    return norm.cdf(np.polyval(values[::-1], statvalue))


# Augmented Dickey-Fuller
def adf_test(x, lag='auto', trend='c', metric='aic', return_tuple=False):
    """
    Augmented Dickey-Fuller test.

    The test is primarily used for unit root testing for univariate time series in order to check for stationnarity.
    The null hypothesis :math:`H_0` is that the root is 1 and series is not stationary.

    The methodology checks the T-statistic of the variable in the linear regression and compares with critical value from MacKinnon.
    The default behavior will determine the lag minimizing the information criteria (AIC or BIC). User can also choose the lag.


    :param x: Time series to be tested.
    :type x: :obj:`pandas.DataFrame` or :obj:`numpy.ndarray`
    :param lag: Lag to be considered in the series, defaults to 'auto'.
    :type lag: :obj:`int`, optional
    :param trend: Trend to be included in the regression, defaults to 'c'. Values can be:

        * 'none': only consider the lag.
        * 'c': only include a constant.
        * 't': only include the trend component.
        * 'ct': include both constant and trend components.

    :type trend: :obj:`str`, optional
    :param metric: Type of metric to use in the linear regression for selecting the optimal lag, defaults to 'aic'. Can either be 'aic' or 'bic'.
    :type metric: :obj:`str`, optional
    :param return_tuple: Return a tuple with test statistic, p-value, lag. Defaults to False.
    :type return_tuple: :obj:`bool`

    :example:

    >>> from statinf import stats
    >>> stats.adf_test(series,  trend='ct')
    ... +------------------------------------------------------------+
    ... |                     Dickey-Fuller test                     |
    ... +------------+----------------+------------+---------+-------+
    ... |     df     | Critical value | Stat value | p-value |   H0  |
    ... +------------+----------------+------------+---------+-------+
    ... |        107 |       -3.41049 |  -1.298438 | 0.88831 | True  |
    ... +------------+----------------+------------+---------+-------+
    ... * We cannot reject H0: the series is not stationarity
    ... * Used 12 lags


    :return: Summary for the test or tuple statistic, critical value, p-value.
    :rtype: :obj:`str` or :obj:`tuple`


    :references: * Hamilton, J. D. (1994). Time series analysis. Princeton university press, Princeton, NJ.
        * Wooldridge, J. M. (2010). Econometric analysis of cross section and panel data. MIT press.
        * Cameron, A. C., & Trivedi, P. K. (2009). Microeconometrics using stata (Vol. 5, p. 706). College Station, TX: Stata press.
        * MacKinnon, J. G. (2010). Critical values for cointegration tests (No. 1227). Queen's Economics Department Working Paper.
    """

    assert trend in ('none', 'c', 'ct'), f"The value for trend needs to be 'none', 'c' or 'ct'. Got '{trend}'"

    _df = pd.DataFrame(x).reset_index(drop=True)
    _df.columns = ['x']
    _df['trend'] = _df.index + 1.
    n = _df.shape[0]

    nb_ext = {'none': 0, 'c': 0, 't': 1, 'ct': 2}

    maxlag = int(np.ceil(12.0 * np.power(n / 100.0, 1 / 4.0)))
    # -1 for the diff
    maxlag = min(n // 2 - nb_ext[trend] - 1, maxlag)

    if lag != 'auto':
        assert lag <= maxlag, ValueError(f'Lag value is too high. Maximum is {maxlag}, got {lag}')

    for _lag in range(1, maxlag + 2):
        _df[f'x_lag_{_lag}'] = _df.x.shift(_lag)

    for _lag in range(1, maxlag + 2):
        if _lag == 1:
            _df['delta_lag_1'] = _df['x'] - _df['x_lag_1']
        else:
            _df[f'delta_lag_{_lag}'] = _df[f'x_lag_{_lag - 1}'] - _df[f'x_lag_{_lag}']

    filtered = _df.dropna().copy()

    # Define whether we need to add trend component
    trend_var = ' + trend ' if trend in ['t', 'ct'] else ''
    # Add an intercept if user want a trend
    cst = trend in ['c', 'ct']

    base_cols = ['delta_lag_1', 'x_lag_1', 'trend']

    # Defining lag
    if lag == 'auto':
        perfs = {}
        # Define optimal lag
        for _lag in range(1, maxlag + 1):
            _cols = [f'delta_lag_{i}' for i in range(2, _lag + 2)]
            _lag_cols = ' + '.join(_cols)
            _reg_formula = 'delta_lag_1 ~ x_lag_1 + ' + _lag_cols
            # Add constant variable if user requests
            _reg_formula += trend_var
            _temp_df = _df[_cols + base_cols].dropna().copy()
            # Fit OLS
            temp_ols = OLS(_reg_formula, data=_temp_df, fit_intercept=cst)
            # perfs[l] = temp_ols
            perfs.update({temp_ols._aic(): _lag})

        best_perf = np.min([k for k in perfs.keys()])
        best_lag = perfs.get(best_perf) - 1
    else:
        best_lag = lag

    # Parse lagged columns to use as formula
    cols_lag = [f'delta_lag_{i}' for i in range(2, best_lag + 2)]

    if best_lag == 0:
        _reg_formula = 'delta_lag_1 ~ x_lag_1'
    else:
        _lag_cols = ' + '.join(cols_lag)
        _reg_formula = 'delta_lag_1 ~ x_lag_1 + ' + _lag_cols

    # Add constant variable if user requests
    _reg_formula += trend_var

    # Prepare final data set with created lags
    filtered = _df[cols_lag + base_cols].dropna().copy()
    n = filtered.shape[0]

    # Run OLS and get the stat value of x_lag_1 from summary
    _ols = OLS(_reg_formula, data=filtered, fit_intercept=cst)
    _summ = _ols.summary(True)
    stat_value = _summ['t-values'][_summ.Variables == 'x_lag_1'].max()

    # Get the MacKinnon pvalue based on the OLS stat value
    p_val = _MacKinnon_pvalues(statvalue=stat_value, trend=trend, k=1)

    # Format summary output
    _summ = test_summary(df=n, critical_value=_crit_values[trend][1], t_value=stat_value,
                         p_value=p_val, title='Dickey-Fuller test',
                         extra=f' * Used {best_lag} lags',
                         h0='the series is not stationarity',
                         h1='the series is stationarity')

    if return_tuple:
        return stat_value, p_val, best_lag, n
    else:
        return _summ


# Cointegration test
def coint_test(x1, x2, lag='auto', trend='c', metric='aic', return_tuple=False):
    """
    Cointegration test.

    Checks the relation between two time series. The null hypothesis :math:`H_0` is that there is no cointegration between both series.
    The methodology consists in regressing one variable on the other and checks the stationarity of the residuals.

    :param x1: First series to be compared.
    :type x1: :obj:`pandas.DataFrame` or :obj:`numpy.ndarray`
    :param x2: Second series to be compared.
    :type x2: :obj:`pandas.DataFrame` or :obj:`numpy.ndarray`
    :param lag: Lag to be considered in the series, defaults to 'auto'.
    :type lag: :obj:`int`, optional
    :param trend: Trend to be included in the regression, defaults to 'c'. See details in :py:meth:`statinf.stats.timeseries.adf_test`.
    :type trend: :obj:`str`, optional
    :param metric: Type of metric to use in the linear regression for selecting the optimal lag, defaults to 'aic'. Can either be 'aic' or 'bic'.
    :type metric: :obj:`str`, optional
    :param return_tuple: [description], defaults to False
    :param return_tuple: Return a tuple with test statistic, p-value, and relation. Defaults to False.
    :type return_tuple: :obj:`bool`

    :example:

    >>> from statinf import stats
    >>> stats.coint_test(series, series2,  trend='ct')
    ... +------------------------------------------------------------+
    ... |                     Cointegration test                     |
    ... +------------+----------------+------------+---------+-------+
    ... |     df     | Critical value | Stat value | p-value |   H0  |
    ... +------------+----------------+------------+---------+-------+
    ... |        120 |       -3.41049 |  -2.213066 |  0.4825 | True  |
    ... +------------+----------------+------------+---------+-------+
    ... * We cannot reject H0: the series are not cointegrated

    :return: Summary for the test or tuple statistic, critical value, p-value.
    :rtype: :obj:`str` or :obj:`tuple`

    :references: * Hamilton, J. D. (1994). Time series analysis. Princeton university press, Princeton, NJ.
        * Wooldridge, J. M. (2010). Econometric analysis of cross section and panel data. MIT press.
        * MacKinnon, J. G. (1991). Critical values for cointegration tests. In Eds., Long-Run Economic Relationship: Readings in Cointegration.
    """
    _df = pd.DataFrame(x1).reset_index(drop=True)
    _df.columns = ['x1']
    _df['x2'] = format_object(x2, name='x2')
    _df['trend'] = _df.index + 1.
    _df['cst'] = 1.
    n, k = _df.shape

    # Add an intercept if user want a trend
    cst = trend in ['c', 'ct']
    _reg_formula = 'x1 ~ x2'
    # Define whether we need to add trend component
    _reg_formula += ' + trend ' if trend in ['t', 'ct'] else ''

    _ols = OLS(_reg_formula, data=_df, fit_intercept=cst)
    _summ_ols = _ols.summary(True)
    res = _ols._get_error()
    # sm_ols = lm.OLS(x1, _df[['x2', 'trend', 'cst']]).fit()
    # print(res)
    # print(sm_ols.resid)

    if _ols.r_squared() < 1. - 1e-4:
        # In case of no perfect fit
        stat_value, p, bl, _n = adf_test(res, lag=lag, trend='none', metric=metric, return_tuple=True)
    else:
        raise ValueError('Your time series seem to be colinear')

    p_val = _MacKinnon_pvalues(statvalue=stat_value, trend=trend, k=k)

    vars_rel = 'negative' if _summ_ols.Coefficients[(_summ_ols.Variables == 'x2')].max() < 0 else 'positive'

    _summ = test_summary(df=n, critical_value=_crit_values[trend][1], t_value=stat_value,
                         p_value=p_val, title='Cointegration test',
                         h1_conclu=f' * The series seem to have a {vars_rel} relation',
                         h0='the series are not cointegrated',
                         h1='the series are cointegrated')

    if return_tuple:
        return stat_value, p_val, vars_rel
    else:
        return _summ

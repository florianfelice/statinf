import warnings
import math
import numpy as np
import pandas as pd
from scipy.stats import norm

# Create custom warning: Value
class ValueWarning(UserWarning):
    pass

# Create custom warning: Convergence
class ConvergenceWarning(UserWarning):
    pass


def get_significance(proba):
    if proba < 0.001:
        return('***')
    elif proba < 0.01:
        return(' **')
    elif proba < 0.05:
        return('  *')
    elif proba < 0.1:
        return('  .')
    else:
        return('   ')


def summary(s):
    """Print summary data frame to string

    :param s: Summary from model as dataframe.
    :type s: :obj:`pandas.DataFrame`

    :return: Formatted summary
    :rtype: :obj:`str`
    """
    max_var = np.max([len(v) for v in s.Variables])

    add_sp = ' ' * np.max([max_var - 17, 0])
    add_sep = '=' * np.max([max_var - 17, 0])
    space = np.max([max_var, 17])

    summ = f"============================================================================================================={add_sep}\n"
    summ += f"| Variables        {add_sp} | Coefficients   | Std. Errors  | t-values   | 95% Conf Int.            | Probabilities |\n"
    summ += f"============================================================================================================={add_sep}\n"

    for i in range(s.shape[0]):
        vari = s.Variables[i]
        c = s.Coefficients[i]
        se = s['Standard Errors'][i]
        tv = s['t-values'][i]
        pb = s['Probabilities'][i]
        sign = s['Significance'][i]
        ci_lo = s['CI_lo'][i]
        ci_hi = s['CI_hi'][i]

        c_val = round(c, 5)
        std_val = round(se, 5)
        tv_val = round(tv, 3)
        pb_val = round(pb, 3) if math.fabs(pb) < 1000 else round(pb, 3 - len(str(int(pb))))
        ci_lo = f'{round(ci_lo, 3):9}'
        ci_hi = f'{round(ci_hi, 3):9}'

        summ += f"| {vari:{space}} |  {c_val:13} | {std_val:12} | {tv_val:10} | [{ci_lo:10}, {ci_lo:10}] |  {pb_val:6}   {sign:3} |\n"
    summ += f"============================================================================================================={add_sep}\n"
    summ += f"| Significance codes: 0. < *** < 0.001 < ** < 0.01 < * < 0.05 < . < 0.1 < '' < 1                            {add_sp}|\n"
    summ += f"============================================================================================================={add_sep}\n"
    return summ


def test_summary(df, critical_value, t_value, p_value, alpha=0.05, title='', h0='H0', h1='H0 does not hold', extra='', h0_conclu='', h1_conclu=''):

    # Format for output

    blanks = int((58 - len(title)) / 2)
    right_space = ' ' * np.max([int(blanks), 0])
    left_space = ' ' * np.max([58 - len(title) - blanks, 0])
    dfree = round(df, 10 - len(str(int(df))))
    cri_value = round(critical_value, 11 - len(str(int(critical_value))))
    t_val = round(t_value, 8 - len(str(int(t_value))))
    p_v = round(p_value, 6 - len(str(int(p_value))))
    r = 'True' if alpha < p_value else 'False'

    if alpha < p_value:
        conclusion = ' * We cannot reject H0: ' + h0 + '\n'
    else:
        conclusion = ' * We reject H0, hence ' + h1 + '\n'

    summ = "+------------------------------------------------------------+\n"
    summ += f"|{right_space} {title} {left_space}|\n"
    summ += "+------------+----------------+------------+---------+-------+\n"
    summ += "|     df     | Critical value | Stat value | p-value |   H0  |\n"
    summ += "+------------+----------------+------------+---------+-------+\n"
    summ += f"| {dfree:10} | {cri_value:14} | {t_val:10} | {p_v:7} | {r:5} |\n"
    summ += "+------------+----------------+------------+---------+-------+\n"
    summ += conclusion
    summ += h0_conclu if r == 'True' else ''
    summ += h1_conclu if r == 'False' else ''
    summ += extra

    return summ


def _to_array(x, name='x'):

    warnings.filterwarnings('ignore')

    if type(x) in [pd.Series, pd.DataFrame]:
        return np.array(x.values)
    elif type(x) in [list]:
        return np.array(x)
    elif type(x) == np.ndarray:
        return x
    else:
        raise TypeError(f'Type for {name} is not valid.')
    warnings.filterwarnings('default')


def _to_list(x, name='x'):
    warnings.filterwarnings('ignore')
    if type(x) in [pd.Series, pd.DataFrame]:
        return list(x.values)
    elif type(x) == list:
        return x
    elif type(x) == np.ndarray:
        if x.shape == (len(x), 1):
            return [x[0] for x in np.asarray(x)]
        elif x.shape == (len(x),):
            return list(x)
        else:
            raise TypeError(f'Cannot properly read shape for {name}.')
    else:
        raise TypeError(f'Type for {name} is not valid.')
    warnings.filterwarnings('default')


def format_object(x, to_type='array', name='x'):
    if to_type in ['array', 'ndarray', 'to_array', 'to array']:
        return _to_array(x, name=name)
    elif to_type in ['list', 'to_list', 'to list']:
        return _to_list(x, name=name)
    else:
        raise ValueError(f'Value for to_type is not value. Can either be array or list. Got {to_type}.')

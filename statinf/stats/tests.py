import numpy as np
import math
from scipy import stats as scp

from . import descriptive as desc


def ttest(x, mu=0, alpha=0.05, two_sided=True, return_tuple=False):
    """One sample Student test

    :param x: Input variable. Format can be :obj:`numpy.array`, :obj:`list` or :obj:`pandas.Series`.
    :type x: :obj:`numpy.array`
    :param mu: Theoretical mean to be evaluated in the null hypothesis, defaults to 0.
    :type mu: :obj:`int`, optional
    :param alpha: Confidence level, defaults to 0.05.
    :type alpha: :obj:`float`, optional
    :param two_sided: Perform a two-sided test, defaults to True.
    :type two_sided: :obj:`bool`, optional
    :param return_tuple: Return a tuple with t statistic and p-value, defaults to False.
    :type return_tuple: :obj:`bool`, optional

    :formula: .. math:: t = \\dfrac{\\bar{X}_{1} - \\bar{X}_{2} }{ \\dfrac{s}{\\sqrt{n}} }

    :example:

    >>> ttest([30.02, 29.99, 30.11, 29.97, 30.01, 29.99], mu=30)
    ... +------------------------------------------------------------+
    ... |                   One Sample Student test                  |
    ... +------------+----------------+------------+---------+-------+
    ... |     df     | Critical value |    T-value | p-value |   H0  |
    ... +------------+----------------+------------+---------+-------+
    ... |          5 |   2.5705818366 |  0.7392961 | 0.49295 | True  |
    ... +------------+----------------+------------+---------+-------+
    ... * Confidence level is 95.0%, we need p > 0.025 for two-sided test
    ... * We cannot reject the hypothesis H0: X_bar = 30

    :reference: * Student. (1908). The probable error of a mean. Biometrika, 1-25.

    :return: Summary for the test or tuple statistic, p-value.
    :rtype: :obj:`str` or :obj:`tuple`
    """

    # Define test degrees of freedom
    if two_sided:
        quant_order = 1 - (alpha/2)
        h0 = f'H0: X_bar = {mu}'
    else:
        quant_order = 1 - alpha
        h0 = f'H0: X_bar <= {mu}'

    # Input vector as array
    x = np.asarray(x)
    # Sample size
    n = len(x)

    # Empirical mean
    x_bar = x.mean()
    # s estimator (variance)
    s2 = desc.var(x)
    # Degrees of freedom
    df = n - 1

    # T-statistics
    t = (x_bar - mu)/(math.sqrt(s2)/math.sqrt(n))
    if two_sided:
        t = math.fabs(t)
    # p and critical values
    p = 2.0 * (1.0 - scp.t.cdf(math.fabs(t), df=df))
    cv = scp.t.ppf(quant_order, df=df)

    # Format for output
    dfree = round(df, 10 - len(str(int(df))))
    cri_value = round(cv, 11 - len(str(int(cv))))
    t_val = round(t, 8 - len(str(int(t))))
    p_v = round(p, 6 - len(str(int(p))))
    r = 'True' if t < cri_value else 'False'
    nb_side = 'two' if two_sided else 'one'
    if t < cri_value:
        conclusion = ' * We cannot reject the hypothesis ' + h0 + '\n'
    else:
        conclusion = ' * We reject the hypothesis ' + h0 + '\n'

    # Test summary
    summ = f"+------------------------------------------------------------+\n"
    summ += "|                   One Sample Student test                  |\n"
    summ += f"+------------+----------------+------------+---------+-------+\n"
    summ += f"|     df     | Critical value |    T-value | p-value |   H0  |\n"
    summ += f"+------------+----------------+------------+---------+-------+\n"
    summ += f"| {dfree:10} | {cri_value:14} | {t_val:10} | {p_v:7} | {r:5} |\n"
    summ += f"+------------+----------------+------------+---------+-------+\n"
    summ += f" * Confidence level is {(1 - alpha)*100}%, we need p > {round(1 - quant_order, 3)} for {nb_side}-sided test\n"
    summ += conclusion

    if return_tuple:
        return t, cv, p
    else:
        return summ


# function for calculating the t-test for two independent samples
def ttest_2samp(x1, x2, alpha=0.95, paired=False, return_tuple=False):
    """Two idependant samples Student's test.

    :param x1: Input variable. Format can be :obj:`numpy.array`, :obj:`list` or :obj:`pandas.Series`.
    :type x1: :obj:`numpy.array`
    :param x2: Input variable. Format can be :obj:`numpy.array`, :obj:`list` or :obj:`pandas.Series`.
    :type x2: :obj:`numpy.array`
    :param alpha: Confidence level, defaults to 0.95.
    :type alpha: :obj:`float`, optional
    :param paired: Performs one sample test of the difference if samples are paired, defaults to False.
    :type paired: :obj:`bool`, optional
    :param return_tuple: Return a tuple with t statistic and p-value, defaults to False.
    :type return_tuple: :obj:`bool`, optional

    :formulae: .. math:: t = \\dfrac{\\bar{X}_{1} - \\bar{X}_{2} }{s}

        * if :math:`\\mathbb{V}(X_{1}) = \\mathbb{V}(X_{2})`: :math:`s = \\sqrt{ \\left( \\dfrac{(n_{1} - 1) s^{2}_{1} + (n_{2} - 1) s^{2}_{2}}{n_{1} + n_{2} - 1} \\right) \\dfrac{1}{n_{1}} \\dfrac{1}{n_{2}}}`
        * if :math:`\\mathbb{V}(X_{1}) \\neq \\mathbb{V}(X_{2})`: :math:`s = \\sqrt{ \\dfrac{s^{2}_{1}}{n_{1}} + \\dfrac{s^{2}_{2}}{n_{2}} }`
        * if :math:`n_{1} = n_{2}`: :math:`s = \\sqrt{ \\dfrac{s^{2}_{1} + s^{2}_{2}}{2} }`

    :example:

    >>> a = [30.02, 29.99, 30.11, 29.97, 30.01, 29.99]
    >>> b = [29.89, 29.93, 29.72, 29.98, 30.02, 29.98]
    >>> ttest(a, b)
    ... +------------------------------------------------------------+
    ... |                   Two Samples Student test                 |
    ... +------------+----------------+------------+---------+-------+
    ... |     df     | Critical value |    T-value | p-value |   H0  |
    ... +------------+----------------+------------+---------+-------+
    ... |         10 |  -1.8124611228 |  1.1310325 | 0.28444 | True  |
    ... +------------+----------------+------------+---------+-------+
    ...  * Confidence level is 95.0%, we need p > 0.05
    ...  * We cannot reject the hypothesis H0: X1 = X2
    ...  * Samples with unequal variances
    ...  * Same sample sizes

    :reference: * Student. (1908). The probable error of a mean. Biometrika, 1-25.

    :return: Summary for the test or tuple statistic, p-value.
    :rtype: :obj:`str` or :obj:`tuple`
    """
    x = np.asarray(x1)
    y = np.asarray(x2)

    # Sample sizes
    n1, n2 = len(x), len(y)

    if n1 == n2:
        same_samp = True
    else:
        same_samp = False

    # Compute means
    mean1, mean2 = x.mean(), y.mean()

    # Compute standard deviations
    s1 = desc.var(x)
    s2 = desc.var(y)

    # Determine whether samples have similar variance
    if (.5 < s1/s2) & (s1/s2 < 2):
        same_var = True
    else:
        same_var = False

    # Compute standard deviation for different cases
    if same_var:
        # # If samples have similar variance
        sd = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
        s1n1 = s1/n1
        s2n2 = s2/n2
        df = (s1n1 + s2n2) ** 2 / ((s1n1**2)/(n1 - 1) + (s2n2**2)/(n2 - 1))
    elif same_samp:
        # # If samples don't have same variance but same sample size
        sd = np.sqrt((s1 + s2)/2)
        df = len(x) + len(y) - 2
    else:
        sd = np.sqrt((s1/n1) + (s2/n2))
        df = len(x) + len(y) - 2

    if paired:
        # If samples are paired, we perform a 1-sample student test
        # We compare if the difference is different from 0.
        d = x - y
        t, cv, p = ttest(d, alpha=alpha, return_tuple=True)
        df = len(d)
    else:
        # Else samples are independent
        # Calculate the t statistic
        t = (mean1 - mean2) / sd

        # calculate the critical value
        cv = scp.t.ppf(1.0 - alpha, df)
        # calculate the p-value
        p = 2.0 * (1.0 - scp.t.cdf(math.fabs(t), df))

    # Format for output
    dfree = round(df, 10 - len(str(int(df))))
    cri_value = round(cv, 12 - len(str(int(cv))))
    t_val = round(t, 8 - len(str(int(t))))
    p_v = round(p, 6 - len(str(int(p))))
    r = 'True' if p > 1 - alpha else 'False'
    if p > 1 - alpha:
        conclusion = ' * We cannot reject the hypothesis H0: X1 = X2 \n'
    else:
        conclusion = ' * We reject the hypothesis H0: X1 = X2 \n'

    # Test summary
    summ = f"+------------------------------------------------------------+\n"
    summ += "|                   Two Samples Student test                 |\n"
    summ += f"+------------+----------------+------------+---------+-------+\n"
    summ += f"|     df     | Critical value |    T-value | p-value |   H0  |\n"
    summ += f"+------------+----------------+------------+---------+-------+\n"
    summ += f"| {dfree:10} | {cri_value:14} | {t_val:10} | {p_v:7} | {r:5} |\n"
    summ += f"+------------+----------------+------------+---------+-------+\n"
    summ += f" * Confidence level is {alpha*100}%, we need p > {round(1-alpha, 3)} \n"
    summ += conclusion
    summ += " * Performed test for paired samples \n" if paired else ''
    summ += " * Samples with similar variances \n" if same_var else ' * Samples with unequal variances \n'
    summ += " * Same sample sizes" if same_samp else ' * Unequal sample sizes'

    if return_tuple:
        return t, cv, p
    else:
        return summ

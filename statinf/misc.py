import warnings
import math

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
    summ = f"| Variables         | Coefficients   | Std. Errors  | t-values   | Probabilities |\n"
    summ += "==================================================================================\n"

    for i in range(s.shape[0]):
        vari = s.Variables[i]
        c = s.Coefficients[i]
        se = s['Standard Errors'][i]
        tv = s['t-values'][i]
        pb = s['Probabilities'][i]
        sign = get_significance(s['Probabilities'][i])

        c_val = round(c, 5)
        std_val = round(se, 5)
        tv_val = round(tv, 3)
        pb_val = round(pb, 3) if math.fabs(pb) < 1000 else round(pb, 3 - len(str(int(pb))))

        summ += f"| {vari:17} |  {c_val:13} | {std_val:12} | {tv_val:10} |  {pb_val:6}   {sign:3} |\n"
    summ += "==================================================================================\n"
    summ += "| Significance codes: 0. < *** < 0.001 < ** < 0.01 < * < 0.05 < . < 0.1 < '' < 1 |\n"
    summ += "==================================================================================\n"
    return summ

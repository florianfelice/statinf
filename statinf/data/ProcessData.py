import numpy as np
import pandas as pd
import re
import warnings
from types import SimpleNamespace

from ..misc import ValueWarning

# Ranking data

def rankdata(x):
    """Assigns rank to data.
    This is mainly used for analysis like Spearman's correlation.

    :param x: Input vector. Format can be :obj:`numpy.array`, :obj:`list` or :obj:`pandas.Series`.
    :type x: :obj:`numpy.array`

    :example:

    >>> rankdata([2., 5.44, 3.93, 3.3, 1.1])
    ... array([1, 4, 3, 2, 0])

    :return: Vector with ranked values.
    :rtype: :obj:`numpy.array`
    """
    x_arr = np.asarray(x)
    sorted_array = sorted(x_arr)
    rk = [sorted_array.index(i) for i in x_arr]
    return np.array(rk)


#######################################################################################################################

# Parse formula and data transformations

def parse_formula(formula, data, check_values=True, return_all=False):
    """This function is used in regression models in order to apply transformations on the data from a formula.
    It allows to apply transformations from a :obj:`str` formula. See below for examples.

    :param formula: Regression formula to be run of the form :obj:`y ~ x1 + x2`. Accepted functions are:

        * :math:`\\log(x)` \\: :obj:`log(X)`
        * :math:`\\exp(x)` \\: :obj:`exp(X)`
        * :math:`\\sqrt{x}` \\: :obj:`sqrt(X)`
        * :math:`\\cos(x)` \\: :obj:`cos(X)`
        * :math:`\\sin(x)` \\: :obj:`sin(X)`
        * :math:`x^{z}` \\: :obj:`X ** Z`
        * :math:`\\dfrac{x}{z}` \\: :obj:`X/Z`
        * :math:`x \\times z` \\: :obj:`X*Z`

    :type formula: :obj:`str`
    :param data: Data on which to perform the transformations.
    :type data: :obj:`pandas.DataFrame`
    :param check_values: For each transformation check whether the data range satisfy the domain definition of the function, defaults to True.
    :type check_values: bool, optional
    :param return_all: Returns the transformed data, column :obj:`Y` and columns :obj:`X`, defaults to False.
    :type return_all: bool, optional

    :example:

    >>> from statinf.data import parse_formula
    >>> print(input_df)
    ... +-----------+-----------+-----------+
    ... |        X1 |        X2 |         Y |
    ... +-----------+-----------+-----------+
    ... |  0.555096 |  0.681083 | -1.383428 |
    ... |  1.155661 |  0.391129 | -7.780989 |
    ... | -0.299251 | -0.445602 | -8.146673 |
    ... | -0.978311 |  1.312146 |  8.653818 |
    ... | -0.225917 |  0.522016 | -9.684332 |
    ... +-----------+-----------+-----------+
    >>> form = 'Y ~ X1 + X2 + exp(X2) + X1*X2'
    >>> new_df = parse_formula(form, data=input_df)
    >>> print(new_df)
    ... +-----------+-----------+-----------+-----------+-----------+
    ... |        X1 |        X2 |         Y |   exp(X2) |     X1*X2 |
    ... +-----------+-----------+-----------+-----------+-----------+
    ... |  0.555096 |  0.681083 | -1.383428 |  1.976017 |  0.378066 |
    ... |  1.155661 |  0.391129 | -7.780989 |  1.478649 |  0.452012 |
    ... | -0.299251 | -0.445602 | -8.146673 |  0.640438 |  0.133347 |
    ... | -0.978311 |  1.312146 |  8.653818 |  3.714134 | -1.283687 |
    ... | -0.225917 |  0.522016 | -9.684332 |  1.685422 | -0.117932 |
    ... +-----------+-----------+-----------+-----------+-----------+

    :raises ValueError: Returns an error when the data cannot satisfy the domain definition for the required transformation.

    :return: Transformed data set
    :rtype: :obj:`pandas.DataFrame`
    """

    # Parse formula
    no_space_formula = formula.replace(' ', '')
    Y_col = no_space_formula.split('~')[0]
    X_col = no_space_formula.split('~')[1].split('+')
    all_cols = data.columns

    # Non-linear transformations
    log_cols = [re.search('(?<=log\().*?(?=\))', x).group(0) for x in X_col if re.findall('log\(', x)]  # log
    exp_cols = [re.search('(?<=exp\().*?(?=\))', x).group(0) for x in X_col if re.findall('exp\(', x)]  # exp
    sqrt_cols = [re.search('(?<=sqrt\().*?(?=\))', x).group(0) for x in X_col if re.findall('sqrt\(', x)]  # sqrt
    cos_cols = [re.search('(?<=cos\().*?(?=\))', x).group(0) for x in X_col if re.findall('cos\(', x)]  # cos
    sin_cols = [re.search('(?<=sin\().*?(?=\))', x).group(0) for x in X_col if re.findall('sin\(', x)]  # sin

    # Transformation functions
    transformations_functional = {'log': {'func': np.log, 'cols': log_cols},
                                  'exp': {'func': np.exp, 'cols': exp_cols},
                                  'cos': {'func': np.cos, 'cols': cos_cols},
                                  'sin': {'func': np.sin, 'cols': sin_cols},
                                  'sqrt': {'func': np.sqrt, 'cols': sqrt_cols},
                                  }
    # Apply transformations
    for key, transformation in transformations_functional.items():
        for c in transformation['cols']:
            col_to_transform = c  # .split('(')[1].split(')')[0]
            # Transform
            data.loc[:, f'{key}({col_to_transform})'] = transformation['func'](data[col_to_transform])

    # Multiplications, power and ration functions
    pow_cols = [x for x in X_col if re.findall('[a-zA-Z0-9\(\)][*][*][a-zA-Z0-9]', x)]  # X1 ** x
    inter_cols = [x for x in X_col if re.findall('[a-zA-Z0-9\(\)][*][a-zA-Z0-9]', x)]  # X1 * X2
    div_cols = [x for x in X_col if re.findall('[a-zA-Z0-9\(\)][/][a-zA-Z0-9]', x)]  # X1 / X2

    # Exponents
    for c in pow_cols:
        c_left = c.split('**')[0]
        c_power = c.split('**')[1]
        # Get components as number or column from data
        left = data[c_left].values if c_left in data.columns else float(c_left)
        power = data[c_power].values if c_power in data.columns else float(c_power)
        # Transform
        data.loc[:, c] = left ** power
    # Multiplications
    for c in inter_cols:
        c_left = c.split('*')[0]
        c_right = c.split('*')[1]
        # Get components as number or column from data
        try:
            left = data[c_left].values if c_left in list(data.columns) + X_col else float(c_left)
            right = data[c_right].values if c_right in list(data.columns) + X_col else float(c_right)
        except Exception:
            raise ValueError(f'Columns {c_left} or {c_right} not found in data.')
        # Transform
        data.loc[:, c] = left * right
    # Divide
    for c in div_cols:
        c_num = c.split('/')[0]
        c_denom = c.split('/')[1]
        # Get components as number or column from data
        num = data[c_num].values if c_num in list(data.columns) + X_col else float(c_num)
        denom = data[c_denom].values if c_denom in list(data.columns) + X_col else float(c_denom)
        if check_values:
            assert (denom == 0.).sum() == 0, f'Column {col_to_transform} contains null values.'
        # Transform
        data.loc[:, c] = num / denom
    if '1' in X_col:
        data['1'] = 1

    # Putting pandas' warning message back
    pd.options.mode.chained_assignment = "warn"

    if return_all:
        return data, X_col, Y_col
    else:
        return data


#######################################################################################################################

# Adding One Hot Encoding
def OneHotEncoding(data, columns, drop=True, verbose=False):
    """Performs One Hot Encoding (OHE) usally used in Machine Learning.

    :param data: Data Frame on which we apply One Hot Encoding.
    :type data: :obj:`pandas.DataFrame`
    :param columns: Column to be converted to dummy variables.
    :type columns: :obj:`list`
    :param drop: Drop the column that needs to be converted to dummies, defaults to True.
    :type drop: :obj:`bool`, optional
    :param verbose: Display progression, defaults to False.
    :type verbose: :obj:`bool`, optional

    :example:

        >>> from statinf.data import OneHotEncoding
        >>> print(df)
        ... +----+--------+----------+-----+
        ... | Id | Gender | Category | Age |
        ... +----+--------+----------+-----+
        ... |  1 | Male   |        A |  23 |
        ... |  2 | Female |        B |  21 |
        ... |  3 | Female |        A |  31 |
        ... |  4 | Male   |        C |  22 |
        ... |  5 | Female |        A |  26 |
        ... +----+--------+----------+-----+
        >>> # Encoding columns "Gender" and "Category"
        >>> new_df = OneHotEncoding(df, columns=["Gender", "Category"])
        >>> print(new_df)
        ... +----+---------------+------------+------------+-----+
        ... | Id | Gender_Female | Category_A | Category_B | Age |
        ... +----+---------------+------------+------------+-----+
        ... |  1 |             0 |          1 |          0 |  23 |
        ... |  2 |             1 |          0 |          1 |  21 |
        ... |  3 |             1 |          1 |          0 |  31 |
        ... |  4 |             0 |          0 |          0 |  22 |
        ... |  5 |             1 |          1 |          0 |  26 |
        ... +----+---------------+------------+------------+-----+

    :return: Transformed data with One Hot Encoded variables.
    :rtype: :obj:`pandas.DataFrame`
    """

    dataset = data.copy()

    try:
        if dataset.meta._ohe_exists:
            dataset.meta._ohe_exists = True
    except Exception:
        dataset.meta = SimpleNamespace()
        dataset.meta._ohe_exists = True
        dataset.meta._ohe = {}

    cols = [columns] if type(columns) == str else columns

    # Start encoding column by column
    for column in cols:
        # Get all values from the column
        all_values = dataset[column].unique()
        # Add column metadata
        dataset.meta._ohe.update({column: [f'{column}_{val}' for val in all_values]})

        # Encode values
        for val in all_values:
            if verbose:
                print('Encoding for value: ' + str(val))
            colname = column + '_' + str(val)
            dataset.loc[:, colname] = 0
            dataset.loc[dataset[column] == val, colname] = 1

        if drop:
            dataset.drop(columns=[column], inplace=True)
    
    return(dataset)


#######################################################################################################################

# convert an array of values into a dataset matrix: used for LSTM data pre-processing
def create_dataset(data, look_back=1):
    """Function to convert a DataFrame to array format readable for keras LSTM.

    :param data: DataFrame on which to aply the transformation.
    :type data: :obj:`pandas.DataFrame`
    :param look_back: Number of periods in the past to consider (defaults 1)., defaults to 1
    :type look_back: :obj:`int`, optional

    :example:
        >>> from statinf.data import create_dataset
        >>> create_dataset(df)

    :return: * Features X converted for keras LSTM.
        * Dependent variable Y converted for keras LSTM.
    :rtype: * :obj:`numpy.array`
        * :obj:`numpy.array`
    """

    dataset = data.copy()

    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

"""
Various functions for preparing a training data set of 
features for use in data-driven modelling and machine 
learning.

Contents
 - split_name(name)
 - t_inc_str(inc)
 - name_with_t_inc(name, inc)
 - add_timestep_indices(data, cols=None)
 - var_name_sequences(names, t0, tn, step=1)
 - add_previous_or_subsequent_value(data, n, cols=None, prev=False,
                                    dropna=False)
 - add_subsequent_values(data, n=1, cols=None, dropna=False)
 - add_previous_values(data, n=1, cols=None, dropna=False)
 - add_differences(data, n=1, cols=None, dropna=False, sub='_m')
 - add_rolling_averages(data, window_length, cols=None,
                        dropna=False, sub='_ra')
 - add_filtered_values_savgol(data, window_length, polyorder, cols=None,
                              dropna=False, pre='', sub='_sgf', *args,
                              **kwargs)
 - add_derivatives_savgol(data, window_length, delta, polyorder=2,
                          cols=None, dropna=False, pre='d',
                          sub='/dt_sgf', *args, **kwargs)
 - add_ewmas(data, cols=None, dropna=False, alpha=0.4, sub='_ewma',
             *args, **kwargs)
 - polynomial_features(y_in, order=3)
 - polynomial_feature_labels(n_vars, order, names=None,
                             vstr='x', psym='**')
 - feature_dataframe_from_expressions(data, expressions)
 - feature_array_from_expressions(data, expressions)
"""
import re
from itertools import chain
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter


def split_name(name):
    """Identifies the timestep increment in a parameter
    name of the format 'Name[t+i]' and returns name, i
    where name is the string before '[' and i is the
    integer timestep increment/decrement.

    Returns:
        i (int): Timestep increment

    Raises:
        ValueError if name contains '[' or ']' but it does
            not match the convention 'Name[t+i]' (spaces
            not allowed inside brackets).

    Examples:
    >>> split_name('T1[t-1]')
    ('T1[t-1]', -1)
    >>> split_name('T2') == ('T2', None)
    True
    """
    if not (']'in name or '[' in name):
        return name, None
    if not re.match(r'^[^\[\]]+(\[[^\[\]]+\])$', name):
        raise ValueError(f"Invalid column name '{name}'")
    matches = re.findall(r'\[(.*?)\]', name)
    inside = matches[0]
    if not re.match(r'^t([+-][0-9]+)?$', inside):
        raise ValueError(f"Invalid column name '{name}'")
    param = re.match(r'^[^\[]+', name).group()
    if inside == 't':
        t_inc = 0
    else:
        t_inc = int(inside[1:])
    return param, t_inc


def t_inc_str(inc):
    if inc == 0 or inc is None:
        return '[t]'
    else:
        return f'[t{inc:+d}]'


def name_with_t_inc(name, inc):
    return f'{name}{t_inc_str(inc)}'


def add_timestep_indices(data, cols=None):
    """Returns a copy of data with incremental timestep
    indices added to column names following the
    convention:
        'X' -> 'X[t]'
        'X[t+n]' -> 'X[t+n]' (i.e. no change)
    """
    if cols is None:
        cols = data.columns
    # Get any current param names and timestep
    # indices (e.g. '[t+1]')
    t_incs = list((col, split_name(col)) for col in cols)

    # Add missing timestep indices - assumes any columns
    # without a timestep are timestep '[t]'
    rename_map = {orig: name_with_t_inc(param, inc)
                  for (orig, (param, inc)) in t_incs}

    # Return a copy so as not to overwrite source dataframe
    return data.copy().rename(columns=rename_map)


def var_name_sequences(names, t0, tn, step=1):
    """Returns a list of variable names based on the
    convention 'X[t+n]' represents the value of the
    variable with name 'X' in the time step n steps
    from the current timestep.

    Example:
    >>> var_name_sequences(['A', 'B'], 0, 3)
    ['A[t]', 'A[t+1]', 'A[t+2]', 'B[t]', 'B[t+1]', 'B[t+2]']

    Args:
        names (list): List of variables names (strings).
        t0 (int): First time step in series.
        tn (int): End of timestep range (n + 1).
        step (int): Timestep size (1 by default).

    Returns:
        x_names (list): List of variables names (strings).
    """
    x_names = [[name_with_t_inc(name, i) for i in range(t0, tn, step)]
               for name in names]
    return list(chain(*x_names))


def add_previous_or_subsequent_value(data, n, cols=None, prev=False,
                                     dropna=False):
    """See functions add_previous_values() and
    add_subsequent_values() for usage.
    """

    data = add_timestep_indices(data)

    # Add values from a previous/future timestep t+n
    direction = -1 if prev else 1
    inc2 = direction * n
    for col in cols:
        param, inc1 = split_name(col)
        inc1 = 0 if inc1 is None else inc1
        new_col = name_with_t_inc(param, inc1 + inc2)
        ref_col = name_with_t_inc(param, inc1)
        data[new_col] = data[ref_col].shift(-inc2)

    if dropna:
        # Remove NaN values
        data = data.dropna()

    return data


def add_subsequent_values(data, n=1, cols=None, dropna=False):
    """Takes a dataframe of serial input data and adds columns
    to each row containing values from subsequent timesteps for
    use in training dynamic predictive models.

    Args:
        data (DataFrame): Input data (X).
        n (int): Number of subsequent timesteps to include.
        cols (list): List of column names to apply to (if
            cols is None, all columns used).
        dropna (bool): Remove all rows containing NaN values if
            True.

    Returns:
        data (DataFrame): New dataframe containing renamed
            columns and new columns.

    Example:
    >>> data = pd.DataFrame({'A': range(50, 55), 'B': range(100, 105)})
    >>> add_subsequent_values(data, 2)
       A[t]  B[t]  A[t+1]  B[t+1]  A[t+2]  B[t+2]
    0    50   100    51.0   101.0    52.0   102.0
    1    51   101    52.0   102.0    53.0   103.0
    2    52   102    53.0   103.0    54.0   104.0
    3    53   103    54.0   104.0     NaN     NaN
    4    54   104     NaN     NaN     NaN     NaN
    """

    # If cols not specified use all columns
    if cols is None:
        cols = data.columns

    for i in range(n):
        data = add_previous_or_subsequent_value(data, i+1, cols=cols,
                                                prev=False, dropna=False)

    if dropna:
        # Remove NaN values
        data = data.dropna()

    return data


def add_previous_values(data, n=1, cols=None, dropna=False):
    """Takes a dataframe of serial input data and adds columns
    to each row containing values from previous timesteps for
    use in training dynamic predictive models.

    Args:
        data (DataFrame): Input data (X).
        n (int): Number of previous timesteps to include.
        cols (list): List of column names to apply to (if
            cols is None, all columns used).
        dropna (bool): Remove all rows containing NaN values if
            True.

    Returns:
        data (DataFrame): New dataframe containing renamed
            columns and new columns.

    Example:
    >>> data = pd.DataFrame({'A': range(50, 55), 'B': range(100, 105)})
    >>> add_previous_values(data, 2)
       A[t]  B[t]  A[t-1]  B[t-1]  A[t-2]  B[t-2]
    0    50   100     NaN     NaN     NaN     NaN
    1    51   101    50.0   100.0     NaN     NaN
    2    52   102    51.0   101.0    50.0   100.0
    3    53   103    52.0   102.0    51.0   101.0
    4    54   104    53.0   103.0    52.0   102.0
    """

    # If cols not specified use all columns
    if cols is None:
        cols = data.columns

    for i in range(n):
        data = add_previous_or_subsequent_value(data, i+1, cols=cols,
                                                prev=True, dropna=False)

    if dropna:
        # Remove NaN values
        data = data.dropna()

    return data


def add_differences(data, n=1, cols=None, dropna=False, sub='_m'):
    """Takes a dataframe of serial input data and adds columns
    containing differences (change in variable between consecutive
    discrete time steps) for use in training dynamic predictive
    models.

    All existing and new columns are named using the following
    conventions:
        'T1[t]': Parameter T1 in current timestep
        'T1[t-1]': Parameter T1 in previous timestep
        'T1_m1[t]': 'T1[t]' - 'T1[t-1]'
        'T1_m2[t]': 'T1[t]' - 'T1[t-2]'

    Returns:
        data (DataFrame): New dataframe containing renamed
            columns and new columns.

    Args:
        data (DataFrame): Input data (X).
        n (int): Timestep decrement to use to create
            differences.
        cols (list or tuple): Columns to make differences for.
        dropna (bool): Remove all rows containing NaN values if
            True.
        sub (str): Sub-string to add to column names to denote
            values from previous timesteps (m means 'minus').
    """

    # If cols not specified use all columns
    if cols is None:
        cols = data.columns

    # Make a copy so as not to overwrite source dataframe
    data = data.copy()

    # Check required columns already exist, if not, create them
    incomplete_cols = []
    for col in cols:
        param, inc = split_name(col)
        inc = 0 if inc is None else inc
        complete = all([name_with_t_inc(param, inc - i - 1) in data
                       for i in range(n)])
        if not complete:
            incomplete_cols.append(col)

    # Add columns for difference calculation
    data = add_previous_values(data, n, cols=incomplete_cols, dropna=False)

    # Add differences
    for col in cols:
        param, inc = split_name(col)
        inc = 0 if inc is None else inc
        first = name_with_t_inc(param, inc)
        second = name_with_t_inc(param, inc - n)
        new_col = name_with_t_inc(f'{param}{sub}{n}', inc)
        data[new_col] = data[first] - data[second]

    if dropna:
        # Remove NaNs
        data = data.dropna()

    return data


def add_rolling_averages(data, window_length, cols=None,
                         dropna=False, sub='_ra'):
    """Takes a dataframe of serial input data and adds columns
    containing rolling averages.

    New columns are named following the convention:
        'x1' -> 'x1_ra'
        'x1[t]' -> 'x1_ra[t]'
        'x1[t+1]' -> 'x1_ra[t+1]'

    Returns:
        data (DataFrame): New dataframe containing new columns.

    Args:
        data (DataFrame): Input data (X).
        cols (list or tuple): Columns to make d/dt estimates
            for.
        window_length (int): length of the filter window.
        dropna (bool): Remove all rows containing NaN values if
            True.
        sub (str): Text to append to end of each parameter name.
    """

    # If cols not specified use all columns
    if cols is None:
        cols = data.columns

    # Make a copy so as not to overwrite source dataframe
    data = data.copy()

    # Add filtered values for selected columns
    for col in cols:
        param, inc = split_name(col)
        new_col = name_with_t_inc(f'{param}{sub}{window_length}', inc)
        data[new_col] = data[col].rolling(window_length).mean()

    if dropna:
        # Remove NaNs
        data = data.dropna()

    return data


def add_filtered_values_savgol(data, window_length, polyorder, cols=None,
                               dropna=False, pre='', sub='_sgf', *args,
                               **kwargs):
    """Takes a dataframe of serial input data and adds columns
    containing filtered values using the Savitzky-Golay filter.

    New columns are named following the convention:
        'x1' -> 'x1_sgf'
        'x1[t]' -> 'x1_sgf[t]'
        'x1[t+1]' -> 'x1_sgf[t+1]'

    Returns:
        data (DataFrame): New dataframe containing new columns.

    Args:
        data (DataFrame): Input data (X).
        window_length (int): length of the filter window.
        polyorder (int): Order of the polynomial used to fit
            the samples.
        cols (list or tuple): Columns to make d/dt estimates
            for.
        dropna (bool): Remove all rows containing NaN values if
            True.
        pre (str): Text to add before each parameter name.
        sub (str): Text to append to end of each parameter name.
        *args, **kwargs: Any other arguments accepted by
            scipy.signal.savgol_filter (refer to Scipy
            documentation).
    """

    # If cols not specified use all columns
    if cols is None:
        cols = data.columns

    # Make a copy so as not to overwrite source dataframe
    data = data.copy()

    # Add filtered values for selected columns
    for col in cols:
        param, inc = split_name(col)
        new_col = name_with_t_inc(pre + param + sub, inc)
        data[new_col] = savgol_filter(data[col], window_length, polyorder,
                                      *args, **kwargs)

    if dropna:
        # Remove NaNs
        data = data.dropna()

    return data


def add_derivatives_savgol(data, window_length, delta, polyorder=2,
                           cols=None, dropna=False, pre='d',
                           sub='/dt_sgf', *args, **kwargs):
    """Takes a dataframe of serial input data and adds columns
    containing estimates of the derivatives using the Savitzky-
    Golay filter.

    New columns are named following the convention:
        'x1' -> 'dx1/dt_sgf'
        'x1[t]' -> 'dx1/dt_sgf[t]'
        'x1[t+1]' -> 'dx1/dt_sgf[t+1]'

    Returns:
        data (DataFrame): New dataframe containing new columns.

    Args:
        data (DataFrame): Input data (X).
        window_length (int): length of the filter window.
        delta (float): The timestep size. Default is 1.0.
        polyorder (int): Order of the polynomial used to fit
            the samples.
        cols (list or tuple): Columns to make d/dt estimates
            for.
        dropna (bool): Remove all rows containing NaN values if
            True.
        pre (str): Text to add before each parameter name.
        sub (str): Text to append to end of each parameter name.
        *args, **kwargs: Any other arguments accepted by
            scipy.signal.savgol_filter (refer to Scipy
            documentation).
    """

    data = add_filtered_values_savgol(data, window_length, polyorder,
                                      cols=cols, dropna=dropna,
                                      pre=pre, sub=sub,
                                      deriv=1, delta=delta,
                                      *args, **kwargs)

    return data


def add_ewmas(data, cols=None, dropna=False, alpha=0.4, sub='_ewma',
              *args, **kwargs):
    """Takes a dataframe of serial input data and adds columns
    containing exponentially-weighted moving averages (EWMA)
    for use in training dynamic predictive models.

    New columns are named following the convention:
        'T1' -> 'T1_ewma'
        'T1[t]' -> 'T1_ewma[t]'
        'T1[t+1]' -> 'T1_ewma[t+1]'

    Returns:
        data (DataFrame): New dataframe containing new columns.

    Args:
        data (DataFrame): Input data (X).
        cols (list or tuple): Columns to make EWMAs for.
        dropna (bool): Remove all rows containing NaN values if
            True.
        alpha (float): Decay parameter (half-life).
        sub (str): Subscript to append to each parameter name.
        *args, **kwargs: Any other arguments accepted by Series.ewm
            method (refer to Pandas documentation).
    """

    # If cols not specified use all columns
    if cols is None:
        cols = data.columns

    # Make a copy so as not to overwrite source dataframe
    data = data.copy()

    # Add EWMA values for selected columns
    for col in cols:
        param, inc = split_name(col)
        new_col = name_with_t_inc(param + sub, inc)
        data[new_col] = data[col].ewm(alpha=alpha, *args, **kwargs).mean()

    if dropna:
        # Remove NaNs
        data = data.dropna()

    return data


def polynomial_features(y_in, order=3):
    """Calculate polynomial terms up to given order for all 
    data points in y_in.  This function is similar to 
    sklearn.preprocessing.PolynomialFeatures method but 
    considerably faster.

    Args:
        y_in (array): m x n array containing m data points 
            for n input variables.
        poly_order (int): Order of polynomial to generate
            terms for (1, 2 or 3).
    
    Returns:
        y_out (array): 
    """
    n = y_in.shape[1]
    y_out_cols = []

    # Poly order 0
    y_out_cols.append(np.ones((len(y_in), 1)))

    # Poly order 1
    y_out_cols.append(y_in)

    # Poly order 2
    if order >= 2:
        for i in range(n):
            y_out_cols.append(y_in[:, i:] * y_in[:, i].reshape(-1, 1))

    # Poly order 3
    if order == 3:
        # Use poly order 2 results
        results = y_out_cols[-n:]
        for j in range(0, n):
            for result in results[j:]:
                y_out_cols.append(result * y_in[:, j].reshape(-1, 1))

    if order > 3:
        raise NotImplementedError("poly_order up to 3 implemented")

    return np.hstack(y_out_cols)


def polynomial_feature_labels(n_vars, order, names=None,
                              vstr='x', psym='**'):
    """Returns a list of strings that represent the expressions 
    of all the combinations of polynomial terms of a function 
    with n_vars variables.  The list is ordered the same way 
    as the features generated by the polynomial_features 
    function.

    Args:
        n_vars (int): Number of feature variables.
        poly_order (int): 1, 2 or 3.
        names (list): List of labels for each variable.
            If not specified, ['x0', 'x1', ... etc.]
            will be used.
        vstr (str): If names is not provided, this string is
            used to construct the variable labels (e.g. if
            vstr = 'x', names = ['x0', 'x1', ...]).
        psym (str): How to represent the power
            operator (e.g. '**' or '^').

    Example 1:
    >>> polynomial_feature_labels(2, 2)
    ['1', 'x0', 'x1', 'x0**2', 'x0*x1', 'x1**2']

    Example 2:
    >>> polynomial_feature_labels(2, 3, names=['X1', 'X2'])
    ['1',
     'X1',
     'X2',
     'X1**2',
     'X1*X2',
     'X2**2',
     'X1**3',
     'X1**2*X2',
     'X1*X2**2',
     'X2**3']

    Note, these string 'expressions' can be used to calculate
    the features using pandas.DataFrame.eval().

    Example 3:
    >>> data = pd.DataFrame({'X1': range(5), 'X2': range(5, 10)})
    >>> exps = polynomial_feature_labels(2, 3, names=data.columns)
    >>> data.eval(exps[3])  # 'X1**2'
    0     0
    1     1
    2     4
    3     9
    4    16
    Name: X1, dtype: int64
    """
    if names is None:
        names = [f'{vstr}{i}' for i in range(n_vars)]
    else:
        names = list(names)

    labels = []

    # Poly order 0
    labels.append('1')

    # Poly order 1
    labels = labels + names

    # Poly order 2
    if order >= 2:
        for i in range(n_vars):
            labels = labels + ['*'.join([names[i], names[j]])
                               for j in range(i, n_vars)]

    # Poly order 3
    if order == 3:
        for i in range(n_vars):
            for j in range(i, n_vars):
                labels = labels + ['*'.join([names[i], names[j], names[k]])
                                   for k in range(j, n_vars)]

    if order > 3:
        raise NotImplementedError("poly_order up to 3 implemented")

    for name in names:
        old = f'{name}*{name}*{name}'
        new = f'{name}{psym}3'
        for i, label in enumerate(labels):
            labels[i] = label.replace(old, new)
        old = f'{name}*{name}'
        new = f'{name}{psym}2'
        for i, label in enumerate(labels):
            labels[i] = label.replace(old, new)

    return labels


def feature_dataframe_from_expressions(data, expressions):
    """Generate dataframe of calculated values using list of expressions
    and set of data.  If data is a dataframe, the column names must
    be used in the expressions.  If data is an array, the expressions
    must use 'x0', 'x1', etc. to reference the columns.

    Args:
        data (DataFrame or array): Input data.
        expressions (list): List of expressions as strings.

    Returns:
        feature_data (DataFrame): Calculated data.

    Example:
    >>> data = pd.DataFrame({'x0': range(4), 'x1': range(1, 5)})
    >>> data
       x0  x1
    0   0   1
    1   1   2
    2   2   3
    3   3   4
    >>> expressions = ['1', 'x0*x1', 'x1**2']
    >>> feature_dataframe_from_expressions(data, expressions)
        1  x0*x1  x1**2
    0  1.0    0.0    1.0
    1  1.0    2.0    4.0
    2  1.0    6.0    9.0
    3  1.0   12.0   16.0
    """
    feature_data = feature_array_from_expressions(data, expressions)
    return pd.DataFrame(feature_data, index=data.index,
                        columns=expressions)


def feature_array_from_expressions(data, expressions):
    """Generate array of calculated values using list of expressions
    and set of data.  If data is a dataframe, the column names must
    be used in the expressions.  If data is an array, the expressions
    must use 'x0', 'x1', etc. to reference the columns.

    Args:
        data (DataFrame or array): Input data.
        expressions (list): List of expressions as strings.

    Returns:
        feature_data (array): Calculated features.

    Example 1:
    >>> data = pd.DataFrame({'x0': range(4), 'x1': range(1, 5)})
    >>> data
       x0  x1
    0   0   1
    1   1   2
    2   2   3
    3   3   4
    >>> expressions = ['1', 'x0*x1', 'x1**2']
    >>> feature_array_from_expressions(data, expressions)
    array([[ 1.,  0.,  1.],
           [ 1.,  2.,  4.],
           [ 1.,  6.,  9.],
           [ 1., 12., 16.]])

    Example 2:
    >>> data = {'x0': 1, 'x1': 2}
    >>> feature_array_from_expressions(data, expressions)
    array([[1., 2., 4.]])
    """
    if isinstance(data, np.ndarray):
        if data.ndim == 1:
            data = data.reshape((1, -1))
        data = pd.DataFrame(data, columns=[f'x{i}' for i in
                                           range(data.shape[1])])
    elif isinstance(data, dict):
        #TODO: Maybe allow dict of arrays/series
        data = pd.DataFrame([data.values()], columns=data.keys())
    feature_data = np.empty((len(data), len(expressions)))
    for i, expr in enumerate(expressions):
        # TODO: Can this be speeded up?  Not all expressions require evaluating.
        feature_data[:, i] = data.eval(expr)
    return feature_data

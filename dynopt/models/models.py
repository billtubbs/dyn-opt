"""Model classes and functions for testing different types 
of regression models and machine learning algorithms for 
data-driven dynamical system identification.

Contents:
- class Model()
- class NonLinearModel(Model)
- class SparseNonLinearModel(NonLinearModel)
- class SVRMultipleOutputs
- class LinearPredictionModel(LinearRegression)
"""

from abc import ABC, abstractmethod
from functools import partial
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.metrics import r2_score
from dynopt.preprocessing.utils import feature_dataframe_from_expressions, \
                                       feature_array_from_expressions
from dynopt.models.sindy import sparsify_dynamics_lstsq
from dynopt.preprocessing.utils import polynomial_features, \
                                polynomial_feature_labels


class Model:
    """Model interface for running model-fitting and evaluation 
    experiments with data.  The interface provides a convenient
    means to generate and fit models to different combinations 
    of data inputs and outputs by using Pandas dataframes which 
    have named columns to identify each data series.

    The benefit of this approach is that you can easily automate
    model design, testing and evaluation without having to re-
    configure input and output data sets (the model instances 
    only use the relevant data they require for training and
    prediction).

    Parameters
    ----------
    x_names : list of strings,
        Names of input variables.
    y_names : list of strings,
        Names of model output variables (predictions).
    estimator : LinearRegression, Lasso, MLPRegressor or similar
        Estimator model to be used to fit the (transformed) features.
        If not specified, a LinearRegression instance is created.
    scale_inputs : boolean, optional, default False
        If True, the regressors (X) will be standardized before
        model-fitting by subtracting the mean and dividing by the
        l2-norm.  See Scikit-Learn documentation for StandardScaler.
    scale_outputs : boolean, optional, default False
        If True, the predictors (Y) will be standardized before
        model-fitting by subtracting the mean and dividing by the
        l2-norm.  See Scikit-Learn documentation for StandardScaler.
    """

    def __init__(self, x_names, y_names, estimator=None, scale_inputs=None, 
                 scale_outputs=None, *args, **kwargs):
        self._x_names = None
        self._x_labels = None
        self._x_rename_map = None
        self._y_names = None
        self._y_labels = None
        self._y_rename_map = None
        self.x_names = list(x_names)
        self.y_names = list(y_names)
        if scale_inputs:
            input_scaler = StandardScaler()
        else:
            input_scaler = None
        if scale_outputs:
            output_scaler = StandardScaler()
        else:
            output_scaler = None
        if estimator is None:
            estimator = LinearRegression(*args, **kwargs)
        self.input_scaler = input_scaler
        self.output_scaler = output_scaler
        self.estimator = estimator
        self.arg_names = ['x_names', 'y_names']
        self.kwarg_names = ['estimator']
        self.class_name = 'Model'

    #TODO: Rename parameters x_names, y_names, and x_features with trailing
    # underscore as per scikit-learn convention

    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        return {"x_names": self.x_names, "y_names": self.y_names,
                "estimator": self.estimator}

    def set_params(self, **parameters):
        """Set the parameters of this estimator."""
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    @property
    def x_names(self):
        return self._x_names

    @x_names.setter
    def x_names(self, value):
        self._x_names = value
        self._x_labels = [f'x{i}' for i in range(len(value))]
        self._x_rename_map = dict(zip(value, self._x_labels))

    @property
    def y_names(self):
        """The names of the output variables in y.
        """
        return self._y_names

    @y_names.setter
    def y_names(self, value):
        self._y_names = value
        self._y_labels = [f'y{i}' for i in range(len(value))]
        self._y_rename_map = dict(zip(value, self._y_labels))

    def fit(self, X, y):
        """Fit linear model to features.
        """
        # TODO: Allow arrays too?
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.DataFrame)

        # Select required data
        X = X[self.x_names]
        y = y[self.y_names]

        # Note: Can't use scikit learn Pipeline becuase it doesn't
        # support transformations to outputs (y)
        if self.input_scaler:
            X = self.input_scaler.fit_transform(X.values.astype(np.float))
        if self.output_scaler:
            y = self.output_scaler.fit_transform(y.values.astype(np.float))
        self.estimator.fit(X, y)

    def predict(self, X):
        """Predict using the model.  If X is a DataFrame with named
        columns (x-data points in rows) then the predicted y-values
        are returned as a DataFrame with named columns.

        If X is a dict, list, Series, or 1-d array, it is assumed 
        to represent one data point and the y-prediction is 
        returned as a dictionary.

        Example 1:
        >>> x_names = ['x_1', 'x_2']
        >>> y_names = ['dx_1/dt', 'dx_2/dt']
        >>> model = Model(x_names, y_names)
        >>> model.fit(X, y)  # X, y are dataframes with named columns
        >>> model.predict(X)
                dx_1/dt  dx_2/dt
        0 -2.664535e-15      2.0
        1  1.000000e+00      2.0
        2  2.000000e+00      5.0
        3  3.000000e+00      5.0
        4  4.000000e+00     10.0
        5  6.000000e+00     10.0

        Example 2:
        >>> x = {'x_1': 2, 'x_2': 2}
        >>> model.predict(x)
        {'dx_1/dt': 2.0, 'dx_2/dt': 5.0}
        """

        if isinstance(X, (dict, pd.Series)):
            # This is 50 times faster than when passing one data
            # point as a dataframe.
            # TODO: This is a big speed improvement but complex...
            if isinstance(X, dict):
                x = np.array(list(X.values()), dtype=np.float).reshape(1, -1)
            else:
                x = X.values.reshape(1, -1)
            if self.input_scaler:
                # Re-scale inputs
                x = self.input_scaler.transform(x)
            y_values = self.estimator.predict(x).reshape(-1)
            if self.output_scaler:
                # Re-scale outputs
                y_values = self.output_scaler.inverse_transform(y_values)
            return dict(zip(self.y_names, y_values))
        elif isinstance(X, pd.DataFrame):
            # Re-label inputs as 'x0', 'x1', etc.
            index = X.index
            X = X[self.x_names].rename(columns=self._x_rename_map)
            if self.input_scaler:
                # Apply the scaling to input data
                X = self.input_scaler.transform(X.values.astype(np.float))
            y = self.estimator.predict(X)
            if self.output_scaler:
                # Reverse the scaling on output predictions
                y = self.output_scaler.inverse_transform(y)
            return pd.DataFrame(y, index=index, columns=self.y_names)
        else:
            X = dict(zip(self.x_names, X))
            return self.predict(X)

    def score(self, X, y, sample_weight=None):
        """Returns the coefficient of determination R^2 of the
        prediction.
        """
        y_selected = y[self.y_names]
        y_pred = self.predict(X)
        return r2_score(y_selected, y_pred, sample_weight=sample_weight,
                        multioutput='uniform_average')

    def __repr__(self):
        all_args = [self.__getattribute__(a).__repr__() for a in self.arg_names] + \
                   [f"{a}={self.__getattribute__(a).__repr__()}"
                    for a in self.kwarg_names
                    if self.__getattribute__(a) is not None]
        return f"{self.class_name}({', '.join(all_args)})"


class NonLinearModel(Model):
    """Model interface for running different model-fitting and
    evaluation experiments with models that include non-linear 
    combinations of the features calculated from the input data.
    These can be used to identify linearizations of non-linear 
    models and to identify sparse non-linear models that have
    only a few active terms.

    The benefit of these model classes is that they have an internal
    representation of the features they are designed to use and 
    will only use the data for those features, even if you pass
    other data.

    Example:
    >>> import numpy as np
    >>> import pandas as pd
    >>> # Simulate a non-linear system of ODEs
    >>> f1 = lambda x1, x2: x2 - 2 * x1 + x1**2
    >>> f2 = lambda x1, x2: 1 + x1**2
    >>> # Generate (X, Y) data samples
    >>> X_data = np.array([[1, 1], [1, 2], [2, 2], [2, 3], [3, 1], [3, 3]])
    >>> Y_data = np.array([[f1(*x), f2(*x)] for x in X_data]).astype(float)
    >>> x_names = ['x_1', 'x_2']  # You can use any names
    >>> y_names = ['dx_1/dt', 'dx_2/dt']  # You can use any names
    >>> # Define features to use (more than needed in this example)
    >>> x_features = ['x0', 'x1', 'x0**2', 'x0*x1', 'x1**2']
    >>> model = NonLinearModel(x_names, y_names, x_features=x_features)
    >>> # Prepare Pandas dataframes with correctly-named columns
    >>> X = pd.DataFrame(X_data, columns=x_names)
    >>> Y = pd.DataFrame(Y_data, columns=y_names)
    >>> model.fit(X, Y)
    >>> # Check model parameters
    >>> model.coef_.round(4)
        x0   x1  x0**2  x0*x1  x1**2
    y0 -2.0  1.0    1.0   -0.0   -0.0
    y1  0.0  0.0    1.0   -0.0   -0.0
    >>> model.intercept_.round(4)
    y0   -0.0
    y1    1.0
    >>> model.predict(X).round(4)
    dx_1/dt  dx_2/dt
    0     -0.0      2.0
    1      1.0      2.0
    2      2.0      5.0
    3      3.0      5.0
    4      4.0     10.0
    5      6.0     10.0

    Parameters
    ----------
    x_names : list of strings,
        Names of input variables.
    y_names : list of strings,
        Names of model output variables (predictions).
    estimator : class
        Estimator model to be used to fit the (transformed) features.
        If not specified, a LinearRegression instance used.
    x_features : list of strings, optional
        List of feature expressions.  Features may simply be
        selected inputs ['x0', 'x1', ... etc.] or may include
        expressions of current inputs such as 'x0**2' and 'x1*x2'
        (the number of current inputs is defined by the length
        of self.x_names).
    scale_inputs : boolean, optional, default False
        If True, the regressors (X) will be standardized before
        model-fitting by subtracting the mean and dividing by the
        l2-norm.  See Scikit-Learn documentation for StandardScaler.
    scale_outputs : boolean, optional, default False
        If True, the predictors (Y) will be standardized before
        model-fitting by subtracting the mean and dividing by the
        l2-norm.  See Scikit-Learn documentation for StandardScaler.
    """

    # Functions which can be used in expressions for calculating
    # input features (more can be added).
    functions = {'sin': np.sin, 'cos': np.cos, 'tan': np.tan,
                 'exp': np.exp, 'log': np.log, 'tanh': np.tanh,
                 'sqrt': np.sqrt}

    def __init__(self, x_names, y_names, estimator=None, x_features=None, 
                 scale_inputs=False, scale_outputs=False, *args, **kwargs):
        super().__init__(x_names, y_names, scale_inputs=scale_inputs, 
                         scale_outputs=scale_outputs, *args, **kwargs)
        if x_features is None:
            x_features = self._x_labels
        self.x_features = x_features
        self.input_transformer = self.input_transformer_(self.x_features)
        self.arg_names = ['x_names', 'y_names']
        self.kwarg_names = ['estimator', 'x_features']
        self.class_name = 'NonLinearModel'

    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        return {"x_names": self.x_names, "y_names": self.y_names, 
                "estimator": self.estimator, "x_features": self.x_features}

    @staticmethod
    def input_transformer_(x_features):
        """Initialize the input feature transformer based on a list 
        of feature expressions.  Features may simply be selected 
        inputs ['x0', 'x1', ... etc.] or may include expressions of
        current inputs such as 'x0**2' and 'x1*x2' (the number of 
        current inputs is defined by the length of self.x_names).

        Parameters
        ----------
        x_features : list of strings
            Features and feature-expressions to be used by the model.
        """
        feature_function = partial(feature_dataframe_from_expressions,
                                   expressions=x_features)
        input_transformer = FunctionTransformer(feature_function, validate=False)
        return input_transformer

    @property
    def coef_(self):
        """The coefficients from the linear model (model.coef_)
        as a Pandas Dataframe.  To access the attribute directly use
        model.estimator.coef_ instead.
        """
        return pd.DataFrame(self.estimator.coef_, index=self._y_labels,
                            columns=self.x_features)

    @coef_.setter
    def coef_(self, values):
        self.estimator.coef_[:] = values

    @property
    def intercept_(self):
        """The intercepts from the linear model (model.intercept_)
        as a Pandas Series.  To access the attribute directly use
        model.estimator.intercept_ instead.
        """
        return pd.Series(self.estimator.intercept_, index=self._y_labels)

    @intercept_.setter
    def intercept_(self, values):
        self.estimator.intercept_[:] = values

    @property
    def n_params(self):
        """Get total number of parameters in the model.
        """
        n_params = 0
        attr_names = ['coef_', 'intercept_', 'coefs_', 'intercepts_']
        param_arrays = []
        for attr_name in attr_names:
            try:
                attr = getattr(self.estimator, attr_name)
            except AttributeError:
                continue
            if isinstance(attr, list):
                param_arrays += attr
            else:
                param_arrays.append(attr)
        n_params = sum([(a != 0).sum() for a in param_arrays])

        return n_params

    def fit(self, X, y):
        """Fit linear model to features.
        """
        # TODO: Allow arrays too?
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.DataFrame)

        # Re-label inputs as 'x0', 'x1', etc.
        X = X[self.x_names].rename(columns=self._x_rename_map)
        y = y[self.y_names]

        # Note: Can't use scikit learn Pipeline becuase it doesn't
        # support transformations to outputs (y)
        if self.input_transformer:
            X = self.input_transformer.fit_transform(X)
        if self.input_scaler:
            X = self.input_scaler.fit_transform(X.values.astype(np.float))
        if self.output_scaler:
            y = self.output_scaler.fit_transform(y.values.astype(np.float))
        self.estimator.fit(X, y)

    def predict(self, X):
        """Predict using the model.  If X is a DataFrame with named
        columns (x-data points in rows) then the predicted y-values
        are returned as a DataFrame with named columns.

        If X is a dict, list, Series, or 1-d array, it is assumed 
        to represent one data point and the y-prediction is 
        returned as a dictionary.

        Example 1:
        >>> x_names = ['x_1', 'x_2']
        >>> y_names = ['dx_1/dt', 'dx_2/dt']
        >>> model = Model(x_names, y_names)
        >>> model.fit(X, y)  # X, y are dataframes with named columns
        >>> model.predict(X)
                dx_1/dt  dx_2/dt
        0 -2.664535e-15      2.0
        1  1.000000e+00      2.0
        2  2.000000e+00      5.0
        3  3.000000e+00      5.0
        4  4.000000e+00     10.0
        5  6.000000e+00     10.0

        Example 2:
        >>> x = {'x_1': 2, 'x_2': 2}
        >>> model.predict(x)
        {'dx_1/dt': 2.0, 'dx_2/dt': 5.0}
        """

        if isinstance(X, (dict, pd.Series)):
            # This is 50 times faster than when passing one data
            # point as a dataframe.
            # TODO: This is a big speed improvement but complex...
            # Re-label inputs as 'x0', 'x1', etc.
            x_values = {self._x_rename_map[k]: v for k, v in X.items()}
            ref_dict = {**x_values, **self.functions}
            # Calculate evaluated features
            x = [eval(expr, ref_dict) for expr in self.x_features]
            x = np.array(x, dtype=np.float).reshape(1, -1)
            if self.input_scaler:
                # Re-scale inputs
                x = self.input_scaler.transform(x)
            y_values = self.estimator.predict(x).reshape(-1)
            if self.output_scaler:
                # Re-scale outputs
                y_values = self.output_scaler.inverse_transform(y_values)
            return dict(zip(self.y_names, y_values))
        elif isinstance(X, pd.DataFrame):
            # Re-label inputs as 'x0', 'x1', etc.
            index = X.index
            X = X[self.x_names].rename(columns=self._x_rename_map)
            if self.input_transformer:
                X = self.input_transformer.transform(X)
            if self.input_scaler:
                # Apply the scaling to input data
                X = self.input_scaler.transform(X.values.astype(np.float))
            y = self.estimator.predict(X)
            if self.output_scaler:
                # Reverse the scaling on output predictions
                y = self.output_scaler.inverse_transform(y)
            return pd.DataFrame(y, index=index, columns=self.y_names)
        else:
            X = dict(zip(self.x_names, X))
            return self.predict(X)

    # TODO: Sub-classes for FP model and include dydt functions for trajectory prediction


class SparseNonLinearModel(NonLinearModel):
    """Model interface to identify sparse non-linear models. 

    Example:
    >>> import numpy as np
    >>> import pandas as pd
    >>> # Simulate a non-linear system of ODEs
    >>> f1 = lambda x1, x2: x2 - 2 * x1 + x1**2
    >>> f2 = lambda x1, x2: 1 + x1**2
    >>> # Generate (X, Y) data samples
    >>> X_data = np.array([[1, 1], [1, 2], [2, 2], [2, 3], [3, 1], [3, 3]])
    >>> Y_data = np.array([[f1(*x), f2(*x)] for x in X_data]).astype(float)
    >>> x_names = ['x_1', 'x_2']  # You can use any names
    >>> y_names = ['dx_1/dt', 'dx_2/dt']  # You can use any names
    >>> # You have to specify the order of the polynomial terms
    >>> model = SparseNonLinearModel(x_names, y_names, poly_order=2)
    >>> # Prepare Pandas dataframes with correctly-named columns
    >>> X = pd.DataFrame(X_data, columns=x_names)
    >>> Y = pd.DataFrame(Y_data, columns=y_names)
    >>> model.fit(X, Y, threshold=0.5)
    >>> # Check model parameters
    >>> model.coef_
        x0   x1  x0**2
    y0 -2.0  1.0    1.0
    y1  0.0  0.0    1.0
    >>> model.intercept_
    y0    0.0
    y1    1.0
    >>> model.predict(X)
            dx_1/dt  dx_2/dt
    0  7.771561e-16      2.0
    1  1.000000e+00      2.0
    2  2.000000e+00      5.0
    3  3.000000e+00      5.0
    4  4.000000e+00     10.0
    5  6.000000e+00     10.0

    Parameters
    ----------
    x_names : list of strings,
        Names of input variables.
    y_names : list of strings,
        Names of model output variables (predictions).
    estimator : class
        Estimator model to be used to fit the (transformed) features.
        If not specified, a LinearRegression instance used.
    custom_features : list of strings, optional
        List of feature expressions to include in addition to the
        polynomial features that will be automatically included.
        Features must be valid expresssions of current inputs such as
        'x0**2' and 'x1*x2'.
    poly_order : int
        Order of polynomial terms to include in the feature library.
    threshold : float
        Sparsification parameter.  See SINDy reference.
    scale_inputs : boolean, optional, default False
        If True, the regressors (X) will be standardized before
        model-fitting by subtracting the mean and dividing by the
        l2-norm.  See Scikit-Learn documentation for StandardScaler.
    scale_outputs : boolean, optional, default False
        If True, the predictors (Y) will be standardized before
        model-fitting by subtracting the mean and dividing by the
        l2-norm.  See Scikit-Learn documentation for StandardScaler.
    """

    def __init__(self, x_names, y_names, estimator=None, custom_features=None,
                 poly_order=3, threshold=None, scale_inputs=False, 
                 scale_outputs=False, *args, **kwargs):
        #if scale_inputs is True:
        # TODO: Need to see how this works first
        #    raise NotImplementedError()
        super().__init__(x_names, y_names, estimator=estimator,
                         x_features=None, scale_inputs=scale_inputs,
                         scale_outputs=scale_outputs, *args, **kwargs)
        # In this model the feature generator is re-initialized
        # later when the fit method is called.
        self.custom_features = custom_features
        self.poly_order = poly_order
        self.threshold = threshold
        self.arg_names = ['x_names', 'y_names']
        self.kwarg_names = ['estimator', 'custom_features', 'poly_order', 'threshold']
        self.class_name = 'SparseNonLinearModel'

    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        return {"x_names": self.x_names, "y_names": self.y_names, 
                "estimator": self.estimator, "x_features": self.x_features,
                "custom_features": self.custom_features, "poly_order": self.poly_order,
                "threshold": self.threshold}

    def fit(self, X, y, threshold=None):
        """Find sparse non-linear model from data by iteratively
        removing model coefficients smaller than the threshold value 
        (SINDy method).
        """
        if threshold is None:
            threshold = self.threshold
        else:
            self.threshold = threshold

        # TODO: Allow arrays too?
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.DataFrame)

        # Re-label inputs as 'x0', 'x1', etc.
        X = X[self.x_names].rename(columns=self._x_rename_map)
        y = y[self.y_names]

        # Calculate library of polynomial features
        # TODO: Use sklearn: poly = PolynomialFeatures(poly_order)
        theta = polynomial_features(X.values, self.poly_order)

        n_in = X.shape[1]
        n_out = y.shape[1]
        feature_labels = polynomial_feature_labels(n_in, self.poly_order)

        # Add any custom labels
        if self.custom_features is not None:
            theta_custom_features = \
                feature_array_from_expressions(X, self.custom_features)
            feature_labels += self.custom_features
            theta = np.hstack([theta, theta_custom_features])

        # Identify sparse model
        coef = sparsify_dynamics_lstsq(theta, y.values, threshold)
        coef_df = pd.DataFrame(coef, index=feature_labels, 
                               columns=self.y_names)

        # Only use features with non-zero coefficients
        non_zero_features = (coef_df != 0).any(axis=1)
        if sum(non_zero_features) == 1:
            # Make sure we don't eliminate all features since
            # most estimators can't handle an empty coef matrix
            # (leave one feature with coefficients set to zero)
            non_zero_features['x0'] = True
        non_zero_coefs = coef_df.loc[non_zero_features]

        if '1' in non_zero_coefs.index:
            intercepts = non_zero_coefs.loc['1'].values
            non_zero_coefs = non_zero_coefs.drop('1')
        else:
            intercepts = np.zeros(n_out)
        
        # Set linear estimator coefficients to values found from
        # sparse identification
        coefficients = non_zero_coefs.T.values
        self.x_features = non_zero_coefs.index.tolist()
        self.estimator.intercept_ = intercepts
        self.estimator.coef_ = coefficients

        if self.input_transformer:
            # Re-initialize input transformer
            self.input_transformer = self.input_transformer_(self.x_features)
            X = self.input_transformer.fit_transform(X) 
        # Scale all inputs and outputs last
        if self.input_scaler:
            X = self.input_scaler.fit_transform(X.values.astype(np.float))
        if self.output_scaler:
            y = self.output_scaler.fit_transform(y.values.astype(np.float))

        # TODO: Is there a more efficient way to do this?
        #TODO: Ultimately, it would be nice to overload the LinearModel's
        # fit method

    def __repr__(self):
        all_args = [self.__getattribute__(a).__repr__() for a in self.arg_names] + \
                   [f"{a}={self.__getattribute__(a).__repr__()}"
                    for a in self.kwarg_names
                    if self.__getattribute__(a) is not None]
        return f"{self.class_name}({', '.join(all_args)})"


class SVRMultipleOutputs:
    #TODO: Could sklearn.multioutput.MultiOutputRegressor be used instead?
    """Support Vector Regression model class that is mostly
    compatible with sklearn.svm.SVR but supports the following:

    1. More than one output variable

    Example 1:
    >>> X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    >>> Y = np.array([[0., 2.], [1., 2.], [2., 5.], [3., 5.]])  # shape(4, 2)
    >>> model = SVRMultipleOutputs(gamma='scale', kernel='linear')
    >>> model.fit(X, Y)
    >>> model.predict(X)
    array([[0.10027962, 2.1       ],
           [1.09944076, 2.82      ],
           [1.90111848, 4.18      ],
           [2.90027962, 4.9       ]])
    >>> model.coefs_
    array([[0.80167772, 0.99916114],
           [1.36      , 0.72      ]])
    >>> model.intercepts_
    array([[-1.70055924],
           [ 0.02      ]])
    >>> model.score(X, Y)
    0.9202111601972919

    2. Uses only a random sample of the X, y data to train when
    fitting to large datasets (> max_data), it which avoids
    excessive computation (unless seed is set to None, this is
    the same sample every time).

    Example 2:
    >>> X = np.random.randn(100000, 4)  # Large data set
    >>> Y = (np.array(range(1, 5))*X).sum(axis=1).reshape(-1, 1)
    >>> model = SVRMultipleOutputs(max_data=1000, gamma='scale')
    >>> %timeit model.fit(X, Y)  # Takes ~36.9 ms
    >>> model = SVRMultipleOutputs(max_data=10000, gamma='scale')
    >>> model.fit(X, Y)  # Takes ~810 ms
    """
    def __init__(self, max_data=1000, seed=1, *args, **kwargs):
        self.max_data = max_data
        self.seed = seed
        self._rs = np.random.RandomState(seed)
        self.args = args
        self.kwargs = kwargs
        self.models = None  # Store one model for each output

    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        return {'max_data': self.max_data, 'seed': self.seed}

    @property
    def coef_(self):
        """Weights assigned to the features. This is only available
        in the case of a linear kernel.
        """
        coef_ = [model.coef_ for model in self.models]
        return np.vstack(coef_)

    @property
    def intercept_(self):
        """Constants in decision function.
        """
        intercepts_ = [model.intercept_ for model in self.models]
        return np.hstack(intercepts_)

    @property
    def n_params(self):
        """Get total number of parameters in the model.
        """
        return self.coef_.size + self.intercept_.size

    def fit(self, X, y, sample_weight=None):
        # If too much data just fit to a sample
        if self.max_data is not None and X.shape[0] > self.max_data:
            self._rs.seed(self.seed)
            sample = self._rs.choice(X.shape[0], size=self.max_data,
                                     replace=False)
            if isinstance(X, pd.DataFrame):
                X = X.iloc[sample, :].values
            else:
                X = X[sample, :]
            if isinstance(y, pd.DataFrame):
                y = y.iloc[sample, :].values
            else:
                y = y[sample, :]
        self.models = []
        for i in range(y.shape[1]):
            model = SVR(*self.args, **self.kwargs)
            model.fit(X, y[:, i], sample_weight)
            self.models.append(model)

    def predict(self, X):
        predictions = []
        for model in self.models:
            y_pred = model.predict(X)
            predictions.append(y_pred)
        return np.array(predictions).transpose()

    def score(self, X, y, sample_weight=None):
        y_pred = self.predict(X)
        return r2_score(y, y_pred, sample_weight=sample_weight,
                        multioutput='uniform_average')

    def __repr__(self):
        all_args = [f"{a}={self.__getattribute__(a).__repr__()}"
                    for a in ['max_data', 'seed']]
        return f"SVRMultipleOutputs({', '.join(all_args)})"


class LinearPredictionModel(LinearRegression):
    """This model is for prediction only.  It has no fit method.
    You can initialize it with fixed values for coefficients 
    and intercepts.
    
    Parameters
    ----------
    coef, intercept : arrays
        See attribute descriptions below.

    Attributes
    ----------
    coef_ : array of shape (n_features, ) or (n_targets, n_features)
        Coefficients of the linear model.  If there are multiple targets
        (y 2D), this is a 2D array of shape (n_targets, n_features), 
        whereas if there is only one target, this is a 1D array of 
        length n_features.
    intercept_ : float or array of shape of (n_targets,)
        Independent term in the linear model.
    """

    def __init__(self, coef=None, intercept=None):
        if coef is not None:
            coef = np.array(coef)
            if intercept is None:
                intercept = np.zeros(coef.shape[0])
            else:
                intercept = np.array(intercept)
            assert coef.shape[0] == intercept.shape[0]
        else:
            if intercept is not None:
                raise ValueError("Provide coef only or both coef and intercept")
        self.intercept_ = intercept
        self.coef_ = coef

    def fit(self, X, y):
        """This model does not have a fit method."""
        raise NotImplementedError("model is only for prediction")

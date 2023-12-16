#TODO: scikit-learn has now implement feature names and checking since I wrote this
#      so needs a thorough review and possibly re-write.
#  E.g. 
# UserWarning: X has feature names, but LinearRegression was fitted without feature names
# 
# And
# 
# ValueError: The feature names should match those that were passed during fit.
# Feature names unseen at fit time:
# - x0
# - x1
# Feature names seen at fit time, yet now missing:
# - x_0
# - x_1
#
# Also:
#  DeprecationWarning: is_sparse is deprecated and will be removed in a future version. 
# From sklearn/utils/validation.py and pandas/core/algorithms.py but not sure which module is calling it.
#

import unittest
import numpy as np
import pandas as pd
from numpy.testing import assert_array_almost_equal
from pandas.testing import assert_index_equal, assert_series_equal, \
                           assert_frame_equal
from numpy.testing import assert_array_equal, assert_array_almost_equal
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Classes/functions to test
from dynopt.models.models import Model, NonLinearModel, SparseNonLinearModel, \
                                 SVRMultipleOutputs, LinearPredictionModel


class ModelTests(unittest.TestCase):

    def test_Model_linear(self):

        # Simple test: y = 1 * x_0 + 2 * x_1 + 3
        X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
        y = np.dot(X, np.array([1, 2])) + 3
        X_test = np.array([
            [3, 5],
            [4, 2]
        ])

        # Make sure it replicates Scikit-Learn's LinearRegression model
        reg = LinearRegression().fit(X, y)
        test_coef_ = reg.coef_
        test_intercept_ = reg.intercept_
        test_score = reg.score(X, y)
        test_prediction = reg.predict(X_test)

        # Initialize model
        x_names = ['x_0', 'x_1']
        y_names = ['y']
        X = pd.DataFrame(X, columns=x_names)
        X_test = pd.DataFrame(X_test, columns=x_names)
        y = pd.DataFrame(y, columns=y_names)
        model = Model(x_names, y_names)
        self.assertTrue(str(model).startswith("Model"))
        self.assertIsInstance(model.estimator, LinearRegression)
        self.assertEqual(model._x_labels, ['x0', 'x1'])
        self.assertEqual(model._x_rename_map, {'x_0': 'x0', 'x_1': 'x1'})
        self.assertEqual(model._y_labels, ['y0'])
        self.assertEqual(model._y_rename_map, {'y': 'y0'})
        params = model.get_params()
        self.assertEqual(params, {"x_names": x_names, "y_names": y_names,
                                  "estimator": model.estimator})
        self.assertIsInstance(model.estimator, LinearRegression)

        # Fit model to data
        model.fit(X, y)
        assert_array_equal(model.estimator.coef_[0], test_coef_)
        assert_array_equal(model.estimator.intercept_[0], test_intercept_)

        # Test R^2 calculation
        score = model.score(X, y)
        self.assertEqual(score, test_score)

        # Test predictions
        prediction = model.predict(X_test)
        test_prediction = pd.DataFrame(test_prediction.reshape(-1, 1), 
                                       columns=['y0'])
        assert_array_equal(prediction, test_prediction)

        # Test single predictions
        for i in X.index:
            x = {k: v for k, v in zip(model.x_names, X.loc[i].values)}
            y_pred = model.predict(x)
            assert_array_equal(np.array(np.array(list(y_pred.values()))), y.loc[i].values)

    def test_NonLinearModel_linear(self):

        # Simple test: y = 1 * x_0 + 2 * x_1 + 3
        X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
        y = np.dot(X, np.array([1, 2])) + 3
        X_test = np.array([
            [3, 5],
            [4, 2]
        ])

        # Make sure it replicates Scikit-Learn's LinearRegression model
        reg = LinearRegression().fit(X, y)
        test_coef_ = reg.coef_
        test_intercept_ = reg.intercept_
        test_score = reg.score(X, y)
        test_prediction = reg.predict(X_test)

        # Initialize model
        x_names = ['x_0', 'x_1']
        y_names = ['y']
        X = pd.DataFrame(X, columns=x_names)
        X_test = pd.DataFrame(X_test, columns=x_names)
        y = pd.DataFrame(y, columns=y_names)
        model = NonLinearModel(x_names, y_names)
        self.assertTrue(str(model).startswith("NonLinearModel"))
        self.assertIsInstance(model.estimator, LinearRegression)
        self.assertTrue(model.input_transformer is not None)
        self.assertEqual(model._x_labels, ['x0', 'x1'])
        self.assertEqual(model._x_rename_map, {'x_0': 'x0', 'x_1': 'x1'})
        self.assertEqual(model.x_features, ['x0', 'x1'])  # No non-linear features
        self.assertEqual(model._y_labels, ['y0'])
        self.assertEqual(model._y_rename_map, {'y': 'y0'})
        params = model.get_params()
        self.assertEqual(params, {"x_names": x_names, "y_names": y_names,
                                  "estimator": model.estimator,
                                  "x_features": ['x0', 'x1']})
        self.assertIsInstance(model.estimator, LinearRegression)

        # Fit model to data
        model.fit(X, y)
        assert_array_equal(model.estimator.coef_[0], test_coef_)
        assert_array_equal(model.estimator.intercept_[0], test_intercept_)
        test_coef_ = pd.DataFrame(test_coef_.reshape(1, -1), 
                                  columns=['x0', 'x1'], index=['y0'])
        assert_frame_equal(model.coef_, test_coef_)
        test_intercept_ = pd.Series(test_intercept_, index=['y0'])
        assert_series_equal(model.intercept_, test_intercept_)

        # Test R^2 calculation
        score = model.score(X, y)
        self.assertEqual(score, test_score)

        # Test parameter count
        self.assertEqual(model.n_params, 3)

        # Test predictions
        prediction = model.predict(X_test)
        test_prediction = pd.DataFrame(test_prediction.reshape(-1, 1), 
                                       columns=['y0'])
        assert_array_almost_equal(prediction, test_prediction)

        # Test single predictions
        for i in X.index:
            x = {k: v for k, v in zip(model.x_names, X.loc[i].values)}
            y_pred = model.predict(x)
            assert_array_almost_equal(np.array(list(y_pred.values())), y.loc[i].values)

    def test_NonlinearModel_nonlinear(self):

        # Simulate a non-linear system of ODEs:
        # dx1_dt = x2 - 2*x1 + x2**2
        # dx2_dt = 1 + x1**2
        f1 = lambda x1, x2: x2 - 2 * x1 + x1**2
        f2 = lambda x1, x2: 1 + x1**2
 
        # Input data samples
        X = np.array([[1, 1], [1, 2], [2, 2], [2, 3], [3, 3]])

        # Target values (LHS of above eq.ns)
        Y = np.array([[f1(*x), f2(*x)] for x in X]).astype(float)

        # Initialize model
        x_names = ['x_1', 'x_2']  # You can use any name
        y_names = ['dx_1/dt', 'dx_2/dt']
        x_features = ['x0', 'x1', 'x0**2']
        model = NonLinearModel(x_names, y_names, x_features=x_features)
        self.assertTrue(str(model).startswith("NonLinearModel"))

        # Fit model to data
        X = pd.DataFrame(X, columns=x_names)
        Y = pd.DataFrame(Y, columns=y_names)
        model.fit(X, Y)

        test_coef_ = pd.DataFrame(
            {'x0': [-2, 0], 'x1': [1, 0], 'x0**2': [1, 1]}, 
            index=['y0', 'y1']
        ).astype(float)
        assert_frame_equal(model.coef_, test_coef_)

        test_intercept_ = pd.Series({'y0': 0., 'y1': 1.})
        assert_series_equal(model.intercept_, test_intercept_)

        # Test predictions
        assert_frame_equal(model.predict(X), Y)

        # Test R^2 calculation
        score = model.score(X, Y)
        self.assertEqual(score, 1.0)

        # Test parameter count
        self.assertEqual(model.n_params, 8)

        # Test single-point predictions with dicts, series, etc.
        x2 = X.loc[2].to_dict()
        y2 = {'dx_1/dt': 2.0, 'dx_2/dt': 5.0}  # Y.loc[2].to_dict()
        assert_series_equal(pd.Series(model.predict(x2)), pd.Series(y2), atol=1e-15)

        x2 = X.loc[2]  # pd.Series
        assert_series_equal(pd.Series(model.predict(x2)), pd.Series(y2), atol=1e-15)

        x2 = X.loc[2].values  # np.ndarray
        assert_series_equal(pd.Series(model.predict(x2)), pd.Series(y2), atol=1e-15)

        x2 = X.loc[2].tolist()  # list
        assert_series_equal(pd.Series(model.predict(x2)), pd.Series(y2), atol=1e-15)

    def test_SparseNonLinearModel(self):

        # Simulate a non-linear system of ODEs:
        # dx1_dt = x2 - 2*x1 + x2**2
        # dx2_dt = 1 + x1**2
        f1 = lambda x1, x2: x2 - 2 * x1 + x1**2
        f2 = lambda x1, x2: 1 + x1**2
 
        # Input data samples
        X = np.array([[1, 1], [1, 2], [2, 2], [2, 3], [3, 3]])

        # Target values (LHS of above eq.ns)
        Y = np.array([[f1(*x), f2(*x)] for x in X]).astype(float)

        # Initialize model
        x_names = ['x_1', 'x_2']  # You can use any name
        y_names = ['dx_1/dt', 'dx_2/dt']
        poly_order = 2
        model = SparseNonLinearModel(x_names, y_names, poly_order=poly_order)
        self.assertTrue(str(model).startswith("SparseNonLinearModel"))

        # Fit model to data
        X = pd.DataFrame(X, columns=x_names)
        Y = pd.DataFrame(Y, columns=y_names)
        threshold = 0.5
        model.fit(X, Y, threshold=threshold)

        test_x_features = ['x0', 'x1', 'x0**2']
        self.assertEqual(model.x_features, test_x_features)
        params = model.get_params()
        test_params = {
            "x_names": x_names, "y_names": y_names,
            "estimator": model.estimator, "x_features": test_x_features,
            "custom_features": None, "poly_order": poly_order, "threshold": threshold,
        }
        self.assertEqual(params, test_params)
        self.assertIsInstance(model.estimator, LinearRegression)

        test_coef_ = pd.DataFrame(
            {'x0': [-2, 0], 'x1': [1, 0], 'x0**2': [1, 1]}, 
            index=['y0', 'y1']
        ).astype(float)
        assert_frame_equal(model.coef_, test_coef_)

        test_intercept_1 = pd.Series({'y0': 0., 'y1': 1.})  # Both equivalent
        test_intercept_2 = pd.Series({'y0': 1., 'y1': 0.})

        self.assertTrue(
            np.allclose(model.intercept_, test_intercept_1)
            or np.allclose(model.intercept_, test_intercept_2)
        )
        self.assertEqual(set(test_intercept_1.index), set(model.intercept_.index))

        # Test predictions
        assert_frame_equal(model.predict(X), Y)

        # Test R^2 calculation
        score = model.score(X, Y)
        self.assertEqual(score, 1.0)
        self.assertEqual(model.n_params, 5)

        # Test single-point predictions with dicts, series, etc.
        x2 = X.loc[2].to_dict()
        y2 = {'dx_1/dt': 2.0, 'dx_2/dt': 5.0}  # Y.loc[2].to_dict()
        assert_series_equal(pd.Series(model.predict(x2)), pd.Series(y2))
        
        x2 = X.loc[2]  # pd.Series
        assert_series_equal(pd.Series(model.predict(x2)), pd.Series(y2))

        x2 = X.loc[2].values  # np.ndarray
        assert_series_equal(pd.Series(model.predict(x2)), pd.Series(y2))

        x2 = X.loc[2].tolist()  # list
        assert_series_equal(pd.Series(model.predict(x2)), pd.Series(y2))

        # Test setting parameters
        new_params = {"poly_order": 3, "threshold": 0.1}
        model.set_params(**new_params)
        params = model.get_params()
        self.assertEqual(params, {'x_names': x_names, 'y_names': y_names,
                                  'estimator': model.estimator,
                                  'x_features': test_x_features,
                                  'custom_features': None,
                                  'poly_order': new_params['poly_order'], 
                                  'threshold': new_params['threshold']})
        model.fit(X, Y)

    def test_SparseNonLinearModel_with_custom_features(self):

        # The following data points were randomly sampled
        # from a time-series generated by simulating a 
        # simple pendulum with the following equations
        # Pendulum system dynamics:
        # y[0] = dXdt[0] = x[1]
        # y[1] = dXdt[1] = -(b/m) * x[1] + (g/L) * np.sin(x[0]) + u(t) / L
        # with m=1, L=2, g=-10, b=1 and X0 = [pi, 0]

        data = np.array([
            [-0.31337, -0.35066, -1.95775],
            [ 0.43377,  0.1971 ,  0.12873],
            [ 0.02058, -0.97899, -0.13313],
            [-0.64254, -2.22694, -0.583  ],
            [ 0.13985, -0.35462, -0.13526],
            [ 0.29423,  0.4599 ,  1.98026],
            [ 0.29177, -0.34668,  0.43937],
            [-0.17303, -0.78595, -0.89596],
            [ 0.3605 , -0.48347, -1.54148],
            [ 0.33364, -0.20928,  0.92452]
        ])
        x_names = ['theta', 'theta_dot', 'u']
        X_u = pd.DataFrame(data, columns=x_names)

        data = np.array([
            [-0.35066,  0.91312],
            [ 0.1971 , -2.23421],
            [-0.97899,  0.80954],
            [-2.22694,  4.93158],
            [-0.35462, -0.40998],
            [ 0.4599 , -0.91979],
            [-0.34668, -0.87185],
            [-0.78595,  1.19881],
            [-0.48347, -2.05099],
            [-0.20928, -0.9659 ]
        ])
        y_names = ['theta_dot', 'theta_ddot']
        dXdt = pd.DataFrame(data, columns=y_names)

        custom_features = ['sin(x0)', 'cos(x0)', 'sin(x0)**2', 
                        'cos(x0)**2', 'sin(x0)*cos(x0)']

        model = SparseNonLinearModel(x_names, y_names, 
                                     custom_features=custom_features,
                                     poly_order=2)

        model.fit(X_u, dXdt, threshold=0.2)
        self.assertEqual(model.n_params, 4)

        test_x_features = ['x1', 'x2', 'sin(x0)']
        self.assertEqual(model.x_features, test_x_features)
        self.assertEqual(model.custom_features, custom_features)

        coef_test = pd.DataFrame(
            {
                'x1': [1.0, -1.0],
                'x2': [0.0, 0.5],
                'sin(x0)': [0.0, -5.0]
            }, 
            index=['y0', 'y1']
        )
        assert_frame_equal(model.coef_, coef_test)

        intercept_test = pd.Series({'y0': 0., 'y1': 0.})
        assert_series_equal(model.intercept_, intercept_test)

        # Test predictions
        assert_frame_equal(model.predict(X_u), dXdt, atol=0.0001)

        # Test R^2 calculation
        score = model.score(X_u, dXdt)
        self.assertEqual(score.round(3), 1.0)

        # Test single-point predictions with dicts, series, etc.
        x2 = X_u.loc[2].to_dict()
        y2 = {'theta_dot': -0.97899, 'theta_ddot': 0.80954}  # Y.loc[2].to_dict()
        assert_series_equal(pd.Series(model.predict(x2)), 
                            pd.Series(y2), atol=0.0001)
        
        x2 = X_u.loc[2]  # pd.Series
        assert_series_equal(pd.Series(model.predict(x2)), 
                            pd.Series(y2), atol=0.0001)

        x2 = X_u.loc[2].values  # np.ndarray
        assert_series_equal(pd.Series(model.predict(x2)), 
                            pd.Series(y2), atol=0.0001)

        x2 = X_u.loc[2].tolist()  # list
        assert_series_equal(pd.Series(model.predict(x2)), 
                            pd.Series(y2), atol=0.0001)

    def test_NonlinearModel_with_scaling(self):
        # Simple test: y = 1 * x_0 + 200 * x_1 + 3
        X = np.array([[100, 1], [100, 2], [200, 2], [200, 3]])
        y = np.dot(X, np.array([1, 200])) + 300
        X_test = np.array([[300, 5], [400, 2] ])
        y_test = np.dot(X_test, np.array([1, 200])) + 300

        # Make sure it replicates Scikit-Learn's LinearRegression model
        reg = LinearRegression().fit(X, y)
        test_coef_ = reg.coef_
        test_intercept_ = reg.intercept_
        test_prediction = reg.predict(X_test)

        # Initialize model
        x_names = ['x_0', 'x_1']
        y_names = ['y']
        X = pd.DataFrame(X, columns=x_names)
        X_test = pd.DataFrame(X_test, columns=x_names)
        y = pd.DataFrame(y, columns=y_names)
        model = NonLinearModel(x_names, y_names)
        model_scaled1 = NonLinearModel(x_names, y_names, scale_inputs=True)
        model_scaled2 = NonLinearModel(x_names, y_names, scale_inputs=True,
                                       scale_outputs=True)
        self.assertIsInstance(model.estimator, LinearRegression)
        self.assertEqual(model._x_labels, ['x0', 'x1'])
        self.assertEqual(model._x_rename_map, {'x_0': 'x0', 'x_1': 'x1'})
        self.assertEqual(model.x_features, ['x0', 'x1'])  # No non-linear features
        self.assertEqual(model._y_labels, ['y0'])
        self.assertEqual(model._y_rename_map, {'y': 'y0'})
        self.assertTrue(model.input_transformer is not None)
        self.assertTrue(model.input_scaler is None)
        self.assertTrue(model_scaled1.input_scaler is not None)
        self.assertTrue(model_scaled2.input_scaler is not None)
        self.assertTrue(model_scaled2.output_scaler is not None)

        # Fit models to data
        model.fit(X, y)
        model_scaled1.fit(X, y)
        model_scaled2.fit(X, y)

        # Test input scaler attributes
        scaler = model_scaled1.input_scaler
        assert_array_almost_equal(scaler.scale_, X.values.std(axis=0))
        assert_array_almost_equal(scaler.mean_, X.values.mean(axis=0))
        scaler = model_scaled2.input_scaler
        assert_array_almost_equal(scaler.scale_, X.values.std(axis=0))
        assert_array_almost_equal(scaler.mean_, X.values.mean(axis=0))

        # Test output scaler attributes
        scaler = model_scaled2.output_scaler
        assert_array_almost_equal(scaler.scale_, y.values.std(axis=0))
        assert_array_almost_equal(scaler.mean_, y.values.mean(axis=0))

        # Test model parameters
        assert_array_almost_equal(model.estimator.coef_[0], test_coef_)
        self.assertAlmostEqual(model.estimator.intercept_[0], test_intercept_)
        assert_array_almost_equal(model_scaled1.estimator.coef_[0], [50., 141.42135624])
        self.assertAlmostEqual(model_scaled1.estimator.intercept_[0], y.values.mean())
        self.assertFalse(np.isclose(model_scaled1.estimator.coef_[0], test_coef_).all())
        self.assertFalse(np.isclose(model_scaled1.estimator.intercept_[0], 
                                    test_intercept_).all())
        self.assertAlmostEqual(model_scaled2.estimator.intercept_[0], 0)
        assert_array_almost_equal(model_scaled2.estimator.coef_[0], [0.27735, 0.784465])
        self.assertFalse(np.isclose(model_scaled1.estimator.coef_[0], 
                                    model_scaled2.estimator.coef_[0]).all())
        self.assertFalse(np.isclose(model_scaled1.estimator.intercept_[0], 
                                    model_scaled2.estimator.intercept_[0]).all())

        # Test predictions
        test_prediction = reg.predict(X_test)
        prediction = model.predict(X_test)['y'].values
        assert_array_almost_equal(prediction, y_test)
        prediction = model_scaled1.predict(X_test)['y'].values
        assert_array_almost_equal(prediction, y_test)
        prediction = model_scaled2.predict(X_test)['y'].values
        assert_array_almost_equal(prediction, y_test)

        # Test R-squared estimates
        test_score = reg.score(X, y)
        self.assertEqual(model.score(X, y), test_score)
        self.assertEqual(model_scaled1.score(X, y), test_score)
        self.assertEqual(model_scaled2.score(X, y), test_score)

        # Test single predictions
        models = [model, model_scaled1, model_scaled2]
        for i in X_test.index:
            for model in models:
                x = {k: v for k, v in zip(model.x_names, X_test.loc[i].values)}
                y_pred = model.predict(x)
                self.assertAlmostEqual(y_pred['y'], y_test[i])

    def test_LinearPredictionModel(self):

        coefficients = np.array([
            [-1.38e-03,  4.90e-04,  1.87e-03, -9.00e-05, -5.00e-05],
            [ 4.29e-03, -1.71e-03, -1.70e-04,  9.10e-04, -1.00e-04]
        ])
        intercepts = np.array([ 0.04856, -0.0038 ])

        model = LinearPredictionModel(coef_=coefficients, intercept_=intercepts)
        self.assertTrue(str(model).startswith("LinearPredictionModel"))
        # TODO: on above line
        # File "/Users/billtubbs/anaconda3/envs/torch/lib/python3.10/site-packages/sklearn/base.py", line 170, in get_params
        #   value = getattr(self, key)
        # AttributeError: 'LinearPredictionModel' object has no attribute 'coef'
        param_names = list(model.get_params().keys())
        self.assertEqual(param_names, ['coef_', 'intercept_'])
        model.set_params(coef_=coefficients*2, intercept_=intercepts+1)
        #TODO: This method does not work (sets model.coef, model.intercept)
        #assert_array_equal(model.coef_, coefficients*2)
        #assert_array_equal(model.intercept_, intercepts+1)

        X = np.random.randn(10,5)
        Y = np.random.randn(10,2)
        self.assertRaises(NotImplementedError, model.fit, X, Y)

    def test_SVRMultipleOutputs(self):

        X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
        Y = np.array([[0., 2.], [1., 2.], [2., 5.], [3., 5.]])  # shape(4, 2)
        model = SVRMultipleOutputs(gamma='scale', kernel='linear')
        self.assertTrue(str(model).startswith("SVRMultipleOutputs"))

        model.fit(X, Y)
        y_pred = model.predict(X)
        true_values = np.array([
            [0.10027962, 2.1 ],
            [1.09944076, 2.82],
            [1.90111848, 4.18],
            [2.90027962, 4.9]
        ])
        self.assertTrue(np.isclose(y_pred, true_values).all())
        
        coef_ = model.coef_
        true_values = np.array([
            [0.80167772, 0.99916114],
            [1.36, 0.72]
        ])
        self.assertTrue(np.isclose(coef_, true_values).all())

        intercept_ = model.intercept_
        true_values = np.array([-1.70055924, 0.02      ])
        self.assertTrue(np.isclose(intercept_, true_values).all())
        self.assertEqual(model.score(X, Y), 0.9202111601972919)
        self.assertEqual(model.n_params, 6)

        n = 1000
        X = np.random.randn(n, 4)
        Y = (np.array(range(1, 5))*X).sum(axis=1).reshape(-1, 1)

        # Make sure random sample is same if fit is repeated
        model = SVRMultipleOutputs(max_data=10, gamma='scale')
        model.fit(X, Y)
        score1 = model.score(X, Y)
        model.fit(X, Y)
        score2 = model.score(X, Y)
        self.assertEqual(score1, score2)

        # Make sure random sample is not the same after init with seed=None
        model = SVRMultipleOutputs(max_data=10, gamma='scale', seed=None)
        model.fit(X, Y)
        score3 = model.score(X, Y)
        self.assertNotEqual(score1, score3)

        # Make sure random sample is the same if seed is set
        model = SVRMultipleOutputs(max_data=10, gamma='scale', seed=123)
        model.fit(X, Y)
        score1 = model.score(X, Y)
        model = SVRMultipleOutputs(max_data=10, gamma='scale', seed=123)
        model.fit(X, Y)
        score2 = model.score(X, Y)
        self.assertEqual(score1, score2)

        # Make sure fit is similar for different sample sizes
        score3 = model.score(X, Y)
        model = SVRMultipleOutputs(max_data=None, gamma='scale', 
                                   kernel='linear', seed=123)
        model.fit(X, Y)
        coef_1 = model.coef_
        model = SVRMultipleOutputs(max_data=n * 9//10, gamma='scale', 
                                   kernel='linear', seed=123)
        model.fit(X, Y)
        coef_2 = model.coef_
        # TODO: These are not very close.  Investigate
        assert_array_almost_equal(coef_1, coef_2, decimal=1)

        self.assertEqual(model.n_params, 5)


if __name__ == '__main__':
    unittest.main()

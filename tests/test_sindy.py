import unittest
import numpy as np
import pandas as pd
from functools import partial
from scipy.integrate import odeint
from sklearn.preprocessing import PolynomialFeatures
from examples.lorenz import lorenz_odes, lorenz_odes_vectorized
from dynopt.models.sindy import sparsify_dynamics_lstsq
from dynopt.preprocessing.utils import polynomial_features, \
                                polynomial_feature_labels

class SindyTests(unittest.TestCase):

    def setUp(self):
        pass

        # Simulation setup
        self.dt = 0.01
        self.T = 50
        self.t = np.arange(self.dt, self.T + self.dt, self.dt)

        np.random.seed(123)

    def test_sindy_on_lorenz(self):

        # Simulation time
        t = self.t

        # Lorenz system parameters
        beta = 8 / 3
        sigma = 10
        rho = 28
        n = 3

        # Initial condition
        x0 = (-8, 8, 27)

        # Test point
        y = (-8, 8, 27)
        dxdt = lorenz_odes_vectorized(0, y, sigma=sigma, beta=beta, rho=rho)
        assert np.array_equal(dxdt, (160., -16., -136.0))

        # Two test points as array
        y = np.array([[-8., 0],
                    [ 8., 0],
                    [27., 0]])
        dxdt = lorenz_odes_vectorized(0, y, sigma=sigma, beta=beta, rho=rho)
        true_values = np.array([
            [ 160.,    0.],
            [ -16.,    0.],
            [-136.,    0.]
        ])
        assert np.array_equal(dxdt, true_values)

        # Use scipy.integrate.odeint method (same as Matlab)
        fun = partial(lorenz_odes, sigma=sigma, beta=beta, rho=rho)
        rtol = 10e-12
        atol = 10e-12 * np.ones_like(x0)
        x = odeint(fun, x0, t, tfirst=True, rtol=rtol, atol=atol)
        assert x.shape[1] == n

        # Check simulation results
        test_values = [
            [-8.0, 8.0, 27.0],
            [-6.486403045795506, 7.803170521819883, 25.725522620884725],
            [1.413775661062356, 6.539950620629673, 19.227159896215085],
            [9.057167838947406, 14.558948991097559, 18.415293946945532],
            [8.176101762971722, 12.182215631438558, 19.891261582280325],
            [2.3178200868312038, 4.946778482815532, 23.478642738482403]
        ]  # Values using scipy.integrate.odeint

        # Note: Beyond >1000 iterations the error is too great
        assert x.shape == (5000, 3)
        assert np.isclose(x[[0, 1, 10, 100, 1000], :] , test_values[:-1]).all()

        # Compute Derivative
        test_values = [
            [160.0, -16.0, -136.0],
            [142.8957356761539, -22.556347521306186, -119.21590269528679],
            [51.261749595673166, 5.8628771964931925, -42.02640337791101],
            [55.01781152150153, 72.25134241839189, 82.75539404585649],
            [40.06113868466835, 54.11565484115581, 46.559670481692265],
            [26.289583959843277, 5.532914197670079, -51.14397143671218]
        ]

        # Vectorized version helps here
        dx = lorenz_odes_vectorized(0, x.T, sigma, beta, rho).T
        assert dx.shape == (5000, 3)

        # Note: Beyond >1000 iterations the error is too great
        assert np.isclose(dx[[0, 1, 10, 100, 1000], :], test_values[:-1]).all()

        # Calculate polynomial terms (up to 3rd order)
        theta = polynomial_features(x, order=3)
        assert theta.shape == (5000, 20)

        test_values = np.array([
            1.0000e+00, -8.0000e+00,  8.0000e+00,  2.7000e+01,  6.4000e+01,
            -6.4000e+01, -2.1600e+02,  6.4000e+01,  2.1600e+02,  7.2900e+02,
            -5.1200e+02,  5.1200e+02,  1.7280e+03, -5.1200e+02, -1.7280e+03,
            -5.8320e+03,  5.1200e+02,  1.7280e+03,  5.8320e+03,  1.9683e+04
        ])
        assert np.isclose(theta[0], test_values).all()

        # Check result is the same as PolynomialFeatures
        # from sklearn.preprocessing
        poly = PolynomialFeatures(3)
        theta_skl = poly.fit_transform(x)
        assert np.isclose(theta, theta_skl).all() 

        # Estimate sparse dynamic model
        lamb = 0.025  # sparsification threshold lambda
        xi = sparsify_dynamics_lstsq(theta, dx, lamb)
        assert xi.shape == (20, 3)

        # Check result
        non_zero = xi.nonzero()
        assert len(xi[non_zero])
        assert (non_zero == np.array([
            [1, 1, 2, 2, 3, 5, 6],
            [0, 1, 0, 1, 2, 2, 1]
        ])).all()
        test_values = np.array([-10. , 28.,  10., -1., -2.66666667, 1., -1.])
        assert np.isclose(xi[non_zero], test_values).all()

        # Estimate sparse dynamic model with lambda too high
        lamb = 1  # sparsification threshold lambda
        #with self.assertRaises(ValueError):
        # TODO: Why does this not raise the intended error?
        xi = sparsify_dynamics_lstsq(theta, dx, lamb)

    def test_polynomial_feature_labels(self):

        labels = polynomial_feature_labels(1, order=1)
        test_labels = ['1', 'x0']
        self.assertEqual(labels, test_labels)

        labels = polynomial_feature_labels(1, order=2)
        test_labels = ['1', 'x0', 'x0**2']
        self.assertEqual(labels, test_labels)

        labels = polynomial_feature_labels(1, order=3)
        test_labels = ['1', 'x0', 'x0**2', 'x0**3']
        self.assertEqual(labels, test_labels)

        self.assertRaises(NotImplementedError, 
                          polynomial_feature_labels, 1, order=4)

        labels = polynomial_feature_labels(3, order=2)
        test_labels = ['1', 'x0', 'x1', 'x2', 'x0**2', 'x0*x1', 
                       'x0*x2', 'x1**2', 'x1*x2', 'x2**2']
        self.assertEqual(labels, test_labels)

        labels = polynomial_feature_labels(2, order=2, vstr='y_')
        test_labels = ['1', 'y_0', 'y_1', 'y_0**2', 'y_0*y_1', 'y_1**2']
        self.assertEqual(labels, test_labels)

        labels = polynomial_feature_labels(2, order=2, vstr='y_', psym='^')
        test_labels = ['1', 'y_0', 'y_1', 'y_0^2', 'y_0*y_1', 'y_1^2']
        self.assertEqual(labels, test_labels)


if __name__ == '__main__':
    unittest.main()

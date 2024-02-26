import os
import unittest
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
import pandas as pd
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from dynopt.sysid.idarx import idar, idarx, idarxct


class AutoRegressiveSysIdTests(unittest.TestCase):

    def setUp(self):

        # Simulation setup
        # Based on homework question from GEL-7017 course
        data_dir = 'tests/data'
        filename = 'TP04_Q3.csv'
        self.id_data = pd.read_csv(os.path.join(data_dir, filename))
        assert self.id_data.shape == (75, 2)

    def test_idar(self):

        # Based on Exercise 3.7 (a) from GEL-7017 course

        na = 1
        y = [0.73, 0.18, -0.36, 0.92, 1.12]
        [p, covp, vres] = idar([na], y)

        # Values from Matlab/Octave
        assert np.array_equal(p, [-0.496853305651074])
        assert np.array_equal(covp, [[0.4070827090837621]])
        assert vres == 1.882309738532408

        # Compare to statsmodels AutoReg model
        model = AutoReg(y, lags=1, trend='n')
        assert model.df_model == 1
        res = model.fit(use_t=True)  # use student's t-distribution
        assert_array_almost_equal([-p], res.params[0])
        assert_array_almost_equal(covp, res.cov_params())

    def test_idarx(self):

        na = 2; nb = 2; nk = 3
        p, covp, vres = idarx([na, nb, nk], self.id_data.u, self.id_data.y)

        assert np.max(np.abs(p - np.array([
            # Output from Octave/Matlab script idarx1.m
            -1.786574,  0.810195, -0.384260,  0.407383
        ]))) < 1e-6

        assert_array_almost_equal(
            p,
            [-1.7865736424,  0.8101952592, -0.3842601544,  0.4073825150]
        )

        assert_array_almost_equal(
            covp,
            np.array([
                [ 5.079376e-03, -5.012962e-03, -2.024573e-03,  2.107575e-03],
                [-5.012962e-03,  5.077026e-03,  2.107429e-03, -2.092299e-03],
                [-2.024573e-03,  2.107429e-03,  3.195526e-01, -3.195038e-01],
                [ 2.107575e-03, -2.092299e-03, -3.195038e-01,  3.243571e-01]
            ])
        )

        assert np.isclose(vres, 2.134978e+01)

        # Compare to statsmodels ARIMA model (which also supports ARMAX)
        y = self.id_data.y[nk:].reset_index(drop=True)
        u = self.id_data.u[:-nk].reset_index(drop=True)  # account for delay
        nd = 0  # order of differences (1=integrator, 0 for ARMA)
        arma_model = ARIMA(y, exog=u, order=(na, nd, nb-1), trend='n').fit()
        assert arma_model.df_model == na + nb + 1
        # TODO: These estimates are not the same
        assert arma_model.params.to_dict() == {
            'u': -0.20132951391935502,
            'ar.L1': 1.7688348711006798,
            'ar.L2': -0.7936183304915454,
            'ma.L1': 0.00598101175778528,
            'sigma2': 0.29631000131195656
        }

    def test_idarxct(self):

        # Based on homework question from GEL-7017 course
        na = 2; nb = 2; nk = 3
        F = np.array([[1.7, 1.7, 1.0, 1.0]])
        G = np.array([[-1.7]])

        p = idarxct([na, nb, nk], self.id_data.u, self.id_data.y, F, G)
        assert np.max(np.abs(p - np.array([
            # Output from Octave/Matlab script idarx1.m
            -1.788782,  0.808797, -0.386398, 0.352373
        ]))) < 1e-4

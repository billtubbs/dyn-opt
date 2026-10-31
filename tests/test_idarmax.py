"""Tests of idarmax with data simulated by scipy.signal.lfilter.

The same systems as in test_idarx_lfilter.py (based on the Matlab
example script ex_mco_procede.m from the GEL-7017 course on system
identification), with coloured noise: the ARMAX model

    A(q^-1) y(k) = B(q^-1) q^-nk u(k) + C(q^-1) e(k)

simulated as in Matlab with

    y = filter([zeros(1, nk) B], [1 A], u) + filter([1 C], [1 A], e)
"""

import unittest

import numpy as np
from scipy.signal import lfilter
from statsmodels.tsa.statespace.sarimax import SARIMAX

from dynopt.sysid.idarmax import idarmax
from dynopt.sysid.idarx import idarx


def simulate_armax(A, B, C, nk, u, e):
    """Simulate the ARMAX model with lfilter, like Matlab's filter()."""
    A = np.concatenate([[1.0], A])
    y = lfilter(np.concatenate([np.zeros(nk), B]), A, u)
    y += lfilter(np.concatenate([[1.0], C]), A, e)
    return y


CASES = [
    # A, B, C, nk
    ([-0.9], [0.1], [0.8], 1),  # ex_mco_procede.m system
    ([-0.9], [0.1], [-0.5, 0.3], 1),
    ([-1.5, 0.7], [1.0, 0.5], [0.5], 1),
    ([-1.5, 0.7], [1.0, 0.5], [-0.6, 0.3], 2),
    ([-1.5, 0.7], [1.0, 0.5], [0.7, 0.2], 0),
]


class IdarmaxTests(unittest.TestCase):
    def test_parameters_recovered(self):
        """The estimates are close to the true values and within 4
        standard errors of them, and the noise variance is recovered."""
        rng = np.random.default_rng(0)
        N = 5000
        sigma_e = 0.2
        for A, B, C, nk in CASES:
            with self.subTest(A=A, B=B, C=C, nk=nk):
                u = np.sign(rng.normal(size=N))  # random binary input
                e = sigma_e * rng.normal(size=N)
                y = simulate_armax(A, B, C, nk, u, e)
                nn = [len(A), len(B), len(C), nk]
                p, covp, vres = idarmax(nn, u, y)

                p_true = np.concatenate([A, B, C])
                std_p = np.sqrt(np.diag(covp))
                assert np.all(np.abs(p - p_true) < 4 * std_p)
                assert np.max(np.abs(p - p_true)) < 0.05

                var_e = vres / (N - len(p))
                assert 0.9 * sigma_e**2 < var_e < 1.1 * sigma_e**2

    def test_arx_is_biased(self):
        """On the same data an ARX model (least squares without the
        noise terms) gives biased estimates of A, which idarmax
        avoids."""
        A, B, C, nk = [-0.9], [0.1], [0.8], 1
        rng = np.random.default_rng(1)
        N = 5000
        u = np.sign(rng.normal(size=N))
        e = 0.2 * rng.normal(size=N)
        y = simulate_armax(A, B, C, nk, u, e)

        p_arx, covp_arx, _ = idarx([1, 1, nk], u, y)
        p, covp, _ = idarmax([1, 1, 1, nk], u, y)

        # ARX: a1 more than 5 standard errors from the true value
        assert abs(p_arx[0] - A[0]) > 5 * np.sqrt(covp_arx[0, 0])
        # ARMAX: within 4 standard errors
        assert abs(p[0] - A[0]) < 4 * np.sqrt(covp[0, 0])

    def test_ex_mco_procede_input(self):
        """The step input of ex_mco_procede.m, repeated to give enough
        data to estimate the noise model."""
        step = 10 * np.concatenate([np.zeros(10), np.ones(100), np.zeros(150)])
        u = np.tile(step, 10)
        A, B, C, nk = [-0.9], [0.1], [0.8], 1
        rng = np.random.default_rng(2)
        e = rng.normal(size=len(u))  # e = sqrt(1)*randn(M,1) in the script
        y = simulate_armax(A, B, C, nk, u, e)

        p, covp, _ = idarmax([1, 1, 1, nk], u, y)
        p_true = np.concatenate([A, B, C])
        assert np.all(np.abs(p - p_true) < 4 * np.sqrt(np.diag(covp)))

    def test_compare_statsmodels(self):
        """Compare with statsmodels' SARIMAX fitted as an ARMAX model:
        MA(nc) errors, order (0, 0, nc), with the ARX regressors
        -y(k-1), ..., -y(k-na), u(k-nk), ..., u(k-nk-nb+1) as exogenous
        variables. This is the same model, fitted by maximum likelihood
        (the same criterion for Gaussian noise), so the estimates and
        their standard errors should agree, apart from the treatment of
        the initial conditions."""
        A, B, C, nk = [-1.5, 0.7], [1.0, 0.5], [-0.6, 0.3], 2
        na, nb, nc = len(A), len(B), len(C)
        rng = np.random.default_rng(0)
        N = 5000
        u = np.sign(rng.normal(size=N))
        e = 0.2 * rng.normal(size=N)
        y = simulate_armax(A, B, C, nk, u, e)

        p, covp, _ = idarmax([na, nb, nc, nk], u, y)
        std_p = np.sqrt(np.diag(covp))

        k = np.arange(max(na, nb + nk - 1), N)
        X = np.column_stack(
            [-y[k - i] for i in range(1, na + 1)] + [u[k - nk - j] for j in range(nb)]
        )
        res = SARIMAX(y[k], exog=X, order=(0, 0, nc), trend="n").fit(disp=False)
        p_sm = np.asarray(res.params)[:-1]  # last one is the noise variance
        std_sm = np.asarray(res.bse)[:-1]

        assert np.all(np.abs(p - p_sm) < 0.25 * std_p)
        assert np.all(np.abs(std_p / std_sm - 1) < 0.1)


if __name__ == "__main__":
    unittest.main()

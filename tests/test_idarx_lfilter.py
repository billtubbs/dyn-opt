"""Tests of idarx with data simulated by scipy.signal.lfilter.

Based on the Matlab example script ex_mco_procede.m from the GEL-7017
course on system identification: simulate an ARX system with filter(),
estimate its parameters by ordinary least squares and check that they
are recovered, without noise (exactly) and with noise (within the
estimated standard errors).

In Matlab, the ARX model A(q^-1) y(k) = B(q^-1) q^-nk u(k) + e(k) is
simulated with

    y = filter([zeros(1, nk) B], [1 A], u) + filter(1, [1 A], e)

and lfilter() is the same function in Python.
"""

import unittest

import numpy as np
from scipy.signal import lfilter

from dynopt.sysid.idarx import idarx


def simulate_arx(A, B, nk, u, e=None):
    """Simulate the ARX model with lfilter, like Matlab's filter()."""
    A = np.concatenate([[1.0], A])
    y = lfilter(np.concatenate([np.zeros(nk), B]), A, u)
    if e is not None:
        y += lfilter([1.0], A, e)
    return y


class IdarxLfilterTests(unittest.TestCase):
    def test_ex_mco_procede(self):
        """The example in ex_mco_procede.m: a step input, no noise."""
        u = 10 * np.concatenate([np.zeros(10), np.ones(100), np.zeros(150)])
        M = len(u)
        y = simulate_arx([-0.9], [0.1], 1, u)

        p, covp, vres = idarx([1, 1, 1], u, y)
        assert np.max(np.abs(p - [-0.9, 0.1])) < 1e-10
        assert vres < 1e-20

        # Same as the regression in ex_mco_procede.m
        phi = np.column_stack([-y[: M - 1], u[: M - 1]])
        theta = np.linalg.solve(phi.T @ phi, phi.T @ y[1:])
        assert np.max(np.abs(p - theta)) < 1e-12

    def test_structures_no_noise(self):
        """Various model structures, including nk = 0 (direct
        transmission) and nk > 1, are recovered exactly."""
        cases = [
            ([-0.9], [0.1], 1),
            ([-0.9], [0.1], 0),
            ([-0.9], [0.1], 3),
            ([-1.5, 0.7], [1.0, 0.5], 1),
            ([-1.5, 0.7], [1.0, 0.5], 2),
            ([-0.5], [1.0, 0.5, 0.25], 1),
            ([-1.5, 0.7, -0.1], [0.2, 0.1], 0),
        ]
        rng = np.random.default_rng(0)
        u = rng.normal(size=200)
        for A, B, nk in cases:
            with self.subTest(A=A, B=B, nk=nk):
                y = simulate_arx(A, B, nk, u)
                p, _, _ = idarx([len(A), len(B), nk], u, y)
                assert np.max(np.abs(p - np.concatenate([A, B]))) < 1e-8

    def test_ex_mco_procede_with_noise(self):
        """The example in ex_mco_procede.m with the noise term switched
        on (e = sqrt(1)*randn(M,1) in the script)."""
        u = 10 * np.concatenate([np.zeros(10), np.ones(100), np.zeros(150)])
        p_true = np.array([-0.9, 0.1])
        rng = np.random.default_rng(1)
        e = rng.normal(size=len(u))
        y = simulate_arx([-0.9], [0.1], 1, u, e)

        p, covp, vres = idarx([1, 1, 1], u, y)

        # Within 4 standard errors of the true values
        std_p = np.sqrt(np.diag(covp))
        assert np.all(np.abs(p - p_true) < 4 * std_p)

        # Noise variance estimate close to 1
        var_e = vres / (len(u) - 1 - len(p))
        assert 0.8 < var_e < 1.2

    def test_structures_with_noise(self):
        """With measurement noise and a random input, the estimates are
        close to the true values and within 4 standard errors of them.
        With ARX (equation error) noise, least squares is unbiased."""
        cases = [
            ([-0.9], [0.1], 1),
            ([-1.5, 0.7], [1.0, 0.5], 1),
            ([-1.5, 0.7], [1.0, 0.5], 2),
            ([-0.5], [1.0, 0.5, 0.25], 0),
        ]
        rng = np.random.default_rng(2)
        N = 2000
        sigma_e = 0.1
        for A, B, nk in cases:
            with self.subTest(A=A, B=B, nk=nk):
                u = np.sign(rng.normal(size=N))  # random binary input
                e = sigma_e * rng.normal(size=N)
                y = simulate_arx(A, B, nk, u, e)
                p, covp, vres = idarx([len(A), len(B), nk], u, y)

                p_true = np.concatenate([A, B])
                std_p = np.sqrt(np.diag(covp))
                assert np.all(np.abs(p - p_true) < 4 * std_p)
                assert np.max(np.abs(p - p_true)) < 0.05

                var_e = vres / (N - len(p))
                assert 0.8 * sigma_e**2 < var_e < 1.2 * sigma_e**2


if __name__ == "__main__":
    unittest.main()

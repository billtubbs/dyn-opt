"""Functions to estimate the parameters of single-input, single-output
AR and ARX models from time series data using least-squares.

The polynomial function for the ARX model is defined:

    A(q^-1) y(k) = B(q^-1) q^-nk u(k) + e(k)

where

    A(q^-1) = 1 + a_1 q^-1 + ... + a_na q^-na
    B(q^-1) = b_1 + b_2 q^-1 + ... + b_nb q^-(nb-1)
    q^-1 is the backwards shift operator
    y(k) is the output at time instant k
    u(k) is the input at time instant k
    e(k) is the error (white noise) at time instant k

The AR model has no exogenous input (i.e. nb = 0).

The parameter vector is defined:

    p = [a1, ..., a_na, b1, ..., b_nb].

The noise-free output is y = filter([zeros(1, nk), B], [1, A], u) in
Matlab/Octave (scipy.signal.lfilter in Python). nk = 0 means u(k) acts
on y(k) directly.

Comparing with Matlab and Octave arx():
- Matlab (System Identification Toolbox): same convention. nk is the
  number of leading zeros of B, so arx(data, [na nb nk]) estimates the
  same model as idarx([na, nb, nk], u, y).
- Octave (control package, checked with version 4.1.1): nk is one less
  than here. Octave's arx() always delays the input one step and adds
  nk steps to that, so arx(dat, 'na', na, 'nb', nb, 'nk', nk) estimates
  the same model as idarx([na, nb, nk + 1], u, y). Its default, nk = 0,
  is the usual one-step delay, and it can't estimate a model with
  direct transmission (nk = 0 here). Its documentation doesn't say so.

Contents:
- idar(nn, y): AR model, structure [na]; returns the parameters, their
  covariance matrix and the sum of squared residuals.
- idarx(nn, u, y): ARX model, structure [na, nb, nk]; returns the same.
- idarxct(nn, u, y, F, G): ARX model with linear equality constraints
  F p = G on the parameters; returns the parameters.
"""

import numpy as np
import scipy


def idar(nn, y):
    """Function to estimate the parameters of a dynamic AR model
    from time series data.  Also computes the covariance matrix
    (covp) and the sum of the squared residuals (vres).

    Arguments:
        nn : list or array
            Structure of the model to be estimated: [na].
        y : array (m, )
            Output time series.

    """

    # AR model structure
    na = nn[0]
    assert na > 0
    m = len(y)

    # Construct data matrices
    Y = np.flip(scipy.linalg.hankel(y[:na], y[na - 1 :]).T, axis=1)[
        : m - na + 1, :
    ]

    # Phi, Y matrices
    n = m - na
    phi = -Y[-n - 1 : -1, :]
    Y = y[-n:]

    # Estimate parameters using ordinary least squares
    # p = np.linalg.inv(phi.T @ phi) @ phi.T @ Y
    p = np.linalg.solve(phi.T @ phi, phi.T @ Y)

    # Residuals
    errors = Y - phi @ p

    # Sum-squared of residuals (minimization criterion)
    vres = errors.T @ errors

    # Estimate of the white noise variance
    var_e = 1 / (n - len(p)) * vres

    # Covariance matrix of parameter estimates
    covp = var_e * np.linalg.inv(phi.T @ phi)

    return p, covp, vres


def idarx(nn, u, y):
    """Function to estimate the parameters of a dynamic ARX model
    from time series data.  Also computes the covariance matrix
    (covp) and the sum of the squared residuals (vres).

    Arguments:
        nn : list or array
            Structure of the model to be estimated: [na, nb, nk].
        u : array (m, )
            Input time series.
        y : array (m, )
            Output time series.

    """

    # ARX model structure
    na = nn[0]
    assert na > 0
    nb = nn[1]
    assert nb > 0
    nk = nn[2]
    assert nk >= 0
    m = len(u)
    assert len(y) == m

    # Construct data matrices
    U = np.flip(scipy.linalg.hankel(u[:nb], u[nb - 1 :]).T, axis=1)
    Y = np.flip(scipy.linalg.hankel(y[:na], y[na - 1 :]).T, axis=1)

    # Phi, Y matrices
    n = m - max(na, nb + nk - 1)
    NY = Y.shape[0]  # = m - na + 1
    NU = U.shape[0]  # = m - nb + 1
    phi = np.concatenate(
        [-Y[NY - n - 1 : NY - 1, :], U[NU - n - nk : NU - nk, :]], axis=1
    )
    Y = y[-n:]

    # Estimate parameters using ordinary least squares
    # p = np.linalg.inv(phi.T @ phi) @ phi.T @ Y
    p = np.linalg.solve(phi.T @ phi, phi.T @ Y)

    # Residuals
    errors = Y - phi @ p

    # Sum-squared of residuals (minimization criterion)
    vres = errors.T @ errors

    # Estimate of the white noise variance
    var_e = 1 / (n - len(p)) * vres

    # Covariance matrix of parameter estimates
    covp = var_e * np.linalg.inv(phi.T @ phi)

    return p, covp, vres


def idarxct(nn, u, y, F, G):
    """Function to estimate the parameters of a dynamic ARX model
    from time series data.  Also computes the covariance matrix
    (covp) and the sum of the squared residuals (vres).

    Arguments:
        nn : list or array
            Structure of the model to be estimated: [na, nb, nk].
        u : array (m, )
            Input time series.
        y : array (m, )
            Output time series.
        F : array (1, na+nb)
            Constraint equation coefficients.
        G : array (na+nb, 1)
            Constraint equation constants.

    """

    # ARX model structure
    na = nn[0]
    assert na > 0
    nb = nn[1]
    assert nb > 0
    nk = nn[2]
    assert nk >= 0
    m = len(u)
    assert len(y) == m

    # Construct data matrices
    U = np.flip(scipy.linalg.hankel(u[:nb], u[nb - 1 :]).T, axis=1)
    Y = np.flip(scipy.linalg.hankel(y[:na], y[na - 1 :]).T, axis=1)

    # Phi, Y matrices
    n = m - max(na, nb + nk - 1)
    NY = Y.shape[0]  # = m - na + 1
    NU = U.shape[0]  # = m - nb + 1
    phi = np.concatenate(
        [-Y[NY - n - 1 : NY - 1, :], U[NU - n - nk : NU - nk, :]], axis=1
    )
    Y = y[-n:]

    # Estimate parameters using ordinary least squares
    # with explicit constraints (Lagrangian)
    phi_t = np.transpose(phi)
    phi_t_phi = phi_t @ phi
    solve_phi_t_phi_div_phi_t_Y = np.linalg.solve(
        phi_t_phi, phi_t @ Y
    ).reshape(-1, 1)
    p = solve_phi_t_phi_div_phi_t_Y - np.linalg.solve(
        phi_t_phi,
        F.T
        @ np.linalg.solve(
            F @ np.linalg.solve(phi_t_phi, F.T),
            F @ solve_phi_t_phi_div_phi_t_Y - G,
        ),
    )
    assert p.shape[1] == 1
    p = p.reshape(-1)

    # Matlab expression:
    # p = phi' * phi \ (phi' * Y) - phi' * phi \ (...
    #    F' * ((F * ((phi' * phi) \ F')) \ (F * ((phi' * phi) \ (phi' * Y)) - G)) ...
    # );

    return p

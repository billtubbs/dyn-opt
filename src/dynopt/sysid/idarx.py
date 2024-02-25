import numpy as np
import scipy


def idarx(nn, u, y):
    """Function to estimate the parameters of a dynamic ARX model
    from time series data.  Also computes the covariance matrix
    (covp) and the sum of the squared residuals (Vres).

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
    U = np.flip(scipy.linalg.hankel(u[:nb], u[nb - 1:]).T, axis=1)[:m - nb + 1, :]
    Y = np.flip(scipy.linalg.hankel(y[:na], y[na - 1:]).T, axis=1)[:m - na + 1, :]

    # Phi, Y matrices
    n = m - max(na, nb + nk - 1)
    phi = np.concatenate([-Y[-n-1:-1, :], U[-n-nk:-nk, :]], axis=1)
    Y = y[-n:]

    # Estimate parameters using ordinary least squares
    # p = np.linalg.inv(phi.T @ phi) @ phi.T @ Y
    p = np.linalg.solve(phi.T @ phi, phi.T @ Y)

    # Residuals
    errors = Y - phi @ p

    # Sum-squared of residuals (minimization criterion)
    Vres = errors.T @ errors

    # Estimate of the white noise variance
    var_e = 1 / (n - len(p)) * Vres

    # Covariance matrix of parameter estimates
    covp = var_e * np.linalg.inv(phi.T @ phi)

    return p, covp, Vres


def idarxct(nn, u, y, F, G):

    """Function to estimate the parameters of a dynamic ARX model
    from time series data.  Also computes the covariance matrix
    (covp) and the sum of the squared residuals (Vres).

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
    U = np.flip(scipy.linalg.hankel(u[:nb], u[nb - 1:]).T, axis=1)[:m - nb + 1, :]
    Y = np.flip(scipy.linalg.hankel(y[:na], y[na - 1:]).T, axis=1)[:m - na + 1, :]

    # Phi, Y matrices
    n = m - max(na, nb + nk - 1)
    phi = np.concatenate([-Y[-n-1:-1, :], U[-n-nk:-nk, :]], axis=1)
    Y = y[-n:]

    # Estimate parameters using ordinary least squares
    # with explicit constraints (Lagrangian)
    phi_t = np.transpose(phi)
    phi_t_phi = phi_t @ phi
    solve_phi_t_phi_div_phi_t_Y = np.linalg.solve(phi_t_phi, phi_t @ Y).reshape(-1, 1)
    p = (solve_phi_t_phi_div_phi_t_Y
        - np.linalg.solve(
            phi_t_phi,
            F.T @ np.linalg.solve(
                F @ np.linalg.solve(phi_t_phi, F.T), 
                F @ solve_phi_t_phi_div_phi_t_Y - G
            )
        )
    )
    assert p.shape[1] == 1
    p = p.reshape(-1)

    # Matlab expression:
    #p = phi' * phi \ (phi' * Y) - phi' * phi \ (...
    #    F' * ((F * ((phi' * phi) \ F')) \ (F * ((phi' * phi) \ (phi' * Y)) - G)) ...
    #);

    return p

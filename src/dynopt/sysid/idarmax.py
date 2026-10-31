"""Estimation of single-input, single-output ARMAX models from time
series data by extended least squares.

The polynomial function for the ARMAX model is defined:

    A(q^-1) y(k) = B(q^-1) q^-nk u(k) + C(q^-1) e(k)

where

    A(q^-1) = 1 + a_1 q^-1 + ... + a_na q^-na
    B(q^-1) = b_1 + b_2 q^-1 + ... + b_nb q^-(nb-1)
    C(q^-1) = 1 + c_1 q^-1 + ... + c_nc q^-nc
    q^-1 is the backwards shift operator
    e(k) is white noise

In Matlab/Octave the model is simulated with

    y = filter([zeros(1, nk), B], [1, A], u) + filter([1, C], [1, A], e)

(scipy.signal.lfilter in Python). The A, B and nk conventions are the
same as in idarx.py (and Matlab's arx() and armax(); see the notes there
about Octave's arx()).

The noise terms e(k-1), ..., e(k-nc) are not measured, so the model
can't be estimated by ordinary least squares directly. Extended least
squares (pseudo-linear regression) estimates them with the prediction
errors of the current model and repeats the least-squares fit:

1. Fit an ARX model [na, nb, nk] and take its residuals as e(k).
2. Fit y(k) by least squares on -y(k-1), ..., u(k-nk), ..., and the
   estimated e(k-1), ..., e(k-nc).
3. Compute the prediction errors of the new model,
   e(k) = (A(q^-1) y(k) - B(q^-1) q^-nk u(k)) / C(q^-1),
   and repeat from 2 until the parameters stop changing.
4. Starting from the extended least-squares estimates, minimize the sum
   of squared prediction errors by Gauss-Newton iterations (the
   prediction error method, as in Matlab's armax()).

Extended least squares alone gives consistent estimates (under a
condition on C that holds for most noise models), but with larger
variances than the prediction error method, and no simple formula for
their covariance. Step 4 gives the prediction error estimates, whose
asymptotic covariance is var_e * inv(psi' psi), where psi(k) are the
regressors filtered by 1/C(q^-1).

The parameter vector is defined:

    p = [a1, ..., a_na, b1, ..., b_nb, c1, ..., c_nc].

Contents:
- idarmax(nn, u, y): ARMAX model, structure [na, nb, nc, nk]; returns
  the parameters, their covariance matrix and the sum of squared
  prediction errors.
"""

import warnings

import numpy as np
from scipy.signal import lfilter


def _regressors(y, u, e, na, nb, nc, nk):
    """Return the regression matrix phi and target vector Y for the
    time steps k at which all the regressors are available."""
    m = len(y)
    k0 = max(na, nb + nk - 1, nc)
    k = np.arange(k0, m)
    cols = (
        [-y[k - i] for i in range(1, na + 1)]
        + [u[k - nk - j] for j in range(nb)]
        + [e[k - i] for i in range(1, nc + 1)]
    )
    return np.column_stack(cols), y[k]


def _prediction_errors(p, y, u, na, nb, nc, nk):
    """Prediction errors e(k) = (A y(k) - B q^-nk u(k)) / C, from zero
    initial conditions."""
    A = np.concatenate([[1.0], p[:na]])
    B = np.concatenate([np.zeros(nk), p[na : na + nb]])
    C = np.concatenate([[1.0], p[na + nb :]])
    v = lfilter(A, [1.0], y) - lfilter(B, [1.0], u)
    return lfilter([1.0], C, v)


def idarmax(nn, u, y, max_iter=100, tol=1e-8):
    """Function to estimate the parameters of a dynamic ARMAX model
    from time series data by extended least squares followed by
    Gauss-Newton iterations (prediction error method).  Also computes
    the covariance matrix (covp) and the sum of the squared prediction
    errors (vres).

    Arguments:
        nn : list or array
            Structure of the model to be estimated: [na, nb, nc, nk].
        u : array (m, )
            Input time series.
        y : array (m, )
            Output time series.
        max_iter : int
            Maximum number of iterations of each of extended least
            squares and Gauss-Newton.
        tol : float
            The iterations stop when no parameter changes by more than
            tol.

    Returns:
        p : array (na + nb + nc, )
            Parameters [a1, ..., a_na, b1, ..., b_nb, c1, ..., c_nc].
        covp : array (na + nb + nc, na + nb + nc)
            Asymptotic covariance matrix of the parameter estimates.
        vres : float
            Sum of the squared prediction errors.

    Warns if the iterations don't converge or if the estimated C
    polynomial is unstable (then the prediction errors grow and the
    estimates aren't reliable).
    """

    # ARMAX model structure
    na, nb, nc, nk = (int(x) for x in nn)
    assert na > 0
    assert nb > 0
    assert nc > 0
    assert nk >= 0
    u = np.asarray(u, dtype=float)
    y = np.asarray(y, dtype=float)
    m = len(u)
    assert len(y) == m

    # 1. ARX fit (no noise terms) and its residuals
    phi, Y = _regressors(y, u, np.zeros(m), na, nb, 0, nk)
    p_arx = np.linalg.lstsq(phi, Y, rcond=None)[0]
    p = np.concatenate([p_arx, np.zeros(nc)])
    e = _prediction_errors(p, y, u, na, nb, nc, nk)

    # 2-3. Least squares with the estimated noise terms, repeated
    for _ in range(max_iter):
        phi, Y = _regressors(y, u, e, na, nb, nc, nk)
        p_new = np.linalg.lstsq(phi, Y, rcond=None)[0]
        converged = np.max(np.abs(p_new - p)) < tol
        p = p_new
        e = _prediction_errors(p, y, u, na, nb, nc, nk)
        if converged:
            break
    else:
        warnings.warn(
            f"idarmax: extended least squares didn't converge in {max_iter} iterations",
            RuntimeWarning,
        )

    # 4. Gauss-Newton iterations minimizing the sum of squared prediction
    # errors. The gradient of e(k) with respect to p is -psi(k), the
    # regressors phi(k) filtered by 1/C, so each step solves psi dp = e
    # by least squares (halving the step if it doesn't reduce the sum).
    k0 = max(na, nb + nk - 1, nc)
    vres = e[k0:] @ e[k0:]
    for _ in range(max_iter):
        phi, _ = _regressors(y, u, e, na, nb, nc, nk)
        psi = lfilter([1.0], np.r_[1.0, p[na + nb :]], phi, axis=0)
        dp = np.linalg.lstsq(psi, e[k0:], rcond=None)[0]
        step = 1.0
        while step > 1e-6:
            p_new = p + step * dp
            e_new = _prediction_errors(p_new, y, u, na, nb, nc, nk)
            vres_new = e_new[k0:] @ e_new[k0:]
            if vres_new <= vres:
                break
            step /= 2
        else:
            break  # no step reduces the sum: at the minimum
        converged = np.max(np.abs(p_new - p)) < tol
        p, e, vres = p_new, e_new, vres_new
        if converged:
            break
    else:
        warnings.warn(
            f"idarmax: Gauss-Newton didn't converge in {max_iter} iterations",
            RuntimeWarning,
        )

    if np.any(np.abs(np.roots(np.r_[1.0, p[na + nb :]])) >= 1):
        warnings.warn("idarmax: estimated C polynomial is unstable", RuntimeWarning)

    # Estimate of the white noise variance
    n = m - k0
    var_e = 1 / (n - len(p)) * vres

    # Asymptotic covariance matrix of the parameter estimates
    phi, _ = _regressors(y, u, e, na, nb, nc, nk)
    psi = lfilter([1.0], np.r_[1.0, p[na + nb :]], phi, axis=0)
    covp = var_e * np.linalg.inv(psi.T @ psi)

    return p, covp, vres

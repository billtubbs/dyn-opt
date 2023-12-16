"""
Implementation of SINDy algorithm based on code by Steven L.
Brunton.  See Paper, 'Discovering Governing Equations from Data:
Sparse Identification of Nonlinear Dynamical Systems' by S. L.
Brunton, J. L. Proctor, and J. N. Kutz.
Original code available from: https://www.eigensteve.com
"""
import numpy as np


def sparsify_dynamics_lstsq(theta, y, lamb, max_iter=20):
    """SINDy algorithm to find sparse polynomial model
    of dynamics using ordinary least-squares.
    """

    n_out = y.shape[1]
    # Initial guess: Least-squares
    xi = np.linalg.lstsq(theta, y, rcond=None)[0]

    for _ in range(max_iter):
        # Find large coefficients above threshold
        big_coefs = np.abs(xi) >= lamb

        # Set small coefficients to zero
        xi[~big_coefs] = 0

        # Check that not all coefficients are zero for any output
        if np.any(np.count_nonzero(big_coefs, axis=0) == 0):
            raise ValueError(
                "Sparsity parameter is too big ({}) and eliminated all "
                "coefficients".format(lamb)
            )

        # Regress dynamics onto remaining terms to find sparse xi
        for i in range(n_out):  # n is state dimension
            coefs_i = big_coefs[:, i]
            xi[coefs_i, i] = np.linalg.lstsq(theta[:, coefs_i], y[:, i],
                                             rcond=None)[0]

    return xi

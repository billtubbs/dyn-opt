"""Python module polynomials.py

Functions for polynomial function manipulation for discrete linear
dynamical systems modelling and control applications.

Contents:

 - sub_poly(a, b)
 - diophantine(C, D)
 - diophantine_recursive(C, D, hp)

1. Polynomial division by recursive solution of the Diophantine equation

    F, M = diophantine_recursive(C, D, hp) solves the following
    diophantine equation for j = 1 to hp:

          C / D = F + z^(-j) * M / D

    where C, D, F, and M are polynomials in decreasing powers of z^-1.

    Example:

    Solve the following for the 2-step-ahead prediction calculation:

        (1 - 0.2z^-1) / (1 - 0.8z^-1) = F + z^(-j) * M / (1 - 0.8z^-1)

    >>> F, M = diophantine_recursive([1, -0.2], [1, -0.8], 2)

    We obtain the results for 1-step ahead and 2-steps ahead:

    >>> F
    array([[1. , 0. ],
        [1. , 0.6]])
    >>> M
    array([[0.6 ],
        [0.48]])

    Therefore:
        (1 - 0.2z^-1) / (1 - 0.8z^-1) = 1 + 0.6z^-1 / (1 - 0.8z^-1)

    and

        (1 - 0.2z^-1) / (1 - 0.8z^-1)
            = 1 + 0.6z^-1 + 0.48z^-2 / (1 - 0.8z^-1)

"""
# Based on the MATLAB script Diophantine_equation.m by André Desbiens
# and Kaddour Najim, Copyright (c), April 1993.
# Translated to Python by Bill Tubbs, July 2020.

import numpy as np


def sub_poly(a, b):
    """Returns the subtraction of two polynomials.

        p = a - b 
    
    where the polynomials `p`, `a`, and `b` are 
    represented as lists or arrays containing the 
    coefficients in decreasing powers of z^-1.

    `a` and `b` do not need to be the same length.

    Parameters:
        a : list or array
            Coefficients of a
        b : list or array
            Coefficients of b

    Returns:
        p : array
            Coefficients of p

    Example:
    >>> sub_poly([1, 0.2], [1, 0.1, 0.8])
    array([ 0. ,  0.1, -0.8])
    """
    a, b = np.array(a), np.array(b)
    la, lb = len(a), len(b)
    if la == lb:
        return a - b
    else:
        p = np.zeros(max((la, lb)))  # Always returns dtype float
        p[:la] = a
        p[:lb] = p[:lb] - b
        return p


def diophantine(C, D):
    """Solves the following diophantine equation:
    
        C / D = quotient + remainder / D
    
    where `C`, `D`, `quotient` and `remainder` are 
    polynomials represented as lists or arrays containing
    the coefficients in decreasing powers of z^-1.
    """
    D = np.array(D)
    quotient = C[0] / D[0]
    remainder = sub_poly(C, quotient * D)
    assert remainder[0] == 0
    return quotient, remainder[1:]


def diophantine_recursive(C, D, hp):
    """Recursive solution of the Diophantine equation.

    Returns the polynomials `F` and `M` that satisfy

        C / D = F + z^(-j) * M / D

    for j = 1 to `hp`, where `C`, `D`, `F`, and `M` are
    polynomials represented as lists or arrays containing
    the coefficients in decreasing powers of z^-1.

    Arguments:
        C : array_like
            Coefficients of `C`.
        D : array_like
            Coefficients of `D`.
        Hp : int
            Number of iterations of recursion.

    Returns:
        F : ndarray (2-D)
            Coefficients of `F` in rows for j = 1 to `hp`.
        M : ndarray (2-D)
            Coefficients of `M` in rows for j = 1 to `hp`.

    Example:
    >>> # Solve for 2-steps ahead:
    >>> # (1 - 0.2z^-1) / (1 - 0.8z^-1) = 1 + 0.6z^-1 + 0.48z^-2 / (1 - 0.8z^-1)
    >>> F, M = diophantine_recursive([1, -0.2], [1, -0.8], 2)
    >>> F
    array([[1. , 0. ],
        [1. , 0.6]])
    >>> M
    array([[0.6 ],
        [0.48]])
    """
    C = np.array(C)
    D = np.array(D)
    F = np.zeros((hp, hp))
    M = np.zeros((hp, max(len(C), len(D)) - 1))
    quotients = []
    numerator = C
    for j in range(hp):
        q, r = diophantine(numerator, D)
        quotients.append(q)
        F[j,:j+1] = quotients
        M[j,:] = r
        numerator = r
    
    return F, M


def test_sub_poly():
    """Unit test sub_poly (compared to MATLAB outputs)"""
    assert np.array_equal(sub_poly([8], [5]), [3])
    assert np.array_equal(sub_poly(np.array([8]), np.array([5])), [3])
    a = [1, 2, 3]
    b = [1, 2]
    assert np.array_equal(sub_poly(a, a), [0., 0., 0.])
    assert np.array_equal(sub_poly(b, b), [0., 0.])
    assert np.array_equal(sub_poly(a, b), [0., 0., 3.])
    assert np.array_equal(sub_poly(b, a), [0., 0., -3.])
    a = np.array([0.5, -0.6, 0.7, -0.8])  # dtype: float
    assert np.array_equal(sub_poly(a, a), [0., 0., 0., 0.])
    assert np.array_equal(sub_poly(a, b), [-0.5, -2.6,  0.7, -0.8])
    assert np.array_equal(sub_poly(b, a), [ 0.5, 2.6,  -0.7, 0.8])


def test_diophantine():
    A = np.array([0.18, 1.14, 1.43, 0.48, 0.49])
    quotient, remainder = diophantine(A, A)
    assert quotient == 1.0
    assert np.array_equal(remainder, np.zeros(len(A) - 1))
    quotient, remainder = diophantine([0], [1, -0.8])
    assert quotient == 0.0
    assert np.array_equal(remainder, [0.0])
    quotient, remainder = diophantine([4], [1, -0.8])
    assert quotient == 4.0
    assert np.array_equal(remainder, [3.2])


def test_diophantine_recursive():
    """Unit test diophantine_recursive (compared to MATLAB outputs)"""
    # Test case 1
    F, M = diophantine_recursive([1], [2], 1)  
    assert np.array_equal(F, np.array([[0.5]]))
    assert np.array_equal(M, np.array([[]]))
    # Test case 2
    C, D, hp = [1, -0.2], [1, -0.8], 2
    F, M = diophantine_recursive(C, D, hp)
    F_test = np.array([
        [1.0, 0.0],
        [1.0, 0.6]
    ])
    M_test = np.array([
        [0.60],
        [0.48]
    ])
    assert np.allclose(F, F_test)
    assert np.allclose(M, M_test)
    # Test case 3
    C, D, hp = [1], [1, -1.8, 0.8], 3
    F, M = diophantine_recursive(C, D, hp)
    F_test = np.array([
        [1.0000,         0,         0],
        [1.0000,    1.8000,         0],
        [1.0000,    1.8000,    2.4400]
    ])
    M_test = np.array([
        [1.8000,   -0.8000],
        [2.4400,   -1.4400],
        [2.9520,   -1.9520]
    ])
    assert np.allclose(F, F_test)
    assert np.allclose(M, M_test)
    # Test case 4
    C, D, hp = [1, 0.6, -0.2], [1, 0.6, -0.2], 3
    F, M = diophantine_recursive(C, D, hp)
    F_test = np.array([
        [1., 0. , 0. ],
        [1., 0. , 0. ],
        [1., 0. , 0. ]
    ])
    M_test = np.zeros((3, 2))
    assert np.allclose(F, F_test)
    assert np.allclose(M, M_test)


if __name__ == '__main__':
    # Run unit tests
    test_sub_poly()
    test_diophantine()
    test_diophantine_recursive()
